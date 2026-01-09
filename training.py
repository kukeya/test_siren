'''Implements a generic training loop.
'''

import torch
import utils
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import shutil


def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
          summary_fn, val_dataloader=None, double_precision=False, clip_grad=False, use_lbfgs=False, loss_schedules=None, use_lr_decay=False):

    optim = torch.optim.Adam(lr=lr, params=model.parameters())

    scheduler = None #[新增]学习率衰减

    if use_lr_decay:
        # 学习率衰减策略：每经过总 Epoch 数的 1/4，学习率乘以 0.5
        # 例如：总共 2000 epochs，则每 500 epochs 衰减一次
        ratios = [0.4, 0.7, 0.9] # 调整里程碑，让前期多训练一会儿
        milestones = [int(epochs * r) for r in ratios]

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optim, 
            milestones=milestones,
            gamma=0.5
        )


    # copy settings from Raissi et al. (2019) and here 
    # https://github.com/maziarraissi/PINNs
    if use_lbfgs:
        optim = torch.optim.LBFGS(lr=lr, params=model.parameters(), max_iter=50000, max_eval=50000,
                                  history_size=50, line_search_fn='strong_wolfe')

    if os.path.exists(model_dir):
        # val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
        # if val == 'y':
        shutil.rmtree(model_dir)

    os.makedirs(model_dir)

    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            if not epoch % epochs_til_checkpoint and epoch:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
                           np.array(train_losses))
                
            # epoch_loss_dict = {} # [新增]用于累积每一项 loss

            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()
            
                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}

                if double_precision:
                    model_input = {key: value.double() for key, value in model_input.items()}
                    gt = {key: value.double() for key, value in gt.items()}

                if use_lbfgs:
                    def closure():
                        optim.zero_grad()
                        model_output = model(model_input)
                        losses = loss_fn(model_output, gt)
                        train_loss = 0.
                        for loss_name, loss in losses.items():
                            train_loss += loss.mean() 
                        train_loss.backward()
                        return train_loss
                    optim.step(closure)

                
                model_output = model(model_input)
                losses = loss_fn(model_output, gt)

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    # --- [新增] 累积每一项 loss ---
                    # if loss_name not in epoch_loss_dict:
                    #     epoch_loss_dict[loss_name] = 0.0
                    # epoch_loss_dict[loss_name] += single_loss.item()
                    # -----------------------------

                    if loss_schedules is not None and loss_name in loss_schedules:
                        writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                        single_loss *= loss_schedules[loss_name](total_steps)

                    writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss

                if isinstance(train_loss, torch.Tensor):
                    train_losses.append(train_loss.item())
                else:
                    train_losses.append(train_loss)
                writer.add_scalar("total_train_loss", train_loss, total_steps)

                if not total_steps % steps_til_summary:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_current.pth'))
                    summary_fn(model, model_input, gt, model_output, writer, total_steps)

                if not use_lbfgs:
                    optim.zero_grad()
                    # ensure train_loss is a tensor before calling backward
                    if not isinstance(train_loss, torch.Tensor):
                        param = next(model.parameters())
                        train_loss = torch.tensor(train_loss, device=param.device, dtype=param.dtype)
                    train_loss.backward()

                    if clip_grad:
                        if isinstance(clip_grad, bool):
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                    optim.step()

                pbar.update(1)

                if not total_steps % steps_til_summary:
                    tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))
                    # loss_str = ", ".join([f"{k}: {v / (step + 1):.6f}" for k, v in epoch_loss_dict.items()])
                    # tqdm.write("Epoch %d, Total: %0.6f | %s" % (epoch, train_loss, loss_str))
                    # 重置累积字典，避免数值过大且方便观察当前阶段
                    # epoch_loss_dict = {} 
                    # ---------------------------
                    
                    current_lr = optim.param_groups[0]['lr'] #[新增]记录当前学习率
                    writer.add_scalar("learning_rate", current_lr, total_steps) #[新增]将当前学习率写入 TensorBoard

                    if val_dataloader is not None:
                        print("Running validation set...")
                        model.eval()
                        with torch.no_grad():
                            val_losses = []
                            for (model_input, gt) in val_dataloader:
                                model_output = model(model_input)
                                val_loss = loss_fn(model_output, gt)
                                val_losses.append(val_loss)

                            writer.add_scalar("val_loss", np.mean(val_losses), total_steps)
                        model.train()

                total_steps += 1

            if scheduler is not None:
                scheduler.step()  #[新增]更新学习率

        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))


class LinearDecaySchedule():
    def __init__(self, start_val, final_val, num_steps):
        self.start_val = start_val
        self.final_val = final_val
        self.num_steps = num_steps

    def __call__(self, iter):
        return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)
