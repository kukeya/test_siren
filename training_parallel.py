'''Implements a generic training loop.
'''

import torch
import torch.distributed as dist  # 添加 distributed
import utils
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import shutil


def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
          summary_fn, val_dataloader=None, double_precision=False, clip_grad=False, use_lbfgs=False, loss_schedules=None,
          train_sampler=None): # 1. 新增 train_sampler 参数

    optim = torch.optim.Adam(lr=lr, params=model.parameters())

    # 2. 判断当前进程是否为主进程 (Rank 0)
    is_ddp = dist.is_initialized()
    rank = dist.get_rank() if is_ddp else 0
    is_master = (rank == 0)

    # 3. 只有主进程负责创建目录和清理
    if is_master:
        if os.path.exists(model_dir):
            # DDP 模式下最好不要交互式输入，直接覆盖或在外部处理
            if not is_ddp: 
                val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
                if val == 'y':
                    shutil.rmtree(model_dir)
            else:
                print(f"DDP mode: Overwriting {model_dir}")
                # shutil.rmtree(model_dir) # 小心使用，或者手动清理

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        summaries_dir = os.path.join(model_dir, 'summaries')
        utils.cond_mkdir(summaries_dir)

        checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        utils.cond_mkdir(checkpoints_dir)

        writer = SummaryWriter(summaries_dir)
    else:
        writer = None # 非主进程没有 writer

    # 4. 同步：等待主进程创建好目录
    if is_ddp:
        dist.barrier()

    total_steps = 0
    
    # 5. 只有主进程显示进度条
    with tqdm(total=len(train_dataloader) * epochs, disable=not is_master) as pbar:
        train_losses = []
        for epoch in range(epochs):
            
            # 6. DDP 必须设置 epoch 以保证 shuffle
            if is_ddp and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            # 7. 保存模型：只在主进程，且需要解包 DDP 模型
            if not epoch % epochs_til_checkpoint and epoch and is_master:
                # 获取原始模型（去掉 DDP 的 wrapper）
                model_to_save = model.module if is_ddp else model
                torch.save(model_to_save.state_dict(),
                           os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
                           np.array(train_losses))

            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()
            
                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}

                if double_precision:
                    model_input = {key: value.double() for key, value in model_input.items()}
                    gt = {key: value.double() for key, value in gt.items()}

                
                model_output = model(model_input)
                losses = loss_fn(model_output, gt)

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if loss_schedules is not None and loss_name in loss_schedules:
                        if is_master: # 只在主进程记录
                            writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                        single_loss *= loss_schedules[loss_name](total_steps)

                    if is_master: # 只在主进程记录
                        writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss

                if is_master:
                    if isinstance(train_loss, torch.Tensor):
                        train_losses.append(train_loss.item())
                    else:
                        train_losses.append(train_loss)
                    writer.add_scalar("total_train_loss", train_loss, total_steps)

                # 8. 保存中间结果：只在主进程
                if not total_steps % steps_til_summary and is_master:
                    model_to_save = model.module if is_ddp else model
                    torch.save(model_to_save.state_dict(),
                               os.path.join(checkpoints_dir, 'model_current.pth'))
                    summary_fn(model, model_input, gt, model_output, writer, total_steps)

                if not use_lbfgs:
                    optim.zero_grad()
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

                if not total_steps % steps_til_summary and is_master:
                    tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))

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

        # 9. 最终保存：只在主进程
        if is_master:
            model_to_save = model.module if is_ddp else model
            torch.save(model_to_save.state_dict(),
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
