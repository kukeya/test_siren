'''Reproduces Sec. 4.2 in main paper and Sec. 4 in Supplement.
'''

# Enable import from parent package
import sys
import os
import torch
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import dataio, utils, modules, training_parallel, loss_functions_adpt_weight
import configargparse
# import gpu_utils

local_rank = int(os.environ.get("LOCAL_RANK", -1))
is_ddp = local_rank != -1
if is_ddp:
    # 初始化进程组
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
else:
    # 单卡回退逻辑
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# gpu_utils.auto_select_gpu()

p = configargparse.ArgumentParser()
p.add_argument('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, default='exp2',
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=15000)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=5e-5')
p.add_argument('--num_epochs', type=int, default=3000,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=3000,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=3000,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--model_type', type=str, default='sine',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')
p.add_argument('--point_cloud_path', type=str, default='ruyi14w_n_deformed.xyz',
               help='Path to the point cloud file.')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')

# Loss weights
p.add_argument('--sdf_weight', type=float, default=3e3, help='Weight for SDF loss')
p.add_argument('--inter_weight', type=float, default=1e2, help='Weight for inter loss')
p.add_argument('--normal_weight', type=float, default=1e2, help='Weight for normal loss')
p.add_argument('--grad_weight', type=float, default=5e1, help='Weight for eikonal gradient loss')
p.add_argument('--thin_plate_weight', type=float, default=1.0, help='Weight for thin-plate bending loss')
p.add_argument('--thin_plate_epochs', type=int, default=100,
               help='Extra epochs at the end to ramp up thin-plate loss')

p.add_argument('--hidden_features', type=int, default=512, help='Number of hidden features in the model')
p.add_argument('--num_hidden_layers', type=int, default=5, help='Number of hidden layers in the model')


opt = p.parse_args()


sdf_dataset = dataio.PointCloud(opt.point_cloud_path, on_surface_points=opt.batch_size)

if is_ddp:
    # 使用 DistributedSampler
    sampler = DistributedSampler(sdf_dataset, shuffle=True)
    # 注意: shuffle 必须为 False，因为 sampler 会处理 shuffle
    dataloader = DataLoader(sdf_dataset, shuffle=False, batch_size=1, pin_memory=True, num_workers=4, sampler=sampler)
else:
    dataloader = DataLoader(sdf_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=4)

# dataloader = DataLoader(sdf_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)

# Define the model.
# if opt.model_type == 'nerf':
#     model = modules.SingleBVPNet(type='relu', mode='nerf', in_features=3)
# else:
#     model = modules.SingleBVPNet(type=opt.model_type, in_features=3)
    
# model = modules.SingleBVPNet(type=opt.model_type, in_features=3)

model = modules.SingleBVPNet(type=opt.model_type, in_features=3, 
                             hidden_features=opt.hidden_features,
                             num_hidden_layers=opt.num_hidden_layers)

# Load checkpoint if provided
if opt.checkpoint_path is not None:
    checkpoint = torch.load(opt.checkpoint_path)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint) # 兼容只保存了 state_dict 的情况
    print(f"Loaded checkpoint from {opt.checkpoint_path}")

# model.cuda()
model.to(device)
if is_ddp:
    # 包装模型
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# Define the loss 
from functools import partial
loss_fn = partial(loss_functions_adpt_weight.sdf, 
                  sdf_weight=opt.sdf_weight, 
                  inter_weight=opt.inter_weight, 
                  normal_weight=opt.normal_weight,
                  grad_weight=opt.grad_weight,
                  thin_plate_weight=opt.thin_plate_weight)
summary_fn = utils.write_sdf_summary

root_path = os.path.join(opt.logging_root, opt.experiment_name)

steps_per_epoch = len(dataloader)
thin_plate_steps = max(opt.thin_plate_epochs * steps_per_epoch, 1)
thin_plate_start = opt.num_epochs * steps_per_epoch

def thin_plate_schedule(step):
    if step < thin_plate_start:
        return 0.0
    return min((step - thin_plate_start) / thin_plate_steps, 1.0)

loss_schedules = None
total_epochs = opt.num_epochs
if opt.thin_plate_weight > 0.0 and opt.thin_plate_epochs > 0:
    loss_schedules = {'thin_plate': thin_plate_schedule}
    total_epochs += opt.thin_plate_epochs

training_parallel.train(model=model, train_dataloader=dataloader, epochs=total_epochs, lr=opt.lr,
               steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
               model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, double_precision=False,
               clip_grad=True,
               train_sampler=sampler if is_ddp else None,
               loss_schedules=loss_schedules)
