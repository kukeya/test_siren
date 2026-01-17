'''Reproduces Sec. 4.2 in main paper and Sec. 4 in Supplement.
'''

# Enable import from parent package
import sys
import os
import torch
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio_with_weights_gpu as dataio_with_weights, utils, training, modules
import configargparse
import gpu_utils
import importlib  # [新增]
from torch.utils.data import DataLoader


gpu_utils.auto_select_gpu()

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

# [新增] 负样本路径参数
p.add_argument('--negative_path', type=str, default=None, help='Path to the negative sample point cloud (old surface).')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')

# Loss weights
p.add_argument('--sdf_weight', type=float, default=3e3, help='Weight for SDF loss')
p.add_argument('--inter_weight', type=float, default=1e2, help='Weight for inter loss')
p.add_argument('--normal_weight', type=float, default=1e2, help='Weight for normal loss')
p.add_argument('--grad_weight', type=float, default=5e1, help='Weight for eikonal gradient loss')
p.add_argument('--thin_plate_weight', type=float, default=1.0, help='Weight for thin-plate bending loss')
p.add_argument('--thin_plate_epochs', type=int, default=100,
               help='Extra epochs at the end to ramp up thin-plate loss')
p.add_argument('--thin_plate_radius', type=float, default=0.05,
               help='Radius for thin-plate mask falloff (<=0 for auto)')
p.add_argument('--thin_plate_sigma', type=float, default=None,
               help='Sigma for thin-plate Gaussian falloff (default: radius/2)')
p.add_argument('--thin_plate_radius_factor', type=float, default=3.0,
               help='Auto radius factor times mean nearest deform distance')
p.add_argument('--thin_plate_radius_samples', type=int, default=1024,
               help='Max deform samples for auto radius estimation')
p.add_argument('--smooth_normal_weight', type=float, default=0.0,
               help='Weight for deform-only normal smoothness loss')
p.add_argument('--smooth_normal_k', type=int, default=16,
               help='kNN size for normal smoothness')
p.add_argument('--smooth_normal_radius', type=float, default=None,
               help='Optional radius cutoff for normal smoothness (same units as coords)')
p.add_argument('--smooth_normal_use_projection', action='store_true',
               help='Project editable points to sdf=0 before smoothing')
p.add_argument('--smooth_normal_feat_sigma', type=float, default=None,
               help='Feature protection sigma based on GT normal variation')
p.add_argument('--smooth_normal_max_points', type=int, default=2048,
               help='Max editable points used for smoothness per batch')
p.add_argument('--smooth_normal_ramp_epochs', type=int, default=0,
               help='Ramp epochs for smoothness weight after base epochs')
p.add_argument('--smooth_position_weight', type=float, default=0.0,
               help='Weight for deform-only position smoothness loss')
p.add_argument('--smooth_position_k', type=int, default=16,
               help='kNN size for position smoothness')
p.add_argument('--smooth_position_radius', type=float, default=None,
               help='Optional radius cutoff for position smoothness (same units as coords)')
p.add_argument('--smooth_position_use_projection', action='store_true',
               help='Project editable points to sdf=0 before position smoothing')
p.add_argument('--smooth_position_max_points', type=int, default=2048,
               help='Max editable points used for position smoothing per batch')
p.add_argument('--smooth_position_ramp_epochs', type=int, default=0,
               help='Ramp epochs for position smoothing weight after base epochs')

p.add_argument('--hidden_features', type=int, default=256, help='Number of hidden features in the model')
p.add_argument('--num_hidden_layers', type=int, default=3, help='Number of hidden layers in the model')
p.add_argument('--log_dir', type=str, default='', help='Direct path to save logs and checkpoints')

# [新增] 选择损失版本：默认 L1，可显式 --L2
loss_group = p.add_mutually_exclusive_group()
loss_group.add_argument('--L1', action='store_true', help='Use loss_function.loss_functions (L1)')
loss_group.add_argument('--L2', action='store_true', help='Use loss_function.loss_functionsL2 (L2)')


opt = p.parse_args()
# --- 修改路径逻辑 ---
if opt.log_dir:
    # 如果指定了 log_dir，直接使用它，不再在下面创建子文件夹
    root_path = opt.log_dir
    os.makedirs(root_path, exist_ok=True)
else:
    # 保持原有逻辑 (兼容以前的实验)
    root_path = os.path.join(opt.logging_root, opt.experiment_name)

# Detect device and ensure CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type != 'cuda':
    raise RuntimeError('需要可用的 CUDA 设备来训练权重模型。')

# [修改] 传入 negative_sample_path
sdf_dataset = dataio_with_weights.PointCloud(
    opt.point_cloud_path,
    on_surface_points=opt.batch_size,
    negative_sample_path=opt.negative_path,
    inner_ratio=0.15,
    thin_plate_radius=opt.thin_plate_radius,
    thin_plate_sigma=opt.thin_plate_sigma,
    thin_plate_radius_factor=opt.thin_plate_radius_factor,
    thin_plate_radius_samples=opt.thin_plate_radius_samples,
)
# dataloader = DataLoader(
#     sdf_dataset,
#     shuffle=True,
#     batch_size=1,
#     pin_memory=True,           # 允许非阻塞搬运
#     num_workers=8,             # 根据CPU核心调整
#     persistent_workers=True,
#     prefetch_factor=4
# )
dataloader = DataLoader(sdf_dataset, shuffle=True, batch_size=1, pin_memory=False, num_workers=0)


# Define the model.
model = modules.SingleBVPNet(type=opt.model_type, in_features=3, 
                             hidden_features=opt.hidden_features,
                             num_hidden_layers=opt.num_hidden_layers)

# Load checkpoint if provided
if opt.checkpoint_path is not None:
    checkpoint = torch.load(opt.checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)  # 兼容只保存了 state_dict 的情况
    print(f"Loaded checkpoint from {opt.checkpoint_path}")

model.cuda()

# [替换原固定导入方式]
# from functools import partial
# import loss_function.loss_functions as loss_functions  <-- 删除固定导入

# [新增] 按参数动态导入模块并构建 loss_fn
loss_module_name = 'loss_function.loss_functions_L2' if opt.L2 else 'loss_function.loss_functions'
loss_module = importlib.import_module(loss_module_name)
from functools import partial
loss_fn = partial(
    loss_module.sdf,
    sdf_weight=opt.sdf_weight,
    inter_weight=opt.inter_weight,
    normal_weight=opt.normal_weight,
    grad_weight=opt.grad_weight,
    thin_plate_weight=opt.thin_plate_weight,
    smooth_normal_weight=opt.smooth_normal_weight,
    smooth_normal_k=opt.smooth_normal_k,
    smooth_normal_radius=opt.smooth_normal_radius,
    smooth_normal_use_projection=opt.smooth_normal_use_projection,
    smooth_normal_feat_sigma=opt.smooth_normal_feat_sigma,
    smooth_normal_max_points=opt.smooth_normal_max_points,
    smooth_position_weight=opt.smooth_position_weight,
    smooth_position_k=opt.smooth_position_k,
    smooth_position_radius=opt.smooth_position_radius,
    smooth_position_use_projection=opt.smooth_position_use_projection,
    smooth_position_max_points=opt.smooth_position_max_points,
)
summary_fn = utils.write_sdf_summary

print(f"--------------------------------------------------")
print(f"[Train] Model Output Dir: {root_path}")
print(f"[Train] Log Dir arg: {opt.log_dir}")
print(f"[Train] Using loss module: {loss_module_name}")  # [新增]
print(f"--------------------------------------------------")


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
if opt.smooth_normal_weight > 0.0 and opt.smooth_normal_ramp_epochs > 0:
    smooth_steps = max(opt.smooth_normal_ramp_epochs * steps_per_epoch, 1)
    smooth_start = opt.num_epochs * steps_per_epoch

    def smooth_normal_schedule(step):
        if step < smooth_start:
            return 0.0
        return min((step - smooth_start) / smooth_steps, 1.0)

    loss_schedules = loss_schedules or {}
    loss_schedules['smooth_normal'] = smooth_normal_schedule
    total_epochs += opt.smooth_normal_ramp_epochs
if opt.smooth_position_weight > 0.0 and opt.smooth_position_ramp_epochs > 0:
    smooth_pos_steps = max(opt.smooth_position_ramp_epochs * steps_per_epoch, 1)
    smooth_pos_start = opt.num_epochs * steps_per_epoch

    def smooth_position_schedule(step):
        if step < smooth_pos_start:
            return 0.0
        return min((step - smooth_pos_start) / smooth_pos_steps, 1.0)

    loss_schedules = loss_schedules or {}
    loss_schedules['smooth_position'] = smooth_position_schedule
    total_epochs += opt.smooth_position_ramp_epochs

training.train(model=model, train_dataloader=dataloader, epochs=total_epochs, lr=opt.lr,
               steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
               model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, double_precision=False,
               clip_grad=True, use_lr_decay=opt.use_lr_decay, loss_schedules=loss_schedules)
