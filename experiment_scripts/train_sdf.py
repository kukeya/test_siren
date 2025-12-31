'''Reproduces Sec. 4.2 in main paper and Sec. 4 in Supplement.
'''

# Enable import from parent package
import sys
import os
import torch
import warnings

# 忽略 RTX 5090 兼容性警告
warnings.filterwarnings("ignore", message=".*NVIDIA GeForce RTX 5090.*")

sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, utils, training, modules
import loss_function.loss_functions as loss_functions

from torch.utils.data import DataLoader
import configargparse
import gpu_utils


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

p.add_argument('--hidden_features', type=int, default=256, help='Number of hidden features in the model')
p.add_argument('--num_hidden_layers', type=int, default=3, help='Number of hidden layers in the model')


opt = p.parse_args()


sdf_dataset = dataio.PointCloud(opt.point_cloud_path, on_surface_points=opt.batch_size)
dataloader = DataLoader(sdf_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)

# Define the model.
model = modules.SingleBVPNet(type=opt.model_type, in_features=3, 
                             hidden_features=opt.hidden_features,
                             num_hidden_layers=opt.num_hidden_layers)

# Load checkpoint if provided
if opt.checkpoint_path is not None:
    checkpoint = torch.load(opt.checkpoint_path, map_location=lambda storage, loc: storage.cuda())
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint) # 兼容只保存了 state_dict 的情况
    print(f"Loaded checkpoint from {opt.checkpoint_path}")

model.cuda()

# Define the loss 
from functools import partial
loss_fn = partial(loss_functions.sdf, 
                  sdf_weight=opt.sdf_weight, 
                  inter_weight=opt.inter_weight, 
                  normal_weight=opt.normal_weight, 
                  grad_weight=opt.grad_weight)
summary_fn = utils.write_sdf_summary

root_path = os.path.join(opt.logging_root, opt.experiment_name)

training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
               steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
               model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, double_precision=False,
               clip_grad=True)
