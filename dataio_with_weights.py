import numpy as np
from torch.utils.data import Dataset
import torch
import os

class PointCloud(Dataset):
    def __init__(self, pointcloud_path, on_surface_points, keep_aspect_ratio=True, negative_sample_path=None, inner_ratio=0.15):
        super().__init__()

        print("Loading point cloud")
        point_cloud = np.genfromtxt(pointcloud_path)
        print("Finished loading point cloud")

        coords = point_cloud[:, :3]
        self.normals = point_cloud[:, 3:6]
        self.coords = coords

        if point_cloud.shape[1] > 6:
            # self.sdf_gt = point_cloud[:, 6:7]
            self.is_deform = point_cloud[:, 6:7]
            print(f"检测到变形标记列, {len(self.is_deform)} 个点")
        else:
            self.is_deform = np.zeros((coords.shape[0], 1))
            
            # === 动态生成权重逻辑 ===
            # 初始化权重为 1
            # self.weights = np.ones_like(self.sdf_gt)
            
            # 找到所有负值点 (膨胀/内部点)
            # 注意：这里假设你的膨胀点 SDF 是负数 (如 -0.0024)
            # is_inner_point = self.sdf_gt < -1e-6
            # print(f"检测到 {np.sum(is_inner_point)} 个内部点 (SDF < -1e-6)")
            # 给这些点赋予高权重 (例如 1000 倍)
            # 你可以根据效果调整这个数值，数值越小越难学，权重就要越大
            # self.weights[is_inner_point] = 200.0
            # self.weights[self.is_deform == 1] = 200.0
            
            # print(f"已为 {np.sum(is_inner_point)} 个内部点赋予高权重 (100.0)")
        # else:
            # self.sdf_gt = np.zeros((coords.shape[0], 1))
            # self.weights = np.ones((coords.shape[0], 1))
            # print("未检测到SDF列，默认为0，权重为1")

        # [新增] 预先分离索引
        # 找出关键点（膨胀点/负值点）的索引
        # self.inner_indices = np.where(self.sdf_gt < -1e-6)[0]
        # 找出普通表面点的索引
        # self.surface_indices = np.where(np.abs(self.sdf_gt) < 1e-6)[0]
        
        # print(f"采样策略初始化: 关键点 {len(self.inner_indices)} 个, 普通点 {len(self.surface_indices)} 个")

        self.inner_indices = np.where(self.is_deform == 1)[0]
        self.surface_indices = np.where(self.is_deform == 0)[0]
        print(f"采样策略初始化: 关键点 {len(self.inner_indices)} 个, 普通点 {len(self.surface_indices)} 个")
        
        # 设定关键点在每个 Batch 中的占比 (例如 20% ~ 50%)
        # 即使它们只占总数的 1%，我们也强制让它们占 Batch 的 20%
        # v4 增加采样频次，因为点少了
        self.inner_ratio = inner_ratio

        self.on_surface_points = on_surface_points

    def __len__(self):
        return self.coords.shape[0] // self.on_surface_points

    def __getitem__(self, idx):
        # 计算每个 Batch 需要多少个关键点
        num_inner = int(self.on_surface_points * self.inner_ratio)
        num_surface = self.on_surface_points - num_inner
        
        # 1. 采样关键点 (如果关键点太少，允许重复采样 replace=True)
        if len(self.inner_indices) > 0:
            rand_inner = np.random.choice(self.inner_indices, size=num_inner, replace=True)
        else:
            # 如果没有关键点，全采普通点
            rand_inner = np.array([], dtype=int)
            num_surface = self.on_surface_points

        # 2. 采样普通表面点
        rand_surface = np.random.choice(self.surface_indices, size=num_surface, replace=False)
        
        # 3. 合并索引
        rand_idcs = np.concatenate((rand_inner, rand_surface))
        
        # --- 获取 On-surface 数据 ---
        on_surface_coords = self.coords[rand_idcs, :]
        on_surface_normals = self.normals[rand_idcs, :]
        # on_surface_sdf = self.sdf_gt[rand_idcs, :]
        # on_surface_weights = self.weights[rand_idcs, :]
        on_surface_is_deform = self.is_deform[rand_idcs, :]  # 新增


        # --- 准备 Off-surface 数据 ---
        off_surface_samples = self.on_surface_points
        total_samples = self.on_surface_points + off_surface_samples

        off_surface_coords = np.random.uniform(-1, 1, size=(off_surface_samples, 3))
        off_surface_normals = np.ones((off_surface_samples, 3)) * -1
        # off_surface_sdf = np.ones((off_surface_samples, 1)) * -1
        # off_surface_weights = np.ones((off_surface_samples, 1)) # 空间点的权重设为 1
        off_surface_is_deform = np.zeros((off_surface_samples, 1))  # 新增：空间点不是膨胀点


        # --- 拼接所有数据 ---
        sdf = np.zeros((total_samples, 1))  # on-surface = 0
        sdf[self.on_surface_points:, :] = -1  # off-surface = -1

        coords = np.concatenate((on_surface_coords, off_surface_coords), axis=0)
        normals = np.concatenate((on_surface_normals, off_surface_normals), axis=0)
        
        # weights = np.concatenate((on_surface_weights, off_surface_weights), axis=0)
        is_deform = np.concatenate((on_surface_is_deform, off_surface_is_deform), axis=0)  # 新增


        return {'coords': torch.from_numpy(coords).float()}, {'sdf': torch.from_numpy(sdf).float(),
                                                              'normals': torch.from_numpy(normals).float(), 
                                                              'is_deform': torch.from_numpy(is_deform).bool()}


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        meshgrid = np.mgrid[:sidelen[0], :sidelen[1]]
        pixel_coords = np.stack([meshgrid[0], meshgrid[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
    elif dim == 3:
        meshgrid = np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]]
        pixel_coords = np.stack([meshgrid[0], meshgrid[1], meshgrid[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords

def lin2img(tensor, image_resolution=None):
    batch_size, num_samples, channels = tensor.shape
    if image_resolution is None:
        width = np.sqrt(num_samples).astype(int)
        height = width
    else:
        height = image_resolution[0]
        width = image_resolution[1]

    return tensor.permute(0, 2, 1).view(batch_size, channels, height, width)
