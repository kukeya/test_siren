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

        coords_np = point_cloud[:, :3].astype(np.float32)
        normals_np = point_cloud[:, 3:6].astype(np.float32)

        # 转为 torch 张量（CPU）
        self.coords = torch.from_numpy(coords_np)       # [N,3] float32
        self.normals = torch.from_numpy(normals_np)     # [N,3] float32

        if point_cloud.shape[1] > 6:
            is_def_np = point_cloud[:, 6:7].astype(np.int32)
            print(f"检测到变形标记列, {is_def_np.shape[0]} 个点")
        else:
            is_def_np = np.zeros((coords_np.shape[0], 1), dtype=np.int32)

        self.is_deform = torch.from_numpy(is_def_np)    # [N,1] int32

        # 预先分离索引（torch 张量）
        self.inner_indices = torch.nonzero(self.is_deform.view(-1) == 1, as_tuple=False).view(-1)
        self.surface_indices = torch.nonzero(self.is_deform.view(-1) == 0, as_tuple=False).view(-1)
        print(f"采样策略初始化: 关键点 {self.inner_indices.numel()} 个, 普通点 {self.surface_indices.numel()} 个")

        self.inner_ratio = inner_ratio
        self.on_surface_points = int(on_surface_points)

    def __len__(self):
        # 一个 epoch 的 step 数由外部控制；这里给出保守值
        return max(1, self.coords.shape[0] // self.on_surface_points)

    def __getitem__(self, idx):
        # [删除] 不要每次都重置种子
        # gen = torch.Generator()
        # gen.manual_seed(torch.initial_seed())

        num_inner = int(self.on_surface_points * self.inner_ratio)
        num_surface = self.on_surface_points - num_inner

        # 1) 关键点采样（允许重复）
        if self.inner_indices.numel() > 0 and num_inner > 0:
            # [修改] 移除 generator=gen
            inner_sel = self.inner_indices[torch.randint(0, self.inner_indices.numel(), (num_inner,))] 
        else:
            inner_sel = torch.empty(0, dtype=torch.long)
            num_surface = self.on_surface_points

        # 2) 普通点采样（不重复）
        if self.surface_indices.numel() >= num_surface:
            # [修改] 移除 generator=gen
            perm = torch.randperm(self.surface_indices.numel())
            surface_sel = self.surface_indices[perm[:num_surface]]
        else:
            # [修改] 移除 generator=gen
            surface_sel = self.surface_indices[torch.randint(0, self.surface_indices.numel(), (num_surface,))]

        # 3) 合并索引并切片
        sel = torch.cat([inner_sel, surface_sel], dim=0)

        on_coords = self.coords.index_select(0, sel)
        on_normals = self.normals.index_select(0, sel)
        on_is_def = self.is_deform.index_select(0, sel).to(torch.int32)

        # Off-surface 采样
        off_n = self.on_surface_points
        off_n = int(off_n)
        # [修改] 移除 generator=gen
        off_coords = torch.empty((off_n, 3), dtype=torch.float32).uniform_(-1.0, 1.0)
        off_normals = torch.full((off_n, 3), -1.0, dtype=torch.float32)
        off_is_def = torch.zeros((off_n, 1), dtype=torch.int32)

        # 拼接
        coords = torch.cat([on_coords, off_coords], dim=0)              # [2M,3]
        normals = torch.cat([on_normals, off_normals], dim=0)           # [2M,3]
        is_deform = torch.cat([on_is_def, off_is_def], dim=0)           # [2M,1]

        sdf = torch.zeros((self.on_surface_points + off_n, 1), dtype=torch.float32)
        sdf[self.on_surface_points:, :] = -1.0

        return {'coords': coords}, {'sdf': sdf, 'normals': normals, 'is_deform': is_deform}