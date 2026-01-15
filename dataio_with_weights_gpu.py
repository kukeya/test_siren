import numpy as np
from torch.utils.data import Dataset
import torch
import os

class PointCloud(Dataset):
    def __init__(self, pointcloud_path, on_surface_points, keep_aspect_ratio=True, negative_sample_path=None, 
                 inner_ratio=0.15, 
                 device='cuda',
                 thin_plate_radius=0.03,
                 thin_plate_sigma=None):
        super().__init__()
        
        self.device = torch.device(device)
        print(f"Loading point cloud on device: {self.device}")
        point_cloud = np.genfromtxt(pointcloud_path)
        print("Finished loading point cloud")

        coords_np = point_cloud[:, :3].astype(np.float32)
        normals_np = point_cloud[:, 3:6].astype(np.float32)

        # 直接在 GPU 上创建张量
        self.coords = torch.from_numpy(coords_np).to(self.device)       # [N,3] float32
        self.normals = torch.from_numpy(normals_np).to(self.device)     # [N,3] float32

        if point_cloud.shape[1] > 6:
            is_def_np = point_cloud[:, 6:7].astype(np.int32)
            print(f"检测到变形标记列, {is_def_np.shape[0]} 个点")
        else:
            is_def_np = np.zeros((coords_np.shape[0], 1), dtype=np.int32)

        self.is_deform = torch.from_numpy(is_def_np).to(self.device)    # [N,1] int32

        # 预先分离索引（GPU 张量）
        self.inner_indices = torch.nonzero(self.is_deform.view(-1) == 1, as_tuple=False).view(-1)
        self.surface_indices = torch.nonzero(self.is_deform.view(-1) == 0, as_tuple=False).view(-1)
        print(f"采样策略初始化: 关键点 {self.inner_indices.numel()} 个, 普通点 {self.surface_indices.numel()} 个")

        self.inner_ratio = inner_ratio
        self.on_surface_points = int(on_surface_points)

        # 新增
        self.deform_coords = self.coords.index_select(0, self.inner_indices) if self.inner_indices.numel() > 0 else None
        self.thin_plate_radius = float(thin_plate_radius) if thin_plate_radius is not None else None
        if thin_plate_sigma is None and self.thin_plate_radius is not None:
            thin_plate_sigma = self.thin_plate_radius / 2.0
        self.thin_plate_sigma = float(thin_plate_sigma) if thin_plate_sigma is not None else None

    def __len__(self):
        return max(1, self.coords.shape[0] // self.on_surface_points)

    def __getitem__(self, idx):
        num_inner = int(self.on_surface_points * self.inner_ratio)
        num_surface = self.on_surface_points - num_inner

        # 1) 关键点采样（允许重复，GPU 采样）
        if self.inner_indices.numel() > 0 and num_inner > 0:
            inner_sel = self.inner_indices[torch.randint(0, self.inner_indices.numel(), (num_inner,), device=self.device)]
        else:
            inner_sel = torch.empty(0, dtype=torch.long, device=self.device)
            num_surface = self.on_surface_points

        # 2) 普通点采样（不重复，GPU 采样）
        if self.surface_indices.numel() >= num_surface:
            perm = torch.randperm(self.surface_indices.numel(), device=self.device)
            surface_sel = self.surface_indices[perm[:num_surface]]
        else:
            surface_sel = self.surface_indices[torch.randint(0, self.surface_indices.numel(), (num_surface,), device=self.device)]

        # 3) 合并索引并切片（全在 GPU 上）
        sel = torch.cat([inner_sel, surface_sel], dim=0)

        on_coords = self.coords.index_select(0, sel)
        on_normals = self.normals.index_select(0, sel)
        on_is_def = self.is_deform.index_select(0, sel).to(torch.int32)
        
        # Off-surface 采样（直接在 GPU 上生成）
        off_n = self.on_surface_points
        off_coords = torch.empty((off_n, 3), dtype=torch.float32, device=self.device).uniform_(-1.0, 1.0)
        off_normals = torch.full((off_n, 3), -1.0, dtype=torch.float32, device=self.device)
        off_is_def = torch.zeros((off_n, 1), dtype=torch.int32, device=self.device)
        # on_thin_plate_mask = on_is_def.float()
        # 新增
        # if self.deform_coords is not None and self.thin_plate_sigma is not None:
        #     dist_min = torch.cdist(on_coords, self.deform_coords).min(dim=1).values
        #     on_thin_plate_mask = torch.exp(-0.5 * (dist_min / self.thin_plate_sigma) ** 2).unsqueeze(-1)
        #     if self.thin_plate_radius is not None:
        #         on_thin_plate_mask = torch.where(
        #             dist_min.unsqueeze(-1) <= self.thin_plate_radius,
        #             on_thin_plate_mask,
        #             torch.zeros_like(on_thin_plate_mask),
        #         )
        # else:
        #     on_thin_plate_mask = on_is_def.float()
        
        # 全局的thin-plate mask
        on_thin_plate_mask = torch.ones((on_coords.shape[0], 1), dtype=torch.float32, device=self.device)
        global_thin_plate_mask = torch.zeros((off_n, 1), dtype=torch.float32, device=self.device)

        # 拼接（全在 GPU 上）
        coords = torch.cat([on_coords, off_coords], dim=0)              # [2M,3]
        normals = torch.cat([on_normals, off_normals], dim=0)           # [2M,3]
        is_deform = torch.cat([on_is_def, off_is_def], dim=0)           # [2M,1]
        thin_plate_mask = torch.cat([on_thin_plate_mask, global_thin_plate_mask], dim=0)

        sdf = torch.zeros((self.on_surface_points + off_n, 1), dtype=torch.float32, device=self.device)
        sdf[self.on_surface_points:, :] = -1.0

        return {'coords': coords}, {'sdf': sdf, 'normals': normals, 'is_deform': is_deform,
                                    'thin_plate_mask': thin_plate_mask}