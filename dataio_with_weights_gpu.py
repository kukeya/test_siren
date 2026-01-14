import numpy as np
from torch.utils.data import Dataset
import torch
import os
from scipy.spatial import cKDTree  # [新增] 用于计算局部密度

class PointCloud(Dataset):
    def __init__(self, pointcloud_path, on_surface_points, keep_aspect_ratio=True, negative_sample_path=None, inner_ratio=0.15):
        super().__init__()

        # [新增] 定义设备，优先使用 GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Data processing device: {self.device}")

        print("Loading point cloud")
        point_cloud = np.genfromtxt(pointcloud_path)
        print("Finished loading point cloud")

        coords_np = point_cloud[:, :3].astype(np.float32)
        normals_np = point_cloud[:, 3:6].astype(np.float32)

        # [修改] 转为 torch 张量并立即移动到 GPU
        self.coords = torch.from_numpy(coords_np).to(self.device)       # [N,3] float32 @ GPU
        self.normals = torch.from_numpy(normals_np).to(self.device)     # [N,3] float32 @ GPU

        if point_cloud.shape[1] > 6:
            is_def_np = point_cloud[:, 6:7].astype(np.int32)
            print(f"检测到变形标记列, {is_def_np.shape[0]} 个点")
        else:
            is_def_np = np.zeros((coords_np.shape[0], 1), dtype=np.int32)

        # [修改] 移动到 GPU
        self.is_deform = torch.from_numpy(is_def_np).to(self.device)    # [N,1] int32 @ GPU

        # 预先分离索引（torch 张量）
        # [修改] 操作都在 GPU 上进行
        self.inner_indices = torch.nonzero(self.is_deform.view(-1) == 1, as_tuple=False).view(-1)
        self.surface_indices = torch.nonzero(self.is_deform.view(-1) == 0, as_tuple=False).view(-1)
        print(f"采样策略初始化: 关键点 {self.inner_indices.numel()} 个, 普通点 {self.surface_indices.numel()} 个")

        self.inner_ratio = inner_ratio
        self.on_surface_points = int(on_surface_points)

        # ==========================================
        # [新增] 计算局部 Sigma (Local Density)
        # ==========================================
        print("Computing local sigmas via KDTree (k=50)...")
        # KDTree 依旧使用 numpy 在 CPU 上构建，这部分通常很快且只运行一次
        ptree = cKDTree(coords_np)
        sigma_set = []
        # 分块计算以防内存爆掉
        for p in np.array_split(coords_np, 100, axis=0):
            # 查询 51 个最近邻 (第一个是自己)，取第 51 个的距离
            d, _ = ptree.query(p, k= 5 + 1)
            sigma_set.append(d[:, -1])
        
        # [修改] 将结果 numpy 拼接后转 tensor 并移动到 GPU
        self.local_sigma = torch.from_numpy(np.concatenate(sigma_set)).float().to(self.device) # [N] @ GPU
        print("Finished computing local sigmas.")


    def __len__(self):
        # 一个 epoch 的 step 数由外部控制；这里给出保守值
        return max(1, self.coords.shape[0] // self.on_surface_points)

    def __getitem__(self, idx):
        # [删除] 不要每次都重置种子
        # gen = torch.Generator()
        # gen.manual_seed(torch.initial_seed())

        # ---------------------------------------------------
        # 1. 表面点采样 (On-Surface) - 保持原样
        # ---------------------------------------------------
        num_inner = int(self.on_surface_points * self.inner_ratio)
        num_surface = self.on_surface_points - num_inner

        # 1.1) 关键点采样（允许重复）
        if self.inner_indices.numel() > 0 and num_inner > 0:
            # [修改] randint 直接在 device 上运行，比 CPU 快
            rand_idx = torch.randint(0, self.inner_indices.numel(), (num_inner,), device=self.device)
            inner_sel = self.inner_indices[rand_idx] 
        else:
            inner_sel = torch.empty(0, dtype=torch.long, device=self.device)
            num_surface = self.on_surface_points # 补齐

        # 1.2) 普通点采样（不重复）
        if self.surface_indices.numel() >= num_surface:
            # [修改] randperm 在 device 上运行
            perm = torch.randperm(self.surface_indices.numel(), device=self.device)
            surface_sel = self.surface_indices[perm[:num_surface]]
        else:
            rand_idx = torch.randint(0, self.surface_indices.numel(), (num_surface,), device=self.device)
            surface_sel = self.surface_indices[rand_idx]

        # 1.3) 合并索引并提取数据
        sel = torch.cat([inner_sel, surface_sel], dim=0)

        # 索引选择操作完全在 GPU 上进行
        on_coords = self.coords.index_select(0, sel)
        on_normals = self.normals.index_select(0, sel)
        on_is_def = self.is_deform.index_select(0, sel).to(torch.int32)
        on_thin_plate_mask = on_is_def.float()
        
        # [新增] 获取对应点的 Sigma
        on_sigmas = self.local_sigma.index_select(0, sel).unsqueeze(-1) # [M, 1]

        # ---------------------------------------------------
        # 2. 离面点采样 (Off-Surface) - 修改逻辑
        # ---------------------------------------------------
        
        # A. 局部采样 (Local Perturbation)：在表面点附近加高斯噪声
        # 这些点用于学习 Manifold 附近的 Eikonal 约束
        # [加速] randn_like 会自动继承 on_coords 的设备(GPU)，完全并行计算
        # local_perturb_coords = on_coords + (torch.randn_like(on_coords) * on_sigmas)
        # 1. 取随机步长，正态分布
        step = torch.randn((on_coords.shape[0], 1), device=self.device) * on_sigmas
        # 2. 沿法线移动
        local_perturb_coords = on_coords + on_normals * step
        # 3. 再加一点点非常微小的各向同性噪声，防止过拟合到直线上
        local_perturb_coords += torch.randn_like(on_coords) * (on_sigmas * 0.1)

        
        # B. 全局采样 (Global Uniform)：减少采样数 (1/8)
        # 这些点用于探索整个空间
        num_global = self.on_surface_points // 8
        # [修改] 必须指定 device=self.device，否则默认在 CPU 创建然后复制，浪费时间
        global_coords = torch.empty((num_global, 3), dtype=torch.float32, device=self.device).uniform_(-1.0, 1.0)
        
        # ---------------------------------------------------
        # 3. 组装数据
        # ---------------------------------------------------
        
        # 为了保持维度一致，对于非表面点，法向和 SDF 设为无效值(-1)
        # Local 点
        local_normals = torch.full_like(local_perturb_coords, -1.0) # 继承 GPU
        local_is_def = torch.zeros((local_perturb_coords.shape[0], 1), dtype=torch.int32, device=self.device)
        local_thin_plate_mask = on_thin_plate_mask
        
        # Global 点
        global_normals = torch.full_like(global_coords, -1.0) # 继承 GPU
        global_is_def = torch.zeros((num_global, 1), dtype=torch.int32, device=self.device)
        global_thin_plate_mask = torch.zeros((num_global, 1), dtype=torch.float32, device=self.device)

        # 拼接所有坐标: [On-Surface, Local-Perturbed, Global-Uniform]
        # 所有张量都在 GPU 上，cat 操作非常快
        coords = torch.cat([on_coords, local_perturb_coords, global_coords], dim=0)
        normals = torch.cat([on_normals, local_normals, global_normals], dim=0)
        is_deform = torch.cat([on_is_def, local_is_def, global_is_def], dim=0)
        thin_plate_mask = torch.cat([on_thin_plate_mask, local_thin_plate_mask, global_thin_plate_mask], dim=0)

        # 构造 SDF 标签
        # On-Surface = 0
        # Off-Surface (Local + Global) = -1 (Dummy value)
        
        # [修改] 直接在 GPU 上创建 zeros
        sdf = torch.zeros((coords.shape[0], 1), dtype=torch.float32, device=self.device)
        # 将非表面点的 SDF 设为 -1 (假设 Loss 函数会根据 sdf==0 区分表面点，或忽略 -1)
        sdf[self.on_surface_points:, :] = -1.0

        return {'coords': coords}, {'sdf': sdf, 'normals': normals, 'is_deform': is_deform,
                                    'thin_plate_mask': thin_plate_mask}
