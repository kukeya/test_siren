
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial import cKDTree
import os

# ============================
# 1. 辅助函数：导出 PLY 文件
# ============================
def save_colored_ply(coords, colors, filename):
    """
    coords: [N, 3] numpy array
    colors: [N, 3] numpy array (0-255)
    """
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(coords)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for i in range(len(coords)):
            p = coords[i]
            c = colors[i]
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")
    
    print(f"Saved visualization to: {filename}")

# ============================
# 2. 修改后的 Dataset 类
# ============================
class PointCloudDebug(Dataset):
    def __init__(self, pointcloud_path, on_surface_points, k_neighbors=20):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.on_surface_points = on_surface_points
        
        print(f"Loading: {pointcloud_path}")
        point_cloud = np.genfromtxt(pointcloud_path)
        coords_np = point_cloud[:, :3].astype(np.float32)
        normals_np = point_cloud[:, 3:6].astype(np.float32)

        self.coords = torch.from_numpy(coords_np).to(self.device)
        self.normals = torch.from_numpy(normals_np).to(self.device)

        # 简单的 KDTree 计算 Sigma
        # 【注意】这里为了测试，将 k 设为了传入参数，建议先用 20 试试
        print(f"Computing local sigmas (k={k_neighbors})...")
        ptree = cKDTree(coords_np)
        sigma_set = []
        for p in np.array_split(coords_np, 100, axis=0):
            d, _ = ptree.query(p, k=k_neighbors + 1)
            sigma_set.append(d[:, -1])
        
        # 限制 sigma 最大值，防止飞出天际 (假设模型在 unit sphere 内，最大给 0.05)
        sigmas_np = np.concatenate(sigma_set)
        sigmas_np = np.clip(sigmas_np, a_min=1e-6, a_max=0.05) 
        
        self.local_sigma = torch.from_numpy(sigmas_np).float().to(self.device)
        print("Init done.")

    def __getitem__(self, idx):
        # 1. 简单随机采样表面点
        rand_idx = torch.randint(0, self.coords.shape[0], (self.on_surface_points,), device=self.device)
        
        on_coords = self.coords[rand_idx]
        on_normals = self.normals[rand_idx]
        on_sigmas = self.local_sigma[rand_idx].unsqueeze(-1)

        # ==========================================
        # 2. 你的新代码：沿着法线方向扰动
        # ==========================================
        # (1) 取随机步长，正态分布
        step = torch.randn((on_coords.shape[0], 1), device=self.device) * on_sigmas
        
        # (2) 沿法线移动
        # 注意：这里假设 normals 已经是单位向量
        local_perturb_coords = on_coords + on_normals * step
        
        # (3) (可选) 再加一点点非常微小的各向同性噪声
        local_perturb_coords += torch.randn_like(on_coords) * (on_sigmas * 0.1)
        # ==========================================

        # 3. 全局采样 (为了显示方便，这里数量设为和表面点一样)
        global_coords = torch.empty((self.on_surface_points, 3), device=self.device).uniform_(-1.0, 1.0)

        # 返回分开的数据以便可视化
        return {
            'on_surface': on_coords.cpu().numpy(),
            'perturbed': local_perturb_coords.cpu().numpy(),
            'global': global_coords.cpu().numpy()
        }

# ============================
# 3. 主执行逻辑
# ============================
if __name__ == "__main__":
    # --- 配置 ---
    FILE_PATH = "/home/jym/Repos/Experiments/IGR_sample/Opener_IGRsample2k=10 _single_recur/02_iterations/iter_16/pcd_deformed_w.xyz"  # <--- 请修改这里
    N_POINTS = 15000                    # 采样点数
    K_NEIGHBORS = 10                    # KDTree 的 K 值，建议不要太大

    if not os.path.exists(FILE_PATH):
        print(f"Error: 文件 {FILE_PATH} 不存在，请生成或指定正确的 .xyz 文件")
    else:
        # 初始化数据集
        ds = PointCloudDebug(FILE_PATH, on_surface_points=N_POINTS, k_neighbors=K_NEIGHBORS)

        # 运行一次 __getitem__
        data = ds[0]

        # 准备颜色
        # 1. 表面点 = 绿色
        c_surface = np.tile(np.array([0, 255, 0]), (N_POINTS, 1))
        # 2. 扰动点 = 红色 (这就是我们要检查的！)
        c_perturb = np.tile(np.array([255, 0, 0]), (N_POINTS, 1))
        # 3. 全局点 = 蓝色 (半透明视觉效果不佳，所以用蓝色淡化)
        c_global = np.tile(np.array([0, 0, 255]), (N_POINTS, 1))

        # 合并所有点
        all_coords = np.vstack([data['on_surface'], data['perturbed'], data['global']])
        all_colors = np.vstack([c_surface, c_perturb, c_global])

        # 保存
        out_name = "/home/jym/Repos/Experiments/IGR_sample/Opener_IGRsample2k=10 _single_recur/02_iterations/iter_16/pcd_deformed_w_xyz_debug.ply"  # <--- 请修改这里
        save_colored_ply(all_coords, all_colors, out_name)

        print("\n------------------------------------------------")
        print(f"完成！请用 MeshLab 或 CloudCompare 打开: {out_name}")
        print("观察重点：")
        print("1. 红色点(Perturbed)是否紧贴着绿色点(Surface)分布？")
        print("2. 在深洞区域，红色点是否还会堵死洞口？")
        print("   (理论上现在红色点应该只会在法线方向前后移动，从而保留洞的空腔)")
        print("------------------------------------------------")