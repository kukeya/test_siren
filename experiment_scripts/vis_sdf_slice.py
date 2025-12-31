"""
SDF 截面可视化工具
生成指定平面上的 SDF 等值面图（等高线图）
"""

import sys
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import modules


def load_model(checkpoint_path: str, hidden_features: int = 256, num_hidden_layers: int = 3):
    """加载训练好的模型"""
    model = modules.SingleBVPNet(
        type='sine',
        in_features=3,
        hidden_features=hidden_features,
        num_hidden_layers=num_hidden_layers
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda())
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.cuda()
    model.eval()
    print(f"Loaded model from: {checkpoint_path}")
    return model


def generate_slice_grid(y_value: float, x_range: tuple, z_range: tuple, resolution: int):
    """
    在 y=y_value 平面上生成采样网格
    
    Args:
        y_value: 截面 y 坐标
        x_range: (x_min, x_max)
        z_range: (z_min, z_max)
        resolution: 网格分辨率
        
    Returns:
        coords: (N, 3) 坐标数组
        X, Z: 用于绘图的网格
    """
    x = np.linspace(x_range[0], x_range[1], resolution)
    z = np.linspace(z_range[0], z_range[1], resolution)
    X, Z = np.meshgrid(x, z)
    
    # 展平并添加 y 坐标
    x_flat = X.flatten()
    z_flat = Z.flatten()
    y_flat = np.full_like(x_flat, y_value)
    
    coords = np.stack([x_flat, y_flat, z_flat], axis=1)
    return coords, X, Z


def evaluate_sdf(model, coords: np.ndarray, batch_size: int = 100000):
    """
    批量评估 SDF 值
    
    Args:
        model: 训练好的模型
        coords: (N, 3) 坐标
        batch_size: 批大小
        
    Returns:
        sdf_values: (N,) SDF 值
    """
    sdf_values = []
    
    with torch.no_grad():
        for i in range(0, len(coords), batch_size):
            batch = coords[i:i+batch_size]
            batch_tensor = torch.from_numpy(batch).float().cuda()
            
            # 模型输入格式: {'coords': (B, N, 3)}
            model_input = {'coords': batch_tensor.unsqueeze(0)}
            output = model(model_input)
            
            # 输出格式: {'model_out': (B, N, 1)}
            sdf = output['model_out'].squeeze().cpu().numpy()
            sdf_values.append(sdf)
    
    return np.concatenate(sdf_values)


def plot_sdf_contour(X, Z, sdf_grid, output_path: str = None, 
                     num_levels: int = 30, y_value: float = 0.0,
                     show_colorbar: bool = True):
    """
    绘制 SDF 等高线图
    
    Args:
        X, Z: 网格坐标
        sdf_grid: SDF 值网格
        output_path: 保存路径
        num_levels: 等高线数量
        y_value: 截面 y 值（用于标题）
        show_colorbar: 是否显示颜色条
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 计算等高线级别
    vmin, vmax = sdf_grid.min(), sdf_grid.max()
    # 确保 0 在等高线中
    if vmin < 0 < vmax:
        # 创建对称的等高线级别
        abs_max = max(abs(vmin), abs(vmax))
        levels = np.linspace(-abs_max, abs_max, num_levels)
    else:
        levels = np.linspace(vmin, vmax, num_levels)
    
    # 填充等高线
    contourf = ax.contourf(X, Z, sdf_grid, levels=levels, cmap='RdYlBu_r', extend='both')
    
    # 等高线轮廓
    contour = ax.contour(X, Z, sdf_grid, levels=levels, colors='black', linewidths=0.3, alpha=0.5)
    
    # 零等值面（模型表面）高亮
    zero_contour = ax.contour(X, Z, sdf_grid, levels=[0], colors='black', linewidths=2)
    ax.clabel(zero_contour, inline=True, fontsize=10, fmt='0')
    
    if show_colorbar:
        cbar = plt.colorbar(contourf, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('SDF Value', fontsize=12)
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Z', fontsize=12)
    ax.set_title(f'SDF Cross-Section at Y = {y_value:.6f}', fontsize=14)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"Saved figure to: {output_path}")
    
    plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser(description='SDF Slice Visualization')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--y_value', type=float, default=0.05812123,
                        help='Y coordinate of the slice plane')
    parser.add_argument('--x_range', type=float, nargs=2, default=[-0.2, 0.2],
                        help='X range: min max')
    parser.add_argument('--z_range', type=float, nargs=2, default=[0.0, 0.4],
                        help='Z range: min max')
    parser.add_argument('--resolution', type=int, default=512,
                        help='Grid resolution')
    parser.add_argument('--num_levels', type=int, default=30,
                        help='Number of contour levels')
    parser.add_argument('--output', type=str, default=None,
                        help='Output image path')
    parser.add_argument('--hidden_features', type=int, default=256,
                        help='Model hidden features')
    parser.add_argument('--num_hidden_layers', type=int, default=3,
                        help='Model hidden layers')
    
    args = parser.parse_args()
    
    # 加载模型
    model = load_model(args.checkpoint, args.hidden_features, args.num_hidden_layers)
    
    # 生成网格
    print(f"Generating grid at Y = {args.y_value}...")
    print(f"X range: {args.x_range}, Z range: {args.z_range}")
    print(f"Resolution: {args.resolution} x {args.resolution}")
    
    coords, X, Z = generate_slice_grid(
        y_value=args.y_value,
        x_range=tuple(args.x_range),
        z_range=tuple(args.z_range),
        resolution=args.resolution
    )
    
    # 评估 SDF
    print("Evaluating SDF values...")
    sdf_values = evaluate_sdf(model, coords)
    sdf_grid = sdf_values.reshape(args.resolution, args.resolution)
    
    print(f"SDF range: [{sdf_grid.min():.4f}, {sdf_grid.max():.4f}]")
    
    # 绘图
    plot_sdf_contour(
        X, Z, sdf_grid,
        output_path=args.output,
        num_levels=args.num_levels,
        y_value=args.y_value
    )


# ==========================================
# 快速使用示例（直接运行时使用默认参数）
# ==========================================
def quick_visualize(checkpoint_path: str, y_value: float = 0.133856, output_path: str = None):
    """
    快速可视化函数，可在 Jupyter 或交互式环境中直接调用
    
    Example:
        from visualize_sdf_slice import quick_visualize
        quick_visualize('/path/to/checkpoint.pth', y_value=0.133856)
    """
    model = load_model(checkpoint_path)
    
    # 使用标准化模型的范围 [-0.5, 0.5]
    coords, X, Z = generate_slice_grid(
        y_value=y_value,
        x_range=(-1, 1),
        z_range=(-1, 1),
        resolution=512
    )
    
    sdf_values = evaluate_sdf(model, coords)
    sdf_grid = sdf_values.reshape(512, 512)
    
    fig = plot_sdf_contour(X, Z, sdf_grid, output_path=output_path, y_value=y_value)
    return fig, sdf_grid


if __name__ == '__main__':
    main()
