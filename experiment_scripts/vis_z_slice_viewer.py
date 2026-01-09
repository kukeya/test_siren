"""
SDF Z-Axis Tomography Viewer (层析成像查看器)
针对单个模型，生成 Z 轴方向上的连续 XY 切片
"""

import sys
import os
import argparse
import numpy as np
import torch
from pathlib import Path
import base64
from io import BytesIO

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import modules

# 设置 matplotlib 后端
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_model(checkpoint_path: str, hidden_features: int = 256, num_hidden_layers: int = 3):
    """加载训练好的模型"""
    model = modules.SingleBVPNet(
        type='sine',
        in_features=3,
        hidden_features=hidden_features,
        num_hidden_layers=num_hidden_layers
    )
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda())
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.cuda()
    model.eval()
    return model

def generate_xy_grid(z_value: float, x_range: tuple, y_range: tuple, resolution: int):
    """在 z=z_value 平面上生成 XY 采样网格"""
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = np.full_like(x_flat, z_value)
    
    # 构造坐标 [x, y, z]
    coords = np.stack([x_flat, y_flat, z_flat], axis=1)
    return coords, X, Y

def evaluate_sdf(model, coords: np.ndarray, batch_size: int = 100000):
    """批量评估 SDF 值"""
    sdf_values = []
    with torch.no_grad():
        for i in range(0, len(coords), batch_size):
            batch = coords[i:i+batch_size]
            batch_tensor = torch.from_numpy(batch).float().cuda()
            model_input = {'coords': batch_tensor.unsqueeze(0)}
            output = model(model_input)
            sdf = output['model_out'].squeeze().cpu().numpy()
            sdf_values.append(sdf)
    return np.concatenate(sdf_values)

def plot_slice_to_base64(X, Y, sdf_grid, z_value, num_levels=30, global_vmin=None, global_vmax=None):
    """绘制 XY 平面的 SDF 等高线"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 统一颜色范围
    if global_vmin is not None and global_vmax is not None:
        abs_max = max(abs(global_vmin), abs(global_vmax))
    else:
        abs_max = max(abs(sdf_grid.min()), abs(sdf_grid.max()))

    # [修改] 使用对称的对数级别，让 0 附近更密
    # num_levels 建议为偶数；若为奇数会自动补 0
    n_side = max(3, num_levels // 2)  # 每侧至少 3 个级别
    eps = abs_max * 1e-4 + 1e-8       # 防止 log(0)
    pos = np.geomspace(eps, abs_max, n_side)
    neg = -pos[::-1]
    if num_levels % 2 == 0:
        levels = np.concatenate([neg, pos])
    else:
        levels = np.concatenate([neg, [0.0], pos])

    # 绘制填充等高线
    contourf = ax.contourf(X, Y, sdf_grid, levels=levels, cmap='RdYlBu_r', extend='both')
    # 绘制线条
    ax.contour(X, Y, sdf_grid, levels=levels, colors='black', linewidths=0.3, alpha=0.3)
    # 强调 0 等值线 (表面)
    ax.contour(X, Y, sdf_grid, levels=[0], colors='black', linewidths=2.0)
    
    cbar = plt.colorbar(contourf, ax=ax, shrink=0.8)
    cbar.set_label('SDF Value')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Z = {z_value:.5f}')
    ax.set_aspect('equal')
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

def generate_html(images_data, output_path, info_str):
    """生成简单的 HTML 查看器"""
    js_data = ",\n".join([f'{{z: "{z}", src: "data:image/png;base64,{img}"}}' for z, img in images_data])
    
    html = f'''
    <html>
    <body style="background:#1a1a1a; color:white; text-align:center; font-family:sans-serif;">
        <h2>SDF Z-Slice Viewer ({info_str})</h2>
        <div>
            <img id="viewer" style="max-width:800px; border:1px solid #444; border-radius:8px;">
        </div>
        <div style="margin:20px;">
            <input type="range" id="slider" min="0" max="{len(images_data)-1}" value="0" style="width:600px;">
        </div>
        <div id="label" style="font-size:1.5em; color:#00ccff;"></div>
        <script>
            const data = [{js_data}];
            const img = document.getElementById('viewer');
            const slider = document.getElementById('slider');
            const label = document.getElementById('label');
            
            function update() {{
                const idx = slider.value;
                img.src = data[idx].src;
                label.innerText = "Z = " + data[idx].z;
            }}
            
            slider.oninput = update;
            update();
            
            // 键盘左右键控制
            document.addEventListener('keydown', function(e) {{
                if(e.key === "ArrowLeft") {{ slider.value = Math.max(0, parseInt(slider.value)-1); update(); }}
                if(e.key === "ArrowRight") {{ slider.value = Math.min(data.length-1, parseInt(slider.value)+1); update(); }}
            }});
        </script>
    </body>
    </html>
    '''
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    # 路径相关
    parser.add_argument('--logs_dir', type=str, required=True)
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--iter_idx', type=int, default=0, help="要查看哪一次迭代的模型")
    
    # 范围相关
    parser.add_argument('--z_range', type=float, nargs=2, required=True, help="Z start Z end")
    parser.add_argument('--num_slices', type=int, default=50, help="切多少片")
    parser.add_argument('--xy_range', type=float, nargs=2, default=[-0.15, 0.15], help="XY平面的可视范围")
    
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--output', type=str, default="sdf_slices.html")
    
    args = parser.parse_args()

    # 1. 寻找模型路径
    logs_dir = Path(args.logs_dir)
    # 兼容新旧路径逻辑
    ckpt_candidates = [
        logs_dir / args.exp_name / "02_iterations" / f"iter_{args.iter_idx:02d}" / "siren_checkpoints" / "checkpoints" / "model_final.pth",
        logs_dir / args.exp_name / "02_iterations" / f"iter_{args.iter_idx:02d}" / "checkpoints" / "model_final.pth",
        logs_dir / args.exp_name / f"{args.exp_name}_{args.iter_idx}" / "checkpoints" / "model_final.pth"
    ]
    
    model_path = None
    for p in ckpt_candidates:
        if p.exists():
            model_path = p
            break
            
    if not model_path:
        print(f"Error: Could not find model for iteration {args.iter_idx}")
        print(f"Searched: {ckpt_candidates}")
        return

    # 2. 加载模型 (只加载一次)
    model = load_model(str(model_path))

    # 3. 生成 Z 轴序列
    z_levels = np.linspace(args.z_range[0], args.z_range[1], args.num_slices)
    
    # 4. 计算所有切片
    print(f"Computing {len(z_levels)} slices from Z={args.z_range[0]} to {args.z_range[1]}...")
    
    # 第一遍：计算全局最值 (为了统一颜色条)
    global_min, global_max = float('inf'), float('-inf')
    slices_data = [] # 暂存 (z, field, X, Y)
    
    for i, z in enumerate(z_levels):
        print(f"\r  Pass 1 (Calc): {i+1}/{len(z_levels)}", end="")
        coords, X, Y = generate_xy_grid(z, tuple(args.xy_range), tuple(args.xy_range), args.resolution)
        sdf_vals = evaluate_sdf(model, coords)
        sdf_grid = sdf_vals.reshape(args.resolution, args.resolution)
        
        global_min = min(global_min, sdf_grid.min())
        global_max = max(global_max, sdf_grid.max())
        slices_data.append((z, sdf_grid, X, Y))
        
    print(f"\nGlobal SDF Range: [{global_min:.4f}, {global_max:.4f}]")
    
    # 第二遍：绘图
    images_base64 = []
    for i, (z, sdf_grid, X, Y) in enumerate(slices_data):
        print(f"\r  Pass 2 (Plot): {i+1}/{len(z_levels)}", end="")
        img_str = plot_slice_to_base64(X, Y, sdf_grid, z, num_levels=30, 
                                     global_vmin=global_min, global_vmax=global_max)
        images_base64.append((f"{z:.5f}", img_str))

    print("\nGenerating HTML...")
    generate_html(images_base64, args.output, f"{args.exp_name} Iter {args.iter_idx}")

if __name__ == "__main__":
    main()