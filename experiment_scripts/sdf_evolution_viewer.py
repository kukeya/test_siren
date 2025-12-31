"""
SDF 截面动画/对比可视化工具
生成多个迭代的 SDF 截面图，输出为可交互的 HTML 文件
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
from matplotlib import cm


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
    return model


def generate_slice_grid(y_value: float, x_range: tuple, z_range: tuple, resolution: int):
    """在 y=y_value 平面上生成采样网格"""
    x = np.linspace(x_range[0], x_range[1], resolution)
    z = np.linspace(z_range[0], z_range[1], resolution)
    X, Z = np.meshgrid(x, z)
    
    x_flat = X.flatten()
    z_flat = Z.flatten()
    y_flat = np.full_like(x_flat, y_value)
    
    coords = np.stack([x_flat, y_flat, z_flat], axis=1)
    return coords, X, Z


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


def plot_sdf_contour_to_base64(X, Z, sdf_grid, num_levels: int = 30, 
                                y_value: float = 0.0, title: str = None,
                                global_vmin: float = None, global_vmax: float = None):
    """
    绘制 SDF 等高线图并返回 base64 编码的图片
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 使用全局范围以保持一致的颜色映射
    if global_vmin is not None and global_vmax is not None:
        abs_max = max(abs(global_vmin), abs(global_vmax))
        levels = np.linspace(-abs_max, abs_max, num_levels)
    else:
        vmin, vmax = sdf_grid.min(), sdf_grid.max()
        if vmin < 0 < vmax:
            abs_max = max(abs(vmin), abs(vmax))
            levels = np.linspace(-abs_max, abs_max, num_levels)
        else:
            levels = np.linspace(vmin, vmax, num_levels)
    
    # 填充等高线
    contourf = ax.contourf(X, Z, sdf_grid, levels=levels, cmap='RdYlBu_r', extend='both')
    
    # 等高线轮廓
    ax.contour(X, Z, sdf_grid, levels=levels, colors='black', linewidths=0.3, alpha=0.5)
    
    # 零等值面高亮
    zero_contour = ax.contour(X, Z, sdf_grid, levels=[0], colors='black', linewidths=2)
    ax.clabel(zero_contour, inline=True, fontsize=10, fmt='0')
    
    cbar = plt.colorbar(contourf, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('SDF Value', fontsize=10)
    
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Z', fontsize=10)
    if title:
        ax.set_title(title, fontsize=12)
    else:
        ax.set_title(f'SDF Cross-Section at Y = {y_value:.6f}', fontsize=12)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    # 保存到 BytesIO 并转换为 base64
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return img_base64


def generate_html_viewer(images_data: list, output_path: str, y_value: float):
    """
    生成带滑块的 HTML 查看器
    
    Args:
        images_data: [(iteration_name, base64_img), ...]
        output_path: 输出 HTML 路径
        y_value: 截面 Y 值
    """
    
    # 生成 JavaScript 图片数组
    js_images = ",\n            ".join([
        f'{{name: "{name}", src: "data:image/png;base64,{img}"}}'
        for name, img in images_data
    ])
    
    html_template = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SDF Evolution Viewer - Y = {y_value:.6f}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            padding: 20px;
            color: #fff;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            margin-bottom: 10px;
            font-size: 1.8em;
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .subtitle {{
            text-align: center;
            color: #888;
            margin-bottom: 30px;
        }}
        .viewer {{
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 30px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .image-container {{
            text-align: center;
            margin-bottom: 20px;
        }}
        .image-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }}
        .controls {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 15px;
        }}
        .slider-container {{
            width: 100%;
            max-width: 600px;
        }}
        .slider {{
            width: 100%;
            height: 8px;
            -webkit-appearance: none;
            background: linear-gradient(90deg, #00d4ff 0%, #7b2cbf 100%);
            border-radius: 4px;
            outline: none;
            cursor: pointer;
        }}
        .slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 24px;
            height: 24px;
            background: #fff;
            border-radius: 50%;
            cursor: pointer;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }}
        .slider::-moz-range-thumb {{
            width: 24px;
            height: 24px;
            background: #fff;
            border-radius: 50%;
            cursor: pointer;
            border: none;
        }}
        .info {{
            display: flex;
            align-items: center;
            gap: 20px;
            flex-wrap: wrap;
            justify-content: center;
        }}
        .current-label {{
            font-size: 1.5em;
            font-weight: bold;
            color: #00d4ff;
            min-width: 120px;
            text-align: center;
        }}
        .btn-group {{
            display: flex;
            gap: 10px;
        }}
        button {{
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            transition: all 0.3s;
        }}
        .btn-nav {{
            background: rgba(255,255,255,0.1);
            color: #fff;
        }}
        .btn-nav:hover {{
            background: rgba(255,255,255,0.2);
        }}
        .btn-play {{
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
            color: #fff;
            min-width: 100px;
        }}
        .btn-play:hover {{
            transform: scale(1.05);
        }}
        .speed-control {{
            display: flex;
            align-items: center;
            gap: 10px;
            color: #888;
        }}
        .speed-control select {{
            padding: 5px 10px;
            border-radius: 4px;
            border: 1px solid rgba(255,255,255,0.2);
            background: rgba(255,255,255,0.1);
            color: #fff;
        }}
        .progress-bar {{
            width: 100%;
            max-width: 600px;
            display: flex;
            gap: 4px;
            margin-top: 10px;
        }}
        .progress-dot {{
            flex: 1;
            height: 4px;
            background: rgba(255,255,255,0.2);
            border-radius: 2px;
            cursor: pointer;
            transition: all 0.3s;
        }}
        .progress-dot.active {{
            background: #00d4ff;
        }}
        .progress-dot:hover {{
            background: rgba(255,255,255,0.5);
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 SDF Evolution Viewer</h1>
        <p class="subtitle">Cross-Section at Y = {y_value:.6f} | Total {len(images_data)} iterations</p>
        
        <div class="viewer">
            <div class="image-container">
                <img id="sdfImage" src="" alt="SDF Slice">
            </div>
            
            <div class="controls">
                <div class="slider-container">
                    <input type="range" id="slider" class="slider" min="0" max="{len(images_data)-1}" value="0">
                </div>
                
                <div class="progress-bar" id="progressBar"></div>
                
                <div class="info">
                    <div class="btn-group">
                        <button class="btn-nav" onclick="prev()">◀ Prev</button>
                        <button class="btn-play" id="playBtn" onclick="togglePlay()">▶ Play</button>
                        <button class="btn-nav" onclick="next()">Next ▶</button>
                    </div>
                    <div class="current-label" id="currentLabel">exp16_0</div>
                    <div class="speed-control">
                        <label>Speed:</label>
                        <select id="speedSelect" onchange="updateSpeed()">
                            <option value="2000">0.5x</option>
                            <option value="1000" selected>1x</option>
                            <option value="500">2x</option>
                            <option value="250">4x</option>
                        </select>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const images = [
            {js_images}
        ];
        
        let currentIndex = 0;
        let isPlaying = false;
        let playInterval = null;
        let speed = 1000;
        
        const slider = document.getElementById('slider');
        const img = document.getElementById('sdfImage');
        const label = document.getElementById('currentLabel');
        const playBtn = document.getElementById('playBtn');
        const progressBar = document.getElementById('progressBar');
        
        // 初始化进度点
        images.forEach((_, i) => {{
            const dot = document.createElement('div');
            dot.className = 'progress-dot' + (i === 0 ? ' active' : '');
            dot.onclick = () => goTo(i);
            progressBar.appendChild(dot);
        }});
        
        function updateDisplay() {{
            img.src = images[currentIndex].src;
            label.textContent = images[currentIndex].name;
            slider.value = currentIndex;
            
            // 更新进度点
            document.querySelectorAll('.progress-dot').forEach((dot, i) => {{
                dot.classList.toggle('active', i <= currentIndex);
            }});
        }}
        
        function goTo(index) {{
            currentIndex = Math.max(0, Math.min(images.length - 1, index));
            updateDisplay();
        }}
        
        function prev() {{
            goTo(currentIndex - 1);
        }}
        
        function next() {{
            goTo(currentIndex + 1);
        }}
        
        function togglePlay() {{
            isPlaying = !isPlaying;
            playBtn.textContent = isPlaying ? '⏸ Pause' : '▶ Play';
            
            if (isPlaying) {{
                playInterval = setInterval(() => {{
                    if (currentIndex >= images.length - 1) {{
                        currentIndex = 0;
                    }} else {{
                        currentIndex++;
                    }}
                    updateDisplay();
                }}, speed);
            }} else {{
                clearInterval(playInterval);
            }}
        }}
        
        function updateSpeed() {{
            speed = parseInt(document.getElementById('speedSelect').value);
            if (isPlaying) {{
                clearInterval(playInterval);
                playInterval = setInterval(() => {{
                    if (currentIndex >= images.length - 1) {{
                        currentIndex = 0;
                    }} else {{
                        currentIndex++;
                    }}
                    updateDisplay();
                }}, speed);
            }}
        }}
        
        slider.oninput = function() {{
            goTo(parseInt(this.value));
        }};
        
        // 键盘控制
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'ArrowLeft') prev();
            if (e.key === 'ArrowRight') next();
            if (e.key === ' ') {{ e.preventDefault(); togglePlay(); }}
        }});
        
        // 初始化
        updateDisplay();
    </script>
</body>
</html>
'''
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print(f"HTML viewer saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='SDF Evolution Viewer Generator')
    parser.add_argument('--exp_name', type=str, default='exp16',
                        help='Experiment name prefix')
    parser.add_argument('--start', type=int, default=0,
                        help='Start iteration number')
    parser.add_argument('--end', type=int, default=16,
                        help='End iteration number')
    parser.add_argument('--logs_dir', type=str, 
                        default='/home/group1/jym/Repos/test_siren/logs',
                        help='Logs directory')
    parser.add_argument('--y_value', type=float, default=0.133856,
                        help='Y coordinate of the slice plane')
    parser.add_argument('--x_range', type=float, nargs=2, default=[-0.2, 0.2],
                        help='X range: min max')
    parser.add_argument('--z_range', type=float, nargs=2, default=[0.0, 0.4],
                        help='Z range: min max')
    parser.add_argument('--resolution', type=int, default=256,
                        help='Grid resolution (lower = faster)')
    parser.add_argument('--num_levels', type=int, default=30,
                        help='Number of contour levels')
    parser.add_argument('--output', type=str, default=None,
                        help='Output HTML path')
    parser.add_argument('--hidden_features', type=int, default=256,
                        help='Model hidden features')
    parser.add_argument('--num_hidden_layers', type=int, default=3,
                        help='Model hidden layers')
    
    args = parser.parse_args()
    
    logs_dir = Path(args.logs_dir)
    exp_dir = logs_dir / args.exp_name
    
    # 收集所有 checkpoint 路径
    checkpoints = []
    for i in range(args.start, args.end + 1):
        ckpt_path = exp_dir / f"{args.exp_name}_{i}" / "checkpoints" / "model_final.pth"
        if ckpt_path.exists():
            checkpoints.append((f"{args.exp_name}_{i}", str(ckpt_path)))
        else:
            print(f"Warning: {ckpt_path} not found, skipping...")
    
    if not checkpoints:
        print("Error: No checkpoints found!")
        return
    
    print(f"Found {len(checkpoints)} checkpoints")
    
    # 生成网格（只需一次）
    print(f"\nGenerating grid at Y = {args.y_value}...")
    print(f"X range: {args.x_range}, Z range: {args.z_range}")
    print(f"Resolution: {args.resolution} x {args.resolution}")
    
    coords, X, Z = generate_slice_grid(
        y_value=args.y_value,
        x_range=tuple(args.x_range),
        z_range=tuple(args.z_range),
        resolution=args.resolution
    )
    
    # 第一遍：计算全局 SDF 范围
    print("\nPass 1: Computing global SDF range...")
    global_min, global_max = float('inf'), float('-inf')
    sdf_grids = []
    
    for name, ckpt_path in checkpoints:
        print(f"  Loading {name}...")
        model = load_model(ckpt_path, args.hidden_features, args.num_hidden_layers)
        sdf_values = evaluate_sdf(model, coords)
        sdf_grid = sdf_values.reshape(args.resolution, args.resolution)
        sdf_grids.append((name, sdf_grid))
        global_min = min(global_min, sdf_grid.min())
        global_max = max(global_max, sdf_grid.max())
        del model
        torch.cuda.empty_cache()
    
    print(f"Global SDF range: [{global_min:.4f}, {global_max:.4f}]")
    
    # 第二遍：生成图片
    print("\nPass 2: Generating images...")
    images_data = []
    
    for name, sdf_grid in sdf_grids:
        print(f"  Rendering {name}...")
        img_base64 = plot_sdf_contour_to_base64(
            X, Z, sdf_grid,
            num_levels=args.num_levels,
            y_value=args.y_value,
            title=f'{name} | Y = {args.y_value:.4f}',
            global_vmin=global_min,
            global_vmax=global_max
        )
        images_data.append((name, img_base64))
    
    # 生成 HTML
    output_path = args.output or f"sdf_evolution_{args.exp_name}.html"
    generate_html_viewer(images_data, output_path, args.y_value)
    
    print(f"\n✅ Done! Open {output_path} in a browser to view.")


if __name__ == '__main__':
    main()
