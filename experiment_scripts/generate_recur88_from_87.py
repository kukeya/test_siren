#!/usr/bin/env python3
"""
生成膨胀点云：从 ruyi_recur87_n_deformed.xyz 沿法向外移指定距离，得到 ruyi_recur88_n_deformed.xyz。
默认距离: 0.0024。
用法示例：
    python generate_recur88_from_87.py \
        --input ruyi_recur87_n_deformed.xyz \
        --output ruyi_recur88_n_deformed.xyz \
        --distance 0.0024
"""
import argparse
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='ruyi_recur87_n_deformed.xyz', help='输入点云文件（含法向，至少6列）')
    parser.add_argument('--output', type=str, default='ruyi_recur88_n_deformed.xyz', help='输出膨胀后的点云文件')
    parser.add_argument('--distance', type=float, default=0.0024, help='沿法向外移的距离')
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    data = np.loadtxt(in_path)
    if data.shape[1] < 6:
        raise ValueError(f"Expect at least 6 columns (x y z nx ny nz), got {data.shape[1]}")

    coords = data[:, :3]
    normals = data[:, 3:6]
    extra = data[:, 6:] if data.shape[1] > 6 else None

    # 沿法向外移 distance
    new_coords = coords + args.distance * normals

    # 组合输出，保持原有列数
    if extra is not None and extra.size > 0:
        out_data = np.concatenate([new_coords, normals, extra], axis=1)
    else:
        out_data = np.concatenate([new_coords, normals], axis=1)

    np.savetxt(out_path, out_data, fmt='%.8f')
    print(f"Saved: {out_path} (N={len(out_data)}, distance={args.distance})")


if __name__ == '__main__':
    main()
