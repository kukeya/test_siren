import argparse
import os

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required for visualization.") from exc


def compute_mask(points, deform_points, sigma, radius=None):
    diff = points[:, None, :] - deform_points[None, :, :]
    dist_min = np.linalg.norm(diff, axis=-1).min(axis=1, keepdims=True)
    mask = np.exp(-0.5 * (dist_min / sigma) ** 2)
    if radius is not None:
        mask = np.where(dist_min <= radius, mask, np.zeros_like(mask))
    return mask.reshape(-1)


def axis_index(axis_name):
    axis_map = {"x": 0, "y": 1, "z": 2}
    if axis_name not in axis_map:
        raise ValueError(f"axis must be one of {sorted(axis_map)}")
    return axis_map[axis_name]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--point_cloud_path", required=True, type=str)
    parser.add_argument("--axis", default="z", type=str, help="Slice axis: x, y, or z")
    parser.add_argument("--slice_value", default=0.0, type=float, help="Slice position in normalized space")
    parser.add_argument("--resolution", default=256, type=int, help="Grid resolution per axis")
    parser.add_argument("--thin_plate_radius", default=0.05, type=float)
    parser.add_argument("--thin_plate_sigma", default=None, type=float)
    parser.add_argument("--output_dir", default="outputs", type=str)
    args = parser.parse_args()

    if args.thin_plate_sigma is None:
        args.thin_plate_sigma = args.thin_plate_radius / 2.0

    point_cloud = np.genfromtxt(args.point_cloud_path)
    coords = point_cloud[:, :3]
    if point_cloud.shape[1] > 6:
        is_deform = point_cloud[:, 6:7]
    else:
        is_deform = np.zeros((coords.shape[0], 1), dtype=np.float32)

    deform_points = coords[is_deform.reshape(-1) == 1]
    if deform_points.shape[0] == 0:
        raise RuntimeError("No deform points found (is_deform == 1).")

    axis_idx = axis_index(args.axis)
    grid_axes = [np.linspace(-1.0, 1.0, args.resolution) for _ in range(3)]
    grid_axes[axis_idx] = np.array([args.slice_value])
    gx, gy, gz = np.meshgrid(*grid_axes, indexing="xy")
    grid_points = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    mask = compute_mask(
        grid_points,
        deform_points,
        sigma=args.thin_plate_sigma,
        radius=args.thin_plate_radius,
    )

    other_axes = [0, 1, 2]
    other_axes.remove(axis_idx)
    mask_img = mask.reshape(args.resolution, args.resolution)

    os.makedirs(args.output_dir, exist_ok=True)
    out_base = os.path.join(
        args.output_dir,
        f"thin_plate_mask_{args.axis}_{args.slice_value:.3f}_r{args.thin_plate_radius:.3f}_s{args.thin_plate_sigma:.3f}",
    )

    plt.figure(figsize=(6, 5))
    plt.imshow(mask_img, origin="lower", cmap="magma", extent=[-1, 1, -1, 1])
    plt.colorbar(label="thin_plate_mask")
    plt.title(f"Thin-plate mask slice {args.axis}={args.slice_value:.3f}")
    plt.xlabel(["x", "y", "z"][other_axes[0]])
    plt.ylabel(["x", "y", "z"][other_axes[1]])
    plt.tight_layout()
    plt.savefig(f"{out_base}.png", dpi=150)
    np.save(f"{out_base}.npy", mask_img)
    print(f"Saved: {out_base}.png")
    print(f"Saved: {out_base}.npy")


if __name__ == "__main__":
    main()