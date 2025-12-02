import numpy as np
import configargparse

p = configargparse.ArgumentParser()
p.add_argument('--pointcloud_path', type=str, default='ruyi14w_n.xyz', help='root for logging')

args = p.parse_args()
pointcloud_path = args.pointcloud_path

point_cloud = np.genfromtxt(pointcloud_path)
print("Finished loading point cloud")
coords = point_cloud[:, :3]
normals = point_cloud[:, 3:]
# Reshape point cloud such that it lies in bounding box of (-1, 1) (distorts geometry, but makes for high
# sample efficiency)
coords -= np.mean(coords, axis=0, keepdims=True)
coord_max = np.amax(coords)
coord_min = np.amin(coords)
coords = (coords - coord_min) / (coord_max - coord_min)
coords -= 0.5
coords *= 2.

np.savetxt('normalized_' + pointcloud_path, np.concatenate((coords, normals), axis=1))
print("Finished saving normalized point cloud to: ", 'normalized_' + pointcloud_path)