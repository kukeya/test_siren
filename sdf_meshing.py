'''From the DeepSDF repository https://github.com/facebookresearch/DeepSDF
'''
#!/usr/bin/env python3

import logging
import numpy as np
import plyfile
import skimage.measure
import time
import torch


def create_mesh(
    decoder,
    filename,
    N=256,
    max_batch=64 ** 3,
    offset=None,
    scale=None,
    use_amp=False,
):
    start = time.time()
    ply_filename = filename

    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    # voxel_origin = [-1, -1, -1]
    # voxel_size = 2.0 / (N - 1)

    voxel_origin = [-1.1, -1.1, -1.1]
    voxel_size = 2.2 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = ((overall_index.long() // N) % N).to(samples.dtype)
    samples[:, 0] = ((overall_index.long() // (N * N)) % N).to(samples.dtype)

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False
    in_bounds_mask = (samples[:, 0:3].abs() <= 1.0).all(dim=1)

    head = 0

    with torch.inference_mode():
        while head < num_samples:
            tail = min(head + max_batch, num_samples)
            batch_coords = samples[head:tail, 0:3]
            batch_in_bounds = in_bounds_mask[head:tail]
            batch_sdf = torch.ones((tail - head,), dtype=torch.float32)

            if torch.any(batch_in_bounds):
                sample_subset = batch_coords[batch_in_bounds].cuda(non_blocking=True)
                if use_amp:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        pred = decoder(sample_subset).squeeze(-1)
                else:
                    pred = decoder(sample_subset).squeeze(-1)
                batch_sdf[batch_in_bounds] = pred.detach().float().cpu()

            samples[head:tail, 3] = batch_sdf
            head = tail

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    print("sampling takes: %f" % (end - start))

    convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        ply_filename + ".ply",
        offset,
        scale,
    )


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    mc_start = time.time()
    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes(
            numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
        )
    except:
        pass
    print("marching_cubes takes: %f" % (time.time() - mc_start))

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.empty((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    if num_verts > 0:
        verts_tuple["x"] = mesh_points[:, 0].astype(np.float32, copy=False)
        verts_tuple["y"] = mesh_points[:, 1].astype(np.float32, copy=False)
        verts_tuple["z"] = mesh_points[:, 2].astype(np.float32, copy=False)

    faces_tuple = np.empty((num_faces,), dtype=[("vertex_indices", "i4", (3,))])
    if num_faces > 0:
        faces_tuple["vertex_indices"] = faces.astype(np.int32, copy=False)

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)
    print("ply_write takes: %f" % (time.time() - start_time))

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )
