import torch
import diff_operators
import torch.nn.functional as F

def sdf(model_output, gt, sdf_weight=3e3, inter_weight=1e2, normal_weight=1e2, grad_weight=5e1,
        thin_plate_weight=0.0):
    '''
       x: batch of input coordinates
       y: usually the output of the trial_soln function
       '''
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    gradient = diff_operators.gradient(pred_sdf, coords)

    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    sdf_constraint = torch.where(gt_sdf != -1, pred_sdf, torch.zeros_like(pred_sdf))
    inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
    normal_constraint = torch.where(gt_sdf != -1, 1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None],
                                    torch.zeros_like(gradient[..., :1]))
    grad_constraint = torch.abs(gradient.norm(dim=-1) - 1)
    

    losses = {'sdf': (torch.abs(sdf_constraint)).mean() * sdf_weight,   ## L1 loss for SDF
              # (sdf_constraint**2).mean() * 1e2 * sdf_weight, ## L2 loss for SDF
              'inter': inter_constraint.mean() * inter_weight, 
              'normal_constraint': normal_constraint.mean() * normal_weight,
              'grad_constraint': grad_constraint.mean() * grad_weight}

    if thin_plate_weight > 0.0:
        hessian, _ = diff_operators.hessian(pred_sdf, coords)
        hessian = hessian[..., 0, :, :]
        grad_norm = gradient.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        trace_h = torch.diagonal(hessian, dim1=-2, dim2=-1).sum(-1, keepdim=True)
        g_col = gradient.unsqueeze(-1)
        g_row = gradient.unsqueeze(-2)
        g_h_g = torch.matmul(torch.matmul(g_row, hessian), g_col).squeeze(-1)
        mean_curv = (g_h_g - (grad_norm ** 2) * trace_h) / (2.0 * (grad_norm ** 3))
        matrix = torch.zeros(*hessian.shape[:-2], 4, 4, device=hessian.device, dtype=hessian.dtype)
        matrix[..., :3, :3] = hessian
        matrix[..., :3, 3] = gradient
        matrix[..., 3, :3] = gradient
        det_val = torch.det(matrix)
        gaussian_curv = -det_val / (grad_norm.squeeze(-1) ** 4 + 1e-8)
        smoothness = 4.0 * mean_curv.pow(2) - 2.0 * gaussian_curv.unsqueeze(-1)
        thin_plate_mask = gt.get('thin_plate_mask', None)
        if thin_plate_mask is not None:
            thin_plate_mask = thin_plate_mask.float()
            weight_sum = thin_plate_mask.sum().clamp_min(1e-8)
            thin_plate_loss = (smoothness * thin_plate_mask).sum() / weight_sum
        else:
            thin_plate_loss = smoothness.mean()
        thin_plate_loss = thin_plate_loss * thin_plate_weight
        losses['thin_plate'] = thin_plate_loss

    return losses
