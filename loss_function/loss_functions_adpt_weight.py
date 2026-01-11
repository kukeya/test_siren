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
    # [新增] 获取权重，如果没有则默认为 1
    weights = gt.get('weights', torch.ones_like(gt_sdf))

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    gradient = diff_operators.gradient(pred_sdf, coords)

    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    sdf_constraint = torch.where(gt_sdf != -1, pred_sdf, torch.zeros_like(pred_sdf))
    inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
    normal_constraint = torch.where(gt_sdf != -1, 1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None],
                                    torch.zeros_like(gradient[..., :1]))
    grad_constraint = torch.abs(gradient.norm(dim=-1) - 1)
    
    # [修改] 应用权重到 sdf_constraint
    # 注意：weights 形状可能需要广播，这里假设它是 (batch, 1)
    wc = sdf_constraint
    weighted_sdf_constraint = torch.where(weights > 1.0,
                                          torch.abs(wc),
                                          (wc ** 2) * weights)

    losses = {'sdf': weighted_sdf_constraint.mean() * sdf_weight,  
              'inter': inter_constraint.mean() * inter_weight, 
              'normal_constraint': normal_constraint.mean() * normal_weight,
              'grad_constraint': grad_constraint.mean() * grad_weight}

    if thin_plate_weight > 0.0:
        hessian, _ = diff_operators.hessian(pred_sdf, coords)
        thin_plate_loss = (hessian ** 2).mean() * thin_plate_weight
        losses['thin_plate'] = thin_plate_loss

    return losses
