import torch
import diff_operators
import torch.nn.functional as F

def sdf(model_output, gt, sdf_weight=3e3, inter_weight=1e2, normal_weight=1e2, grad_weight=5e1,
        smooth_normal_weight=0.0, smooth_normal_k=16, smooth_normal_radius=None,
        smooth_normal_use_projection=False, smooth_normal_feat_sigma=None, smooth_normal_max_points=2048,
        smooth_position_weight=0.0, smooth_position_k=16, smooth_position_radius=None,
        smooth_position_use_projection=True, smooth_position_max_points=2048):
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
    

    # Exp      # Lapl
    # -----------------
    losses = {'sdf': (sdf_constraint ** 2).mean() * 1e3 * sdf_weight, ## L2 loss for SDF
              'inter': inter_constraint.mean() * inter_weight,
              'normal_constraint': normal_constraint.mean() * normal_weight,
              'grad_constraint': grad_constraint.mean() * grad_weight}

    if smooth_normal_weight > 0.0:
        is_deform = gt.get('is_deform', None)
        if is_deform is not None:
            deform_idx = torch.nonzero(is_deform.view(-1) == 1, as_tuple=False).view(-1)
            if smooth_normal_max_points is not None and deform_idx.numel() > smooth_normal_max_points:
                perm = torch.randperm(deform_idx.numel(), device=deform_idx.device)[:smooth_normal_max_points]
                deform_idx = deform_idx.index_select(0, perm)
            if deform_idx.numel() >= 2:
                x_edit = coords.index_select(0, deform_idx)
                n_pred = F.normalize(gradient.index_select(0, deform_idx), dim=-1)
                if smooth_normal_use_projection:
                    sdf_edit = pred_sdf.index_select(0, deform_idx)
                    x_edit = x_edit - n_pred * sdf_edit
                dist = torch.cdist(x_edit, x_edit)
                dist = torch.where(dist < 1e-8, torch.full_like(dist, float("inf")), dist)
                if smooth_normal_radius is not None:
                    dist = torch.where(
                        dist <= smooth_normal_radius,
                        dist,
                        torch.full_like(dist, float("inf")),
                    )
                k = min(int(smooth_normal_k), max(dist.shape[1] - 1, 1))
                if k > 0:
                    idx = dist.topk(k, largest=False).indices
                    n_nb = n_pred.index_select(0, idx.reshape(-1)).reshape(idx.shape[0], k, -1)
                    n_mean = n_nb.mean(dim=1)
                    per_point = (n_pred - n_mean).pow(2).sum(dim=-1)
                    if smooth_normal_feat_sigma is not None and smooth_normal_feat_sigma > 0.0:
                        n_gt = F.normalize(gt_normals.index_select(0, deform_idx), dim=-1)
                        n_gt_nb = n_gt.index_select(0, idx.reshape(-1)).reshape(idx.shape[0], k, -1)
                        cos_sim = (n_gt.unsqueeze(1) * n_gt_nb).sum(dim=-1).clamp(-1.0, 1.0)
                        angle_var = (1.0 - cos_sim).mean(dim=1)
                        w_feat = torch.exp(-0.5 * (angle_var / smooth_normal_feat_sigma) ** 2)
                        per_point = per_point * w_feat
                    smooth_normal_loss = per_point.mean()
                else:
                    smooth_normal_loss = pred_sdf.mean() * 0.0
            else:
                smooth_normal_loss = pred_sdf.mean() * 0.0
            losses['smooth_normal'] = smooth_normal_loss * smooth_normal_weight

    if smooth_position_weight > 0.0:
        is_deform = gt.get('is_deform', None)
        if is_deform is not None:
            deform_idx = torch.nonzero(is_deform.view(-1) == 1, as_tuple=False).view(-1)
            if smooth_position_max_points is not None and deform_idx.numel() > smooth_position_max_points:
                perm = torch.randperm(deform_idx.numel(), device=deform_idx.device)[:smooth_position_max_points]
                deform_idx = deform_idx.index_select(0, perm)
            if deform_idx.numel() >= 2:
                x_edit = coords.index_select(0, deform_idx)
                if smooth_position_use_projection:
                    n_pred = F.normalize(gradient.index_select(0, deform_idx), dim=-1)
                    sdf_edit = pred_sdf.index_select(0, deform_idx)
                    x_edit = x_edit - n_pred * sdf_edit
                dist = torch.cdist(x_edit, x_edit)
                dist = torch.where(dist < 1e-8, torch.full_like(dist, float("inf")), dist)
                if smooth_position_radius is not None:
                    dist = torch.where(
                        dist <= smooth_position_radius,
                        dist,
                        torch.full_like(dist, float("inf")),
                    )
                k = min(int(smooth_position_k), max(dist.shape[1] - 1, 1))
                if k > 0:
                    idx = dist.topk(k, largest=False).indices
                    x_nb = x_edit.index_select(0, idx.reshape(-1)).reshape(idx.shape[0], k, -1)
                    x_mean = x_nb.mean(dim=1)
                    smooth_position_loss = (x_edit - x_mean).pow(2).sum(dim=-1).mean()
                else:
                    smooth_position_loss = pred_sdf.mean() * 0.0
            else:
                smooth_position_loss = pred_sdf.mean() * 0.0
            losses['smooth_position'] = smooth_position_loss * smooth_position_weight

    return losses
