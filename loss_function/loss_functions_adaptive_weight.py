import torch
import diff_operators
import torch.nn.functional as F


def sdf(model_output, gt, sdf_weight=3e3, inter_weight=1e2, normal_weight=1e2, grad_weight=5e1,
        thin_plate_weight=0.0):
      gt_sdf = gt['sdf']
      gt_normals = gt['normals']
      coords = model_output['model_in']
      pred_sdf = model_output['model_out']
      
      # [新增] 获取权重，如果没有则默认为 1
      if 'weights' in gt:
         weights = gt['weights']
      else:
         weights = torch.ones_like(gt_sdf)
      
      if 'is_deform' in gt:
         is_deform = gt['is_deform']
   
      gradient = diff_operators.gradient(pred_sdf, coords)

      # SDF Loss (加权)
      # 限制在 [-1, 1] 之间计算 loss，避免离群点影响太大
      sdf_constraint = torch.where(gt_sdf != -1, torch.abs(pred_sdf - gt_sdf), torch.zeros_like(pred_sdf))
      # sign_mismatch = (gt_sdf < -1e-6) & (pred_sdf > 0)
      # sdf_constraint[sign_mismatch] *= 10.0  # 对符号不匹配的点增加惩罚

      # 情况1: SDF<0的点被预测为正值
      # 情况2: SDF=0的膨胀点被预测为较大正值（说明表面收缩了）
      sign_mismatch_inner = is_deform & (gt_sdf < -1e-6) & (pred_sdf > 0)
      sign_mismatch_surface = is_deform & (torch.abs(gt_sdf) < 1e-6) & (pred_sdf > 0.001)
      sign_mismatch = sign_mismatch_inner | sign_mismatch_surface
      sdf_constraint[sign_mismatch] *= 10.0  # 对符号不匹配的点增加惩罚

      inter_constraint = torch.where(gt_sdf == -1, torch.exp(-1e2 * torch.abs(pred_sdf)), torch.zeros_like(pred_sdf))
      normal_constraint = torch.where(gt_sdf != -1, 1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None],
                                    torch.zeros_like(gradient[..., :1]))
      
      # Grad Constraint (Eikonal)
      grad_constraint = torch.abs(gradient.norm(dim=-1) - 1)

      # print("sdf loss:", ((sdf_constraint * weights).mean() * sdf_weight).item())
   
      losses = {'sdf': ((sdf_constraint ** 2) * weights * 100).mean() * sdf_weight,
                'inter': inter_constraint.mean() * inter_weight, 
                'normal_constraint': normal_constraint.mean() * normal_weight,
                'grad_constraint': grad_constraint.mean() * grad_weight}

      if thin_plate_weight > 0.0:
         hessian, _ = diff_operators.hessian(pred_sdf, coords)
         thin_plate_loss = (hessian ** 2).mean() * thin_plate_weight
         losses['thin_plate'] = thin_plate_loss

      return losses
   
