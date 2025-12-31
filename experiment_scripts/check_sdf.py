import os
import sys
import torch
import numpy as np
import configargparse
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import modules

def check_sdf():
    # 1. 配置参数 (请根据你的训练设置修改)
    p = configargparse.ArgumentParser()
    p.add_argument('--checkpoint_path', type=str, required=True, help='Path to the trained model checkpoint')
    p.add_argument('--point_cloud_path', type=str, required=True, help='Path to the .xyz file')
    p.add_argument('--hidden_features', type=int, default=256)
    p.add_argument('--num_hidden_layers', type=int, default=3)
    p.add_argument('--model_type', type=str, default='sine')
    opt = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 加载模型
    model = modules.SingleBVPNet(type=opt.model_type, in_features=3, 
                                 hidden_features=opt.hidden_features,
                                 num_hidden_layers=opt.num_hidden_layers)
    model.to(device)
    
    print(f"Loading checkpoint from {opt.checkpoint_path} ...")
    checkpoint = torch.load(opt.checkpoint_path)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # 3. 加载数据 (不走 DataLoader，直接读 numpy 方便操作)
    print(f"Loading point cloud from {opt.point_cloud_path} ...")
    point_cloud = np.genfromtxt(opt.point_cloud_path)
    
    coords = point_cloud[:, :3]
    
    # 假设第7列是 GT SDF
    if point_cloud.shape[1] > 6:
        gt_sdf = point_cloud[:, 6]
    else:
        print("Error: Point cloud does not have SDF column (7th column).")
        return

    # 检测第8列 is_deformed 标记
    if point_cloud.shape[1] > 7:
        is_deformed = point_cloud[:, 7] == 1
        print(f"检测到 is_deformed 标记列，膨胀相关点数: {np.sum(is_deformed)}")
    else:
        # 兼容旧格式
        is_deformed = gt_sdf < -1e-6
        print(f"未检测到标记列，使用 SDF<0 作为标记")

    # 4. 筛选不同类型的点
    # 类型1: 插值点 (SDF < 0, is_deformed = 1)
    mask_inner = (gt_sdf < -1e-6) & is_deformed
    # 类型2: 膨胀后的表面点 (SDF = 0, is_deformed = 1)
    mask_deformed_surface = (np.abs(gt_sdf) < 1e-6) & is_deformed
    # 类型3: 普通表面点 (SDF = 0, is_deformed = 0)
    mask_normal_surface = (np.abs(gt_sdf) < 1e-6) & (~is_deformed)
    
    inner_coords = coords[mask_inner]
    inner_gt = gt_sdf[mask_inner]
    
    deformed_surface_coords = coords[mask_deformed_surface]
    deformed_surface_gt = gt_sdf[mask_deformed_surface]
    
    normal_surface_coords = coords[mask_normal_surface]
    normal_surface_gt = gt_sdf[mask_normal_surface]

    print(f"\n{'='*60}")
    print(f"数据统计:")
    print(f"{'='*60}")
    print(f"总点数: {len(coords)}")
    print(f"  - 插值点 (SDF < 0): {len(inner_coords)}")
    print(f"  - 膨胀后表面点 (SDF = 0, is_deformed = 1): {len(deformed_surface_coords)}")
    print(f"  - 普通表面点 (SDF = 0, is_deformed = 0): {len(normal_surface_coords)}")

    # 5. 推理函数
    def predict_chunk(coords_chunk):
        model_input = {'coords': torch.from_numpy(coords_chunk).float().to(device)[None, ...]}
        with torch.no_grad():
            model_output = model(model_input)
        return model_output['model_out'].squeeze().cpu().numpy()

    # 6. 检查插值点 (Inner Points)
    if len(inner_coords) > 0:
        print(f"\n{'='*60}")
        print("检查插值点 (SDF < 0)")
        print(f"{'='*60}")
        pred_sdf_inner = []
        chunk_size = 10000
        for i in range(0, len(inner_coords), chunk_size):
            chunk = inner_coords[i:i+chunk_size]
            pred_sdf_inner.append(predict_chunk(chunk))
        pred_sdf_inner = np.concatenate(pred_sdf_inner)

        diff = pred_sdf_inner - inner_gt
        mae = np.mean(np.abs(diff))
        
        print(f"Mean Absolute Error (MAE): {mae:.6f}")
        print(f"Mean Predicted SDF: {np.mean(pred_sdf_inner):.6f} (Target avg: {np.mean(inner_gt):.6f})")
        print(f"Min Predicted SDF:  {np.min(pred_sdf_inner):.6f}")
        print(f"Max Predicted SDF:  {np.max(pred_sdf_inner):.6f}")
        
        num_negative = np.sum(pred_sdf_inner < 0)
        print(f"正确预测为负数: {num_negative} / {len(inner_coords)} ({num_negative/len(inner_coords)*100:.2f}%)")
        
        print("\n样本预测 (前10个):")
        for i in range(min(10, len(inner_coords))):
            status = "✓" if pred_sdf_inner[i] < 0 else "✗"
            print(f"  {status} GT: {inner_gt[i]:.6f} | Pred: {pred_sdf_inner[i]:.6f}")

    # 7. 检查膨胀后的表面点
    if len(deformed_surface_coords) > 0:
        print(f"\n{'='*60}")
        print("检查膨胀后表面点 (SDF = 0, is_deformed = 1)")
        print(f"{'='*60}")
        
        pred_sdf_deformed = []
        for i in range(0, len(deformed_surface_coords), chunk_size):
            chunk = deformed_surface_coords[i:i+chunk_size]
            pred_sdf_deformed.append(predict_chunk(chunk))
        pred_sdf_deformed = np.concatenate(pred_sdf_deformed)
        
        mae = np.mean(np.abs(pred_sdf_deformed - deformed_surface_gt))
        print(f"Mean Absolute Error (MAE): {mae:.6f}")
        print(f"Mean Predicted SDF: {np.mean(pred_sdf_deformed):.6f} (Target: 0)")
        print(f"Min Predicted SDF:  {np.min(pred_sdf_deformed):.6f}")
        print(f"Max Predicted SDF:  {np.max(pred_sdf_deformed):.6f}")
        
        # 关键指标：有多少膨胀点预测为正值（表面收缩）
        num_positive = np.sum(pred_sdf_deformed > 0.001)
        num_negative = np.sum(pred_sdf_deformed < -0.001)
        num_near_zero = np.sum(np.abs(pred_sdf_deformed) <= 0.001)
        
        print(f"\n膨胀点预测分布:")
        print(f"  - 预测 > 0.001 (表面收缩): {num_positive} ({num_positive/len(deformed_surface_coords)*100:.2f}%)")
        print(f"  - 预测 ≈ 0 (正确): {num_near_zero} ({num_near_zero/len(deformed_surface_coords)*100:.2f}%)")
        print(f"  - 预测 < -0.001 (过度膨胀): {num_negative} ({num_negative/len(deformed_surface_coords)*100:.2f}%)")
        
        print("\n样本预测 (前10个):")
        for i in range(min(10, len(deformed_surface_coords))):
            status = "✓" if np.abs(pred_sdf_deformed[i]) <= 0.001 else "✗"
            print(f"  {status} GT: {deformed_surface_gt[i]:.6f} | Pred: {pred_sdf_deformed[i]:.6f}")

    # 8. 检查普通表面点
    if len(normal_surface_coords) > 0:
        print(f"\n{'='*60}")
        print("检查普通表面点 (SDF = 0, is_deformed = 0)")
        print(f"{'='*60}")
        
        # 随机采样 10000 个点
        if len(normal_surface_coords) > 10000:
            idcs = np.random.choice(len(normal_surface_coords), 10000, replace=False)
            check_coords = normal_surface_coords[idcs]
        else:
            check_coords = normal_surface_coords
            
        pred_sdf_normal = predict_chunk(check_coords)
        
        print(f"Mean Absolute Error (MAE): {np.mean(np.abs(pred_sdf_normal)):.6f}")
        print(f"Mean Predicted SDF: {np.mean(pred_sdf_normal):.6f} (Target: 0)")
        print(f"Min Predicted SDF:  {np.min(pred_sdf_normal):.6f}")
        print(f"Max Predicted SDF:  {np.max(pred_sdf_normal):.6f}")
        
if __name__ == '__main__':
    check_sdf()
