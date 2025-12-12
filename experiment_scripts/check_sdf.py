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

    # 4. 筛选出关键点 (膨胀点/负值点)
    # 假设你的膨胀点 SDF < -1e-6
    mask_inner = gt_sdf < -1e-6
    mask_surface = np.abs(gt_sdf) < 1e-6
    
    inner_coords = coords[mask_inner]
    print(f"Total inner in point cloud: {len(inner_coords)}")
    inner_gt = gt_sdf[mask_inner]
    
    surface_coords = coords[mask_surface]
    surface_gt = gt_sdf[mask_surface]

    print(f"\nTotal points: {len(coords)}")
    print(f"Inner/Deformed points (Target < 0): {len(inner_coords)}")
    print(f"Surface points (Target = 0): {len(surface_coords)}")

    # 5. 推理函数
    def predict_chunk(coords_chunk):
        model_input = {'coords': torch.from_numpy(coords_chunk).float().to(device)[None, ...]}
        with torch.no_grad():
            model_output = model(model_input)
        return model_output['model_out'].squeeze().cpu().numpy()

    # 6. 检查膨胀点 (Inner Points)
    if len(inner_coords) > 0:
        print("\n--- Checking Deformed/Inner Points ---")
        # 分块预测防止显存爆炸
        pred_sdf_inner = []
        chunk_size = 10000
        for i in range(0, len(inner_coords), chunk_size):
            chunk = inner_coords[i:i+chunk_size]
            pred_sdf_inner.append(predict_chunk(chunk))
        pred_sdf_inner = np.concatenate(pred_sdf_inner)

        # 统计误差
        diff = pred_sdf_inner - inner_gt
        mae = np.mean(np.abs(diff))
        
        print(f"Mean Absolute Error (MAE): {mae:.6f}")
        print(f"Mean Predicted SDF: {np.mean(pred_sdf_inner):.6f} (Target avg: {np.mean(inner_gt):.6f})")
        print(f"Min Predicted SDF:  {np.min(pred_sdf_inner):.6f}")
        print(f"Max Predicted SDF:  {np.max(pred_sdf_inner):.6f}")
        
        # 关键诊断：有多少点真的变成了负数？
        num_negative = np.sum(pred_sdf_inner < 0)
        print(f"Points successfully predicted < 0: {num_negative} / {len(inner_coords)} ({num_negative/len(inner_coords)*100:.2f}%)")
        
        # 打印前10个样本
        print("\nSample predictions (Target vs Pred):")
        for i in range(min(10, len(inner_coords))):
            print(f"  GT: {inner_gt[i]:.6f} | Pred: {pred_sdf_inner[i]:.6f} | Diff: {np.abs(inner_gt[i]-pred_sdf_inner[i]):.6f}")

    # 7. 检查表面点 (Surface Points)
    if len(surface_coords) > 0:
        print("\n--- Checking Surface Points ---")
        # 随机采样 10000 个点检查即可
        if len(surface_coords) > 10000:
            idcs = np.random.choice(len(surface_coords), 10000, replace=False)
            check_coords = surface_coords[idcs]
        else:
            check_coords = surface_coords
            
        pred_sdf_surface = predict_chunk(check_coords)
        
        print(f"Mean Absolute Error (MAE): {np.mean(np.abs(pred_sdf_surface)):.6f}")
        print(f"Mean Predicted SDF: {np.mean(pred_sdf_surface):.6f}")
        
if __name__ == '__main__':
    check_sdf()