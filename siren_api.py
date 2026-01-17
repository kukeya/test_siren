import argparse
from ast import arg
import os
import sys
import subprocess
from pathlib import Path

def run_siren_step(args):
    """
    Siren 标准化调用接口
    """
    python_exe = sys.executable 
    
    # 1. 基础路径定义
    input_pcd = Path(args.input_pcd).absolute()
    prev_ckpt = Path(args.prev_ckpt).absolute() if args.prev_ckpt else None
    
    # output_dir Receive: .../02_iterations/iter_00
    output_dir = Path(args.output_dir).absolute()
    
    # 2. [关键确定] 定义子文件夹名称，并拼接到最终 log_dir
    # 结果: .../02_iterations/iter_00/siren_checkpoints
    siren_ckpt_folder_name = "siren_checkpoints"
    final_log_dir = output_dir / siren_ckpt_folder_name 
    
    # 3. 定义最终生成的 PLY 位置 (供 IGR 下一步使用)
    output_ply = output_dir / "mesh_input"       
    
    print(f"--------------------------------------------------")
    print(f"[SirenAPI] Start")
    print(f"[SirenAPI] Input:         {input_pcd}")
    print(f"[SirenAPI] Target LogDir: {final_log_dir}")
    print(f"--------------------------------------------------")

    # ==========================================
    # 阶段 1: 训练 (Train)
    # ==========================================
    train_script = "experiment_scripts/train_sdf_weights.py"

    train_cmd = [
        python_exe, train_script,
        "--point_cloud_path", str(input_pcd),
        
        # [确定] 这里传入的是带子目录的完整路径
        # train_sdf_weights.py 会直接用这个路径作为 root，并在其下创建 checkpoints 文件夹
        "--log_dir", str(final_log_dir),
        
        # 这里的名字不再用于路径拼接，仅用于打印日志
        "--experiment_name", "siren_loop", 
        "--use_lr_decay",
        "--num_epochs", str(args.epochs),
        "--epochs_til_ckpt", "1000",
        "--steps_til_summary", "1000"
    ]
    if args.L2:
        train_cmd.append("--L2")
    
    if prev_ckpt and prev_ckpt.exists():
        train_cmd.extend(["--checkpoint_path", str(prev_ckpt)])

    if args.thin_plate_epochs and args.thin_plate_weight:
        train_cmd.extend(["--thin_plate_epochs", str(args.thin_plate_epochs)])
        train_cmd.extend(["--thin_plate_weight", str(args.thin_plate_weight)])

    if args.enable_thin_plate:
        train_cmd.extend(["--enable_thin_plate"])

    print(" ".join(train_cmd))
    
    sys.stdout.flush()
    ret = subprocess.call(train_cmd)
    
    if ret != 0:
        print("[SirenAPI] Error: Training Failed!")
        return False

    # ==========================================
    # 阶段 2: 测试/生成的 (Test/Meshing)
    # ==========================================
    # [确定] 这里去 final_log_dir 下的 checkpoints 找模型
    final_model = final_log_dir / "checkpoints" / "model_final.pth"
    
    if not final_model.exists():
        print(f"[SirenAPI] Error: Expected model file not found at: {final_model}")
        print(f"[SirenAPI] Please check if train_sdf_weights.py respected --log_dir")
        return False
        
    test_script = "experiment_scripts/test_sdf.py"
    test_cmd = [
        python_exe, test_script,
        "--checkpoint_path", str(final_model),
        "--output_ply", str(output_ply),
        "--experiment_name", "meshing"
    ]
    
    print("\n[SirenAPI] Running Meshing...")
    # 打印cmd
    print(" ".join(test_cmd))
    sys.stdout.flush()
    ret = subprocess.call(test_cmd)

    if ret != 0:
        print("[SirenAPI] Meshing Failed!")
        return False
        
    print(f"[SirenAPI] Success! Mesh generated at: {output_ply}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pcd", required=True)
    parser.add_argument("--prev_ckpt", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument('--L2', action='store_true', help='Use loss_function.loss_functionsL2 (L2)')
    parser.add_argument("--thin_plate_epochs", type=int, default=100, help="薄板能量正则化训练 Epochs (0 表示不使用)")
    parser.add_argument("--thin_plate_weight", type=float, default=5e-3, help="薄板能量正则化权重")

    parser.add_argument("--enable_thin_plate", action='store_true', help="薄板能量正则化权重")
    
    
    args = parser.parse_args()
    run_siren_step(args)