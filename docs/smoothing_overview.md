# 平滑项公式与训练流程概览

本文件描述新增的 deform-only 平滑项（法线平滑与位置平滑）的数学形式，并用流程图概括当前训练/损失的整体思路。

## 1) 平滑项公式（新增）

设一批输入坐标为 \(\mathbf{x}_i \in \mathbb{R}^3\)，网络预测 SDF 为 \(f_\theta(\mathbf{x}_i)\)。  
预测法线（SDF 梯度方向）为：

\[
\mathbf{n}_i = \frac{\nabla_{\mathbf{x}} f_\theta(\mathbf{x}_i)}{\|\nabla_{\mathbf{x}} f_\theta(\mathbf{x}_i)\|_2}
\]

仅在可编辑点（deform 点）集合 \(\mathcal{D}=\{i \mid \text{is\_deform}_i=1\}\) 上计算平滑项。  
如果启用了投影，则将点投影到零等值面附近：

\[
\tilde{\mathbf{x}}_i = \mathbf{x}_i - \mathbf{n}_i \, f_\theta(\mathbf{x}_i)
\]

对每个 deform 点 \(i\)，在 \(\tilde{\mathbf{x}}\) 空间内选取 \(k\) 近邻（可选半径裁剪），得到邻域 \(\mathcal{N}(i)\)。  

### (a) 位置平滑（deform-only position smoothing）

\[
L_{\text{pos}} = \frac{1}{|\mathcal{D}|} \sum_{i \in \mathcal{D}}
\left\| \tilde{\mathbf{x}}_i - \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} \tilde{\mathbf{x}}_j \right\|_2^2
\]

> 代码对应：`smooth_position_*`（kNN Laplacian on projected positions）。【F:loss_function/loss_functions.py†L86-L124】【F:loss_function/loss_functions_L2.py†L82-L118】

### (b) 法线平滑（deform-only normal smoothing）

\[
L_{\text{n}} = \frac{1}{|\mathcal{D}|} \sum_{i \in \mathcal{D}}
\left\| \mathbf{n}_i - \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} \mathbf{n}_j \right\|_2^2
\]

若启用特征保护（基于 GT 法线变化）：

\[
w_i = \exp\left(-\frac{1}{2}\left(\frac{\operatorname{Var}_{j \in \mathcal{N}(i)}(1-\cos\angle(\mathbf{n}^{gt}_i,\mathbf{n}^{gt}_j))}{\sigma}\right)^2\right)
\]

\[
L_{\text{n}} = \frac{1}{|\mathcal{D}|} \sum_{i \in \mathcal{D}} w_i \left\| \mathbf{n}_i - \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} \mathbf{n}_j \right\|_2^2
\]

> 代码对应：`smooth_normal_*`（kNN Laplacian on predicted normals + optional feature weight）。【F:loss_function/loss_functions.py†L40-L84】【F:loss_function/loss_functions_L2.py†L40-L80】

### (c) 总损失中的使用方式

\[
L = L_{\text{sdf}} + L_{\text{inter}} + L_{\text{normal}} + L_{\text{eikonal}} + \lambda_{\text{tp}} L_{\text{thin-plate}} + \lambda_{\text{n}} L_{\text{n}} + \lambda_{\text{pos}} L_{\text{pos}}
\]

其中 \(\lambda_{\text{n}}, \lambda_{\text{pos}}\) 由 `smooth_normal_weight` 与 `smooth_position_weight` 控制。  
thin-plate 平滑使用既有实现与 `thin_plate_mask`，保持原逻辑不变。【F:loss_function/loss_functions.py†L126-L153】

---

## 2) 训练/损失整体思路流程图

```mermaid
flowchart TD
    A[点云数据/采样点] --> B[DataLoader\n(is_deform, normals, sdf, thin_plate_mask)]
    B --> C[模型前向\nSDF fθ(x)]
    C --> D[梯度计算\n∇x fθ(x)]
    D --> E[基础损失\nSDF/normal/inter/eikonal]
    D --> F{可选：deform-only 平滑?}
    F -->|smooth_normal| G[法线 kNN Laplacian\n(可投影+特征权重)]
    F -->|smooth_position| H[位置 kNN Laplacian\n(可投影)]
    E --> I[总损失加权求和]
    G --> I
    H --> I
    C --> J{可选：thin-plate?}
    J --> K[薄板平滑\nHessian/curvature + thin_plate_mask]
    K --> I
    I --> L[反向传播/优化]
```

> 相关实现集中在 `loss_function/loss_functions.py` 与 `loss_function/loss_functions_L2.py`（L1/L2 版本）。【F:loss_function/loss_functions.py†L1-L153】【F:loss_function/loss_functions_L2.py†L1-L118】
