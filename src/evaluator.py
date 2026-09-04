import torch
import numpy as np
from config import DEVICE, CSI_THRESHOLDS
from torchmetrics.image import StructuralSimilarityIndexMeasure  # 更新导入路径
import torchmetrics

# 初始化 SSIM 指标（用于验证集累积）
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

# 初始化多个 CSI 指标
csi_metrics = {
    thresh: torchmetrics.JaccardIndex(task="binary", threshold=thresh).to(DEVICE)
    for thresh in CSI_THRESHOLDS
}

def calculate_metrics(predictions, targets, thresholds=CSI_THRESHOLDS):
    """
    计算评估指标：MSE, MAE, 多阈值 CSI, SSIM
    返回字典，包含每个阈值的 CSI 以及平均 SSIM、MSE、MAE
    """
    mse = torch.mean((predictions - targets) ** 2).item()
    mae = torch.mean(torch.abs(predictions - targets)).item()

    # SSIM：批量计算
    batch_size, time_steps, h, w = predictions.shape
    pred_flat = predictions.view(batch_size * time_steps, 1, h, w)
    target_flat = targets.view(batch_size * time_steps, 1, h, w)
    ssim = ssim_metric(pred_flat, target_flat).item()

    # 多阈值 CSI
    csi_results = {}
    for thresh in thresholds:
        pred_bin = (predictions > thresh).float()
        target_bin = (targets > thresh).float()
        csi_val = csi_metrics[thresh](
            pred_bin.view(-1).long(),
            target_bin.view(-1).long()
        ).item()
        csi_results[f"csi_{thresh}"] = csi_val

    metrics = {"mse": mse, "mae": mae, "ssim": ssim}
    metrics.update(csi_results)
    return metrics

def reset_metrics():
    """重置所有指标（每轮验证后调用）"""
    ssim_metric.reset()
    for m in csi_metrics.values():
        m.reset()