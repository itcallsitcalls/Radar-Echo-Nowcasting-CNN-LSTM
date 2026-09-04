import matplotlib.pyplot as plt
import numpy as np
import os
from config import SAVE_PATH, VIS_NUM_SAMPLES

def plot_predictions_and_save(inputs, targets, predictions, save_path=SAVE_PATH, num_samples=VIS_NUM_SAMPLES):
    """可视化输入、目标与预测的多帧对比"""
    # 自动创建保存目录
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    batch_size = inputs.shape[0]
    num_samples = min(num_samples, batch_size)
    total_frames = targets.shape[1]
    if total_frames >= 3:
        frame_indices = [0, total_frames // 2, total_frames - 1]
    else:
        frame_indices = list(range(total_frames))

    fig, axes = plt.subplots(num_samples, len(frame_indices) + 1, figsize=(5*(len(frame_indices)+1), 5*num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    for i in range(num_samples):
        # 第一列：最后一帧输入
        input_img = inputs[i, -1]
        axes[i, 0].imshow(input_img, cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title(f'Sample {i+1} - Last Input')
        axes[i, 0].axis('off')

        # 后续列：目标与预测叠加
        for col, t_idx in enumerate(frame_indices):
            ax = axes[i, col+1]
            target_img = targets[i, t_idx]
            pred_img = predictions[i, t_idx]
            rgb = np.zeros((target_img.shape[0], target_img.shape[1], 3), dtype=np.float32)
            rgb[..., 0] = target_img   # 红色 = 目标
            rgb[..., 1] = pred_img     # 绿色 = 预测
            ax.imshow(rgb, vmin=0, vmax=1)
            ax.set_title(f'Target(R) vs Pred(G) - Frame {t_idx+1}')
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"可视化已保存至 {save_path}")
    plt.close(fig)