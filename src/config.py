import torch
import os

# ==================== 数据路径 ====================
CSV_PATH = r"C:\Users\Evergarden\PycharmProjects\radar_project\Train.csv"
ROOT_DIR = r"C:\Users\Evergarden\PycharmProjects\radar_project\radar"

# ==================== 数据参数 ====================
INPUT_LEN = 10
TARGET_LEN = 10
HEIGHT = 256
WIDTH = 256
VAL_SPLIT = 0.2

# ==================== 模型参数 ====================
HIDDEN_DIM = 512

# ==================== 训练参数 ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MICRO_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
WARMUP_EPOCHS = 5
GRAD_CLIP_NORM = 1.0

# ==================== 损失函数配置 ====================
USE_SSIM_LOSS = True               # 是否在损失中加入 SSIM 项
SSIM_LOSS_WEIGHT = 0.1             # SSIM 损失的权重（λ），loss = MSE + λ * (1 - SSIM)

# ==================== 评估阈值 ====================
# 用于 CSI 计算的多个阈值（列表形式），方便观察模型在不同阈值下的表现
CSI_THRESHOLDS = [0.1, 0.15, 0.184, 0.2, 0.25, 0.3]

# ==================== 早停与模型保存 ====================
PATIENCE = 10
SAVE_BEST = True
BEST_MODEL_PATH = "./best_model.pth"

# ==================== 检查点配置 ====================
CHECKPOINT_DIR = "./checkpoints"
CHECKPOINT_FREQ = 5
RESUME_FROM_CHECKPOINT = True

# ==================== 混合精度 ====================
USE_AMP = True
AMP_GRAD_SCALER = torch.cuda.amp.GradScaler() if USE_AMP else None

# ==================== 数据加载 ====================
NUM_WORKERS = 4
PIN_MEMORY = True

# ==================== 可视化 ====================
SAVE_PATH = "./prediction_visualization.png"
VIS_NUM_SAMPLES = 4