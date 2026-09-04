import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
from config import *
from .evaluator import calculate_metrics, reset_metrics
import copy
import glob
from torchmetrics.functional import structural_similarity_index_measure  # 使用 functional 版本

def weighted_mse(pred, target, weight_scale=5):
    weight = 1 + weight_scale * target
    return torch.mean(weight * (pred - target) ** 2)

def combined_loss(pred, target, mse_criterion, ssim_weight=SSIM_LOSS_WEIGHT):
    mse_loss = weighted_mse(pred, target, weight_scale=5)
    if not USE_SSIM_LOSS or ssim_weight == 0:
        return mse_loss

    batch_size, time_steps, h, w = pred.shape
    pred_flat = pred.view(batch_size * time_steps, 1, h, w)
    target_flat = target.view(batch_size * time_steps, 1, h, w)
    ssim_val = structural_similarity_index_measure(pred_flat, target_flat, data_range=1.0)
    ssim_loss = 1 - ssim_val
    return mse_loss + ssim_weight * ssim_loss

def save_checkpoint(state, filename):
    torch.save(state, filename)
    print(f"检查点已保存至 {filename}")

def load_latest_checkpoint(model, optimizer, scaler=None):
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        return 0, float('inf')
    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_epoch_*.pth"))
    if not checkpoints:
        return 0, float('inf')
    def extract_epoch(fname):
        return int(os.path.splitext(os.path.basename(fname))[0].split('_')[-1])
    latest = max(checkpoints, key=extract_epoch)
    print(f"加载检查点: {latest}")
    checkpoint = torch.load(latest, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    print(f"恢复训练：起始 epoch {start_epoch}，历史最佳验证损失 {best_val_loss:.4f}")
    return start_epoch, best_val_loss

def train_model(model, train_loader, val_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    mse_criterion = nn.MSELoss()
    model.to(DEVICE)

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        else:
            progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
            return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = AMP_GRAD_SCALER

    start_epoch = 0
    best_val_loss = float('inf')
    if RESUME_FROM_CHECKPOINT:
        start_epoch, best_val_loss = load_latest_checkpoint(model, optimizer, scaler)
        # 注意：恢复后需要将调度器步进到当前 epoch，但 LambdaLR 无状态，只需设置 last_epoch
        # 这里采用循环步进的方式（更安全）
        for _ in range(start_epoch):
            scheduler.step()

    patience_counter = 0
    best_model_state = None
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [LR: {scheduler.get_last_lr()[0]:.6f}]")
        for step, (batch_x, batch_y) in enumerate(pbar):
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                outputs = model(batch_x)
                loss = combined_loss(outputs, batch_y, mse_criterion, SSIM_LOSS_WEIGHT)
                loss = loss / GRADIENT_ACCUMULATION_STEPS

            if USE_AMP:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                if USE_AMP:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                    optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            pbar.set_postfix({"batch_loss": loss.item() * GRADIENT_ACCUMULATION_STEPS})

        # 先更新优化器，再步进学习率（消除警告）
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # 验证
        val_metrics = evaluate_model(model, val_loader, mse_criterion)
        current_val_loss = val_metrics['loss']

        # 打印验证指标（包含多阈值 CSI）
        log_msg = f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {current_val_loss:.4f}, SSIM: {val_metrics["ssim"]:.4f}'
        for thresh in CSI_THRESHOLDS:
            log_msg += f', CSI@{thresh}: {val_metrics[f"csi_{thresh}"]:.4f}'
        print(log_msg)

        # 早停检查（基于验证损失）
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"早停触发！验证损失连续 {PATIENCE} 轮未下降。")
                break

        # 定期保存检查点
        if (epoch + 1) % CHECKPOINT_FREQ == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }
            if USE_AMP and scaler is not None:
                checkpoint['scaler_state_dict'] = scaler.state_dict()
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth")
            save_checkpoint(checkpoint, checkpoint_path)

        torch.cuda.empty_cache()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("已加载验证集上最佳模型。")
        if SAVE_BEST:
            torch.save(best_model_state, BEST_MODEL_PATH)
            print(f"最佳模型已保存至 {BEST_MODEL_PATH}")
    return model

@torch.no_grad()
def evaluate_model(model, val_loader, mse_criterion):
    """评估模型，返回平均损失及各指标"""
    model.eval()
    total_loss = 0.0
    metrics_sum = {"mse": 0.0, "mae": 0.0, "ssim": 0.0}
    for thresh in CSI_THRESHOLDS:
        metrics_sum[f"csi_{thresh}"] = 0.0
    sample_count = 0

    reset_metrics()  # 重置累积指标

    for batch_x, batch_y in val_loader:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            outputs = model(batch_x)
            loss = mse_criterion(outputs, batch_y)

        total_loss += loss.item() * batch_x.size(0)
        batch_metrics = calculate_metrics(outputs, batch_y, thresholds=CSI_THRESHOLDS)
        for k in metrics_sum:
            metrics_sum[k] += batch_metrics[k] * batch_x.size(0)
        sample_count += batch_x.size(0)

    avg_metrics = {k: v / sample_count for k, v in metrics_sum.items()}
    avg_metrics['loss'] = total_loss / sample_count
    return avg_metrics