import torch
from config import *
from src.data_processor import load_and_preprocess_data
from src.models import RadarNet
from src.model_trainer import train_model, evaluate_model
from src.visualizer import plot_predictions_and_save

def main():
    print(f"使用设备: {DEVICE}")
    print(f"模型配置: 输入帧数={INPUT_LEN}, 预测帧数={TARGET_LEN}, 图像尺寸={HEIGHT}×{WIDTH}")
    print(f"训练配置: 微批次={MICRO_BATCH_SIZE}, 累积步数={GRADIENT_ACCUMULATION_STEPS}, "
          f"有效批次={MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}, 最大轮数={EPOCHS}")
    print(f"数据路径: CSV={CSV_PATH}, 图像根目录={ROOT_DIR}")
    if USE_SSIM_LOSS:
        print(f"损失函数: MSE + {SSIM_LOSS_WEIGHT} * (1-SSIM)")

    train_loader, val_loader = load_and_preprocess_data()
    model = RadarNet()
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    model = train_model(model, train_loader, val_loader)

    print("\n最终评估...")
    criterion = torch.nn.MSELoss()
    final_metrics = evaluate_model(model, val_loader, criterion)
    print("最终验证集指标:")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")

    # 可视化一批结果
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                pred_y = model(batch_x)
            break
    plot_predictions_and_save(
        inputs=batch_x.cpu().numpy(),
        targets=batch_y.cpu().numpy(),
        predictions=pred_y.cpu().numpy()
    )

    print("\n流程完成！")

if __name__ == "__main__":
    main()