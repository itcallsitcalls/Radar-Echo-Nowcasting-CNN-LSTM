import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as transforms
from config import *

class RadarDataset(Dataset):
    """雷达回波数据集（图像值域归一化到 [0,1]）"""
    def __init__(self, csv_file, root_dir, input_len=INPUT_LEN, target_len=TARGET_LEN,
                 transform=None, train_mode=True):
        self.df = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir
        self.input_len = input_len
        self.target_len = target_len
        self.train_mode = train_mode

        # 定义图像变换
        base_transform = [transforms.Resize((HEIGHT, WIDTH)), transforms.ToTensor()]
        if train_mode:
            base_transform.append(transforms.RandomHorizontalFlip(p=0.5))
            #base_transform.append(transforms.RandomRotation(degrees=5))
        self.transform = transforms.Compose(base_transform)

        # 预处理：验证所有文件是否存在（并添加必要的前缀）
        self.valid_indices = []
        print(f"开始验证 {len(self.df)} 个数据样本...")
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            valid = True
            for j in range(input_len + target_len):
                # 从 CSV 读取原始文件名（如 "00001.png"）
                raw_filename = row.iloc[j]
                # 添加前缀 "radar_" 以匹配实际文件名
                actual_filename = f"radar_{raw_filename}"
                full_path = os.path.join(self.root_dir, actual_filename)
                if not os.path.exists(full_path):
                    # 打印第一个无效路径作为调试信息
                    if len(self.valid_indices) == 0 and j == 0:
                        print(f"示例无效路径：{full_path}")
                    valid = False
                    break
            if valid:
                self.valid_indices.append(i)

        print(f"数据集验证完成。有效样本数: {len(self.valid_indices)}/{len(self.df)}")
        if len(self.valid_indices) == 0:
            raise ValueError("没有找到任何有效的数据样本，请检查 CSV 和图像路径，或调整文件名前缀。")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        row = self.df.iloc[real_idx]
        sequence = []
        for j in range(self.input_len + self.target_len):
            raw_filename = row.iloc[j]
            actual_filename = f"radar_{raw_filename}"
            img_path = os.path.join(self.root_dir, actual_filename)
            try:
                image = Image.open(img_path).convert('L')
                image = self.transform(image).squeeze(0)
            except Exception as e:
                print(f"警告：无法加载图像 {img_path}，错误：{e}，将跳过此样本。")
                return self.__getitem__((idx + 1) % len(self))
            sequence.append(image)
        sequence_tensor = torch.stack(sequence, dim=0)
        x = sequence_tensor[:self.input_len]
        y = sequence_tensor[self.input_len:]
        return x, y

def load_and_preprocess_data():
    """加载数据并返回 DataLoader（训练集和验证集）"""
    # 首先获取所有有效索引
    full_dataset_for_indices = RadarDataset(csv_file=CSV_PATH, root_dir=ROOT_DIR, train_mode=False)
    valid_indices = full_dataset_for_indices.valid_indices

    # 划分索引
    val_size = int(VAL_SPLIT * len(valid_indices))
    train_indices = valid_indices[val_size:]
    val_indices = valid_indices[:val_size]

    # 创建两个 Dataset，分别传入不同的索引列表
    train_dataset = RadarDataset(csv_file=CSV_PATH, root_dir=ROOT_DIR, train_mode=True)
    val_dataset = RadarDataset(csv_file=CSV_PATH, root_dir=ROOT_DIR, train_mode=False)

    # 替换 valid_indices 为划分后的子集
    train_dataset.valid_indices = train_indices
    val_dataset.valid_indices = val_indices

    train_loader = DataLoader(
        train_dataset,
        batch_size=MICRO_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=MICRO_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    return train_loader, val_loader