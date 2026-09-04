import torch
import torch.nn as nn
from config import *

class RadarNet(nn.Module):
    """CNN-LSTM 雷达回波外推模型（输入输出值域均为 [0,1]）"""
    def __init__(self, input_len=INPUT_LEN, target_len=TARGET_LEN, height=HEIGHT, width=WIDTH):
        super(RadarNet, self).__init__()
        self.input_len = input_len
        self.target_len = target_len
        self.reduced_height = height // 8
        self.reduced_width = width // 8

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.cnn_flattened_size = 256 * self.reduced_height * self.reduced_width
        self.compress_fc = nn.Linear(self.cnn_flattened_size, HIDDEN_DIM)
        self.lstm = nn.LSTM(HIDDEN_DIM, HIDDEN_DIM, batch_first=True)
        self.decompress_fc = nn.Linear(HIDDEN_DIM, self.cnn_flattened_size)

        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, seq_len, h, w = x.size()
        x_reshaped = x.unsqueeze(2).view(-1, 1, h, w)
        encoded = self.encoder_cnn(x_reshaped)
        encoded_flat = encoded.view(batch_size, seq_len, -1)
        compressed = self.compress_fc(encoded_flat)
        lstm_out, _ = self.lstm(compressed)

        if self.target_len > seq_len:
            last_frame = lstm_out[:, -1:, :]
            repeat_times = self.target_len - seq_len
            lstm_out_extended = torch.cat([lstm_out, last_frame.repeat(1, repeat_times, 1)], dim=1)
        else:
            lstm_out_extended = lstm_out[:, :self.target_len, :]

        decompressed = self.decompress_fc(lstm_out_extended)
        decompressed_reshaped = decompressed.view(-1, 256, self.reduced_height, self.reduced_width)
        decoded = self.decoder_cnn(decompressed_reshaped)
        decoded = decoded.view(batch_size, self.target_len, h, w)
        return decoded