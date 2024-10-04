# ttt_model_2d_hrnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from block_2d import Block2D
from ttt_config import TTTConfig
from positional_embedding import Learned2DPositionalEmbedding
from segmentation_head import SegmentationHead

class TTTModel2D(nn.Module):
    def __init__(self, config: TTTConfig):
        super().__init__()
        self.config = config
        self.num_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size

        # 初始化高分辨率分支
        self.branch1 = nn.Sequential(
            nn.Conv2d(config.in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # 初始化中等分辨率分支
        self.branch2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),  # 下采样
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # 初始化低分辨率分支
        self.branch3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),  # 下采样
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # 定义高频特征提取卷积层
        self.freq_conv_real = nn.Conv2d(config.in_channels, 64, kernel_size=3, padding=1)
        self.freq_conv_imag = nn.Conv2d(config.in_channels, 64, kernel_size=3, padding=1)
        self.freq_bn_real = nn.BatchNorm2d(64)
        self.freq_bn_imag = nn.BatchNorm2d(64)

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            256+64+64,       # 合并高频实部和虚部后增加的通道数
            #256+config.in_channels*2,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )

        # 初始化位置嵌入
        self.positional_embedding = Learned2DPositionalEmbedding(
            num_patches=config.num_patches,
            embed_dim=config.hidden_size
        )

        # 初始化 Transformer 块
        self.blocks = nn.ModuleList([
            Block2D(config, layer_idx=i) for i in range(config.num_hidden_layers)
        ])

        self.mergebranch1 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1, stride=1),  # 下采样
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.mergebranch2 = nn.Sequential(
            nn.Conv2d(320, 256, kernel_size=3, padding=1, stride=1),  # 下采样
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        '''''
        self.mergebranch3 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1, stride=1),  # 下采样
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        '''''
        # Final normalization and classifier
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 最终分割头
        self.segmentation_head = SegmentationHead(
            embed_dim=config.hidden_size,
            num_classes=config.num_classes
        )

    def compute_fft(self, x):
        fft_x = torch.fft.fft2(x)
        real = fft_x.real
        imag = fft_x.imag
        return real, imag

    def extract_high_frequency(self, real, imag, threshold=0.05):
        B, C, H, W = real.shape
        # 创建频率坐标
        u = torch.fft.fftfreq(H, d=1.0).to(real.device)
        v = torch.fft.fftfreq(W, d=1.0).to(real.device)
        uu, vv = torch.meshgrid(u, v, indexing='ij')
        # 计算频率幅度
        freq_magnitude = torch.sqrt(uu**2 + vv**2)
        # 创建高频掩码
        high_freq_mask = (freq_magnitude > threshold).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        # 应用掩码
        high_freq_real = real * high_freq_mask
        high_freq_imag = imag * high_freq_mask
        return high_freq_real, high_freq_imag

    def forward(self, images, cache_params=None):
        """
        Args:
            images: [B, C, H, W]
            cache_params: optional cache parameters for fast decoding
        Returns:
            outputs: [B, num_classes, H, W]
        """
        B, C, H, W = images.shape

        # 计算傅里叶变换
        real, imag = self.compute_fft(images)  # [B, C, H, W]
        high_freq_real, high_freq_imag = self.extract_high_frequency(real, imag, threshold=0.02)  # [B, C, H, W] each

        # 通过高分辨率分支
        x1 = self.branch1(images)  # [B, 64, H, W]

        # 通过中等分辨率分支
        x2 = self.branch2(x1)  # [B, 128, H/2, W/2]

        # 通过低分辨率分支
        x3 = self.branch3(x2)  # [B, 256, H/4, W/4]

        # 信息交换与特征融合
        # 上采样低分辨率特征并与中等分辨率特征融合
        x2_up = F.interpolate(x3, size=x2.shape[2:], mode='bilinear', align_corners=False)  # [B, 256, H/2, W/2]
        x2 = torch.cat((x2, x2_up), dim=1)  # [B, 128 + 256, H/2, W/2]
        x2 = self.mergebranch1(x2)  # 更新中等分辨率特征

        # 上采样中等分辨率特征并与高分辨率特征融合
        x1_up = F.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=False)  # [B, 128, H, W]
        x1 = torch.cat((x1, x1_up), dim=1)  # [B, 64 + 128, H, W]
        x1 = self.mergebranch2(x1)  # 更新高分辨率特征

        # 处理高频实部和虚部
        freq_real = F.relu(self.freq_conv_real(high_freq_real))  # [B, 64, H, W]
        freq_imag = F.relu(self.freq_conv_imag(high_freq_imag))  # [B, 64, H, W]
        #freq_imag = F.relu(self.freq_conv_imag(imag))
        freq_real = self.freq_bn_real(freq_real)
        freq_imag = self.freq_bn_imag(freq_imag)
        #fft_x = torch.complex(high_freq_real, high_freq_imag)
        #reconstructed = torch.fft.ifft2(fft_x).real
        #freq_imag = F.relu(self.freq_conv_imag(reconstructed))
        #freq_imag = self.freq_bn_imag(freq_imag)
        # 合并高频特征
        #x = torch.cat((x1, high_freq_real, high_freq_imag), dim=1)
        # Patch embedding
        x = torch.cat((x1, freq_real, freq_imag), dim=1)  # [B, 64 + 64 + 64, H, W]
        #x = torch.cat((x1, freq_imag), dim=1)
        #x = self.mergebranch3(x)
        #x=x1
        x = self.patch_embed(x)  # [B, hidden_size, H_p, W_p]
        H_p, W_p = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # [B, N_patches, hidden_size]
        x = x + self.positional_embedding(x)  # [B, N_patches, hidden_size]

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(
                hidden_states=x,
                attention_mask=None,
                position_ids=None,
                cache_params=cache_params,
            )

        # Final normalization
        x = self.final_norm(x)  # [B, N_patches, hidden_size]

        # Add segmentation head
        logits = self.segmentation_head(x)  # [B, num_classes, H_p, W_p]

        # 上采样到原始图像尺寸
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)  # [B, num_classes, H, W]

        return logits
