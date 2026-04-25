#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate synthetic carotid ultrasound images from segmentation masks using a trained ControlNet.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from monai import transforms
from monai.data import DataLoader
from monai.utils import set_determinism
from torch.cuda.amp import autocast
from tqdm import tqdm

from generative.networks.nets import DiffusionModelUNet, ControlNet
from generative.networks.schedulers import DDPMScheduler
from dataset import CarotidDataset2D   # 如果只需要加载真实掩膜示例
import pandas as pd

def load_trained_models(controlnet_ckpt_path, device='cuda'):
    """加载训练好的模型和调度器"""
    # 创建模型结构（与训练时一致）
    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_channels=(128, 256, 256),
        attention_levels=(False, True, True),
        num_res_blocks=1,
        num_head_channels=256,
    ).to(device)

    controlnet = ControlNet(
        spatial_dims=2,
        in_channels=1,
        num_channels=(128, 256, 256),
        attention_levels=(False, True, True),
        num_res_blocks=1,
        num_head_channels=256,
        conditioning_embedding_num_channels=(16,),
    ).to(device)

    checkpoint = torch.load(controlnet_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    controlnet.load_state_dict(checkpoint['controlnet_state_dict'])

    scheduler = DDPMScheduler(num_train_timesteps=1000)
    # 注意：需要从检查点恢复调度器状态吗？通常不需要，只需使用默认的推理timesteps
    # 如果保存了 scheduler_state_dict，可以加载，但这里保持简单
    scheduler.set_timesteps(num_inference_steps=1000)

    model.eval()
    controlnet.eval()
    return model, controlnet, scheduler

def generate_from_mask(mask_tensor, model, controlnet, scheduler, device, num_inference_steps=1000):
    """
    mask_tensor: shape (1, 1, H, W) 或 (B,1,H,W)，取值范围 [0,1]，二值掩膜
    """
    # 确保 mask 在正确的设备上
    mask_tensor = mask_tensor.to(device)
    batch_size = mask_tensor.shape[0]

    # 随机噪声
    sample = torch.randn((batch_size, 1, 128, 128)).to(device)

    # 逐步去噪
    progress_bar = tqdm(scheduler.timesteps, total=len(scheduler.timesteps), ncols=80, desc="Sampling")
    for t in progress_bar:
        with torch.no_grad():
            with autocast(enabled=True):
                down_res, mid_res = controlnet(
                    x=sample,
                    timesteps=torch.full((batch_size,), t, device=device, dtype=torch.long),
                    controlnet_cond=mask_tensor
                )
                noise_pred = model(
                    sample,
                    timesteps=torch.full((batch_size,), t, device=device, dtype=torch.long),
                    down_block_additional_residuals=down_res,
                    mid_block_additional_residual=mid_res,
                )
                sample, _ = scheduler.step(model_output=noise_pred, timestep=t, sample=sample)
    return sample  # 范围 [0,1]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_determinism(42)

    # 指定检查点路径
    ckpt_path = "controlnet_checkpoint_epoch_149.pth"   # 修改为你的文件
    model, controlnet, scheduler = load_trained_models(ckpt_path, device)

    # ----- 示例1：从验证集中的真实掩膜生成 -----
    # 加载一个验证集掩膜
    excel_path = "D:/project/2509-CarotidGeneration/250922-case_datasets.xlsx"
    val_df = pd.read_excel(excel_path, sheet_name="val")
    # 简单取前几个掩膜
    sample_mask_path = val_df.iloc[0]['img'].replace('.png', '-ALLMASK.png')
    # 读取并预处理掩膜（需要与训练相同的resize）
    import numpy as np
    from PIL import Image
    mask = Image.open(sample_mask_path).convert('L')
    mask = mask.resize((128, 128), Image.NEAREST)
    mask_tensor = torch.from_numpy(np.array(mask, dtype=np.float32) / 255.0)
    mask_tensor = mask_tensor[None, None, ...]  # (1,1,128,128)

    # 生成
    synthetic = generate_from_mask(mask_tensor, model, controlnet, scheduler, device)

    # 显示
    fig, axes = plt.subplots(1, 2, figsize=(4,2))
    axes[0].imshow(mask_tensor[0,0].cpu(), cmap='gray')
    axes[0].set_title('Mask')
    axes[0].axis('off')
    axes[1].imshow(synthetic[0,0].cpu(), cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Synthetic')
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

    # ----- 示例2：从自定义掩膜生成（如圆形和正方形）-----
    xx, yy = np.mgrid[:128, :128]
    circle = ((xx - 32) ** 2 + (yy - 32) ** 2) < 30**2
    square = np.zeros((128, 128))
    square[10:50, 10:50] = 1
    manual_mask = np.stack([circle, square], axis=0).astype(np.float32)  # (2,128,128)
    manual_mask_tensor = torch.from_numpy(manual_mask)[:, None, ...]     # (2,1,128,128)

    synthetic_manual = generate_from_mask(manual_mask_tensor, model, controlnet, scheduler, device)

    fig, axes = plt.subplots(2, 2, figsize=(4,4))
    for i in range(2):
        axes[i,0].imshow(manual_mask_tensor[i,0], cmap='gray')
        axes[i,0].axis('off')
        axes[i,0].set_title(f'Mask {i}')
        axes[i,1].imshow(synthetic_manual[i,0].cpu(), cmap='gray', vmin=0, vmax=1)
        axes[i,1].axis('off')
        axes[i,1].set_title('Synthetic')
    plt.tight_layout()
    plt.show()

    # 保存生成的图像（示例）
    # Image.fromarray((synthetic[0,0].cpu().numpy()*255).astype(np.uint8)).save("synthetic_example.png")

if __name__ == "__main__":
    main()
