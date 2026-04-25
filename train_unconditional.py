#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train unconditional DiffusionModelUNet on carotid ultrasound images.
"""

import os
import time
import torch
import torch.nn.functional as F
import pandas as pd
from monai import transforms
from monai.data import DataLoader
from monai.utils import set_determinism
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler

from dataset import CarotidDataset2D   # 共用数据集类（见下文）

def main():
    # ---------- 配置 ----------
    set_determinism(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据路径（请修改为你的excel路径）
    excel_path = "D:/project/2509-CarotidGeneration/250922-case_datasets.xlsx"

    # 数据预处理
    train_transforms = transforms.Compose([
        transforms.LoadImaged(keys=["image", "mask"]),
        transforms.EnsureChannelFirstd(keys=["image", "mask"]),
        transforms.EnsureTyped(keys=["image", "mask"]),
        transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=100, b_min=0, b_max=1),
        transforms.ScaleIntensityRanged(keys="mask", a_min=0, a_max=255, b_min=0, b_max=1, clip=True),
        transforms.Resized(keys=["image", "mask"], spatial_size=(128, 128), mode=["bilinear", "nearest"])
    ])

    # 加载数据
    train_df = pd.read_excel(excel_path, sheet_name="train")
    val_df = pd.read_excel(excel_path, sheet_name="val")
    train_ds = CarotidDataset2D(train_df, img_transform=train_transforms)
    val_ds = CarotidDataset2D(val_df, img_transform=train_transforms)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4, drop_last=True)

    # 模型及优化器
    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_channels=(128, 256, 256),
        attention_levels=(False, True, True),
        num_res_blocks=1,
        num_head_channels=256,
    ).to(device)

    scheduler = DDPMScheduler(num_train_timesteps=1000)
    inferer = DiffusionInferer(scheduler)
    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-5)
    scaler = GradScaler()

    # 训练循环
    n_epochs = 150
    val_interval = 10
    epoch_loss_list = []
    val_epoch_loss_list = []

    total_start = time.time()
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in progress_bar:
            images = batch[0]["image"].to(device)
            noise = torch.randn_like(images).to(device)
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device).long()

            with autocast(enabled=True):
                noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)
                loss = F.mse_loss(noise_pred.float(), noise.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})

        avg_loss = epoch_loss / (step + 1)
        epoch_loss_list.append(avg_loss)
        print(f"Epoch {epoch} finished, avg loss: {avg_loss:.6f}")

        # 验证
        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    images = batch[0]["image"].to(device)
                    noise = torch.randn_like(images).to(device)
                    timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device).long()
                    noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)
                    val_loss += F.mse_loss(noise_pred.float(), noise.float()).item()
            val_avg = val_loss / len(val_loader)
            val_epoch_loss_list.append(val_avg)
            print(f"Validation loss: {val_avg:.6f}")

            # 保存检查点
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'train_losses': epoch_loss_list,
                'val_losses': val_epoch_loss_list,
                'timestamp': time.time()
            }
            torch.save(checkpoint, f"unconditional_checkpoint_epoch_{epoch}.pth")
            print(f"Checkpoint saved: unconditional_checkpoint_epoch_{epoch}.pth")

    total_time = time.time() - total_start
    print(f"Training completed. Total time: {total_time:.2f}s")

if __name__ == "__main__":
    main()
