#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train ControlNet on top of a pretrained unconditional diffusion model.
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
from generative.networks.nets import DiffusionModelUNet, ControlNet
from generative.networks.schedulers import DDPMScheduler

from dataset import CarotidDataset2D

def main():
    set_determinism(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据路径（修改）
    excel_path = "D:/project/2509-CarotidGeneration/250922-case_datasets.xlsx"

    train_transforms = transforms.Compose([
        transforms.LoadImaged(keys=["image", "mask"]),
        transforms.EnsureChannelFirstd(keys=["image", "mask"]),
        transforms.EnsureTyped(keys=["image", "mask"]),
        transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=100, b_min=0, b_max=1),
        transforms.ScaleIntensityRanged(keys="mask", a_min=0, a_max=255, b_min=0, b_max=1, clip=True),
        transforms.Resized(keys=["image", "mask"], spatial_size=(128, 128), mode=["bilinear", "nearest"])
    ])

    train_df = pd.read_excel(excel_path, sheet_name="train")
    val_df = pd.read_excel(excel_path, sheet_name="val")
    train_ds = CarotidDataset2D(train_df, img_transform=train_transforms)
    val_ds = CarotidDataset2D(val_df, img_transform=train_transforms)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4, drop_last=True)

    # 加载预训练的无条件模型
    unconditional_ckpt = "unconditional_checkpoint_epoch_149.pth"  # 请修改为实际路径
    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_channels=(128, 256, 256),
        attention_levels=(False, True, True),
        num_res_blocks=1,
        num_head_channels=256,
    ).to(device)

    # 加载权重（仅模型）
    checkpoint = torch.load(unconditional_ckpt, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded unconditional model from {unconditional_ckpt} (epoch {checkpoint['epoch']})")

    # 创建 ControlNet
    controlnet = ControlNet(
        spatial_dims=2,
        in_channels=1,
        num_channels=(128, 256, 256),
        attention_levels=(False, True, True),
        num_res_blocks=1,
        num_head_channels=256,
        conditioning_embedding_num_channels=(16,),
    ).to(device)

    # 复制 diffusion model 的权重到 controlnet（非严格）
    controlnet.load_state_dict(model.state_dict(), strict=False)

    # 冻结 diffusion model
    for p in model.parameters():
        p.requires_grad = False

    scheduler = DDPMScheduler(num_train_timesteps=1000)
    inferer = DiffusionInferer(scheduler)
    optimizer = torch.optim.Adam(controlnet.parameters(), lr=2.5e-5)
    scaler = GradScaler()

    n_epochs = 150
    val_interval = 10
    epoch_loss_list = []
    val_epoch_loss_list = []

    total_start = time.time()
    for epoch in range(n_epochs):
        model.train()  # 实际只有 controlnet 在训练，但 model 处于 eval 状态可避免 BN 统计量更新
        controlnet.train()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in progress_bar:
            images = batch[0]["image"].to(device)
            masks = batch[0]["mask"].to(device)

            noise = torch.randn_like(images).to(device)
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device).long()

            # 加噪
            images_noised = scheduler.add_noise(images, noise=noise, timesteps=timesteps)

            with autocast(enabled=True):
                # ControlNet forward
                down_res, mid_res = controlnet(x=images_noised, timesteps=timesteps, controlnet_cond=masks)
                noise_pred = model(
                    x=images_noised,
                    timesteps=timesteps,
                    down_block_additional_residuals=down_res,
                    mid_block_additional_residual=mid_res,
                )
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

        if (epoch + 1) % val_interval == 0:
            controlnet.eval()
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    images = batch[0]["image"].to(device)
                    masks = batch[0]["mask"].to(device)
                    noise = torch.randn_like(images).to(device)
                    timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device).long()
                    images_noised = scheduler.add_noise(images, noise=noise, timesteps=timesteps)
                    down_res, mid_res = controlnet(images_noised, timesteps, masks)
                    noise_pred = model(images_noised, timesteps, down_block_additional_residuals=down_res,
                                       mid_block_additional_residual=mid_res)
                    val_loss += F.mse_loss(noise_pred.float(), noise.float()).item()
            val_avg = val_loss / len(val_loader)
            val_epoch_loss_list.append(val_avg)
            print(f"Validation loss: {val_avg:.6f}")

            # 保存检查点
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'controlnet_state_dict': controlnet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'train_losses': epoch_loss_list,
                'val_losses': val_epoch_loss_list,
                'timestamp': time.time()
            }
            torch.save(checkpoint, f"controlnet_checkpoint_epoch_{epoch}.pth")
            print(f"Checkpoint saved: controlnet_checkpoint_epoch_{epoch}.pth")

    total_time = time.time() - total_start
    print(f"Training completed. Total time: {total_time:.2f}s")

if __name__ == "__main__":
    main()
