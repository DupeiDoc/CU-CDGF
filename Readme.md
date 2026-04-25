# CU‑CDGF: Carotid Ultrasound Conditional Diffusion Generation Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-CC_BY--NC--ND_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

## 1. Overview

CU‑CDGF is a conditional diffusion framework (ControlNet + DDPM) that generates realistic carotid artery ultrasound images from segmentation masks. It alleviates data scarcity and improves downstream segmentation, especially for the intima‑media complex (IMC).

This repository provides:
- Training code for unconditional diffusion model (DiffusionModelUNet)
- Training code for mask‑conditioned ControlNet
- Script to generate synthetic ultrasound images from any mask
- Dataset loader compatible with MONAI transforms

## 2. Quick Start

### Clone and install dependencies
```bash
git clone https://github.com/your_username/CU-CDGF.git
cd CU-CDGF
pip install -r requirements.txt
```

### Dataset preparation
Prepare an Excel file (`.xlsx`) with at least two sheets: `train` and `val`. Each sheet must contain a column `img` with full paths to original ultrasound images (PNG format). The corresponding binary masks must be stored at `<image_path_without_ext>-ALLMASK.png`.  
Example:  
`/data/patient1.png` → mask at `/data/patient1-ALLMASK.png`

### Train unconditional diffusion model
```bash
python scripts/train_unconditional.py
```
The script will save checkpoints as `unconditional_checkpoint_epoch_*.pth`.

### Train ControlNet (requires pretrained unconditional model)
```bash
# Modify the checkpoint path inside train_controlnet.py if needed
python scripts/train_controlnet.py
```
Checkpoints are saved as `controlnet_checkpoint_epoch_*.pth`.

### Generate synthetic images from a mask
**Option 1: Using the provided generation script**  
```bash
python scripts/generate.py --mask path/to/mask.png --output output.png
```
(You may need to adapt the script to accept command‑line arguments.)

**Option 2: Python code**  
```python
import torch
from scripts.generate import load_trained_models, generate_from_mask
from PIL import Image
import numpy as np

device = torch.device("cuda")
model, controlnet, scheduler = load_trained_models("controlnet_checkpoint_epoch_149.pth", device)

mask = Image.open("mask.png").convert("L").resize((128, 128))
mask_tensor = torch.from_numpy(np.array(mask, dtype=np.float32) / 255.0)[None, None, ...]

synthetic = generate_from_mask(mask_tensor, model, controlnet, scheduler, device)
Image.fromarray((synthetic[0,0].cpu().numpy() * 255).astype(np.uint8)).save("synthetic.png")
```

## 3. Key Results

| Method | FID↓ | LPIPS↓ |
|--------|------|--------|
| Pix2PixGAN | 106.78 | 0.49 |
| CycleGAN | 202.15 | 0.50 |
| **CU‑CDGF (Ours)** | **87.41** | **0.47** |

**IMC segmentation (internal test set):**

| Training data | Dice↑ | 95% HD (mm)↓ |
|---------------|-------|----------------|
| Real only | 0.60 | 7.67 |
| Real + Synthetic | **0.64** | **4.31** |

External validation confirms consistent improvements.

## 4. License

CC BY‑NC‑ND 4.0 – non‑commercial use only, no modifications.

## 5. Citation

```bibtex
@article{du2026synthetic,
  title={A Synthetic Data-Augmented Deep Learning Framework for Robust Segmentation and Quantification of the Carotid Artery in Ultrasound Images},
  author={Du, Pei and Shen, Fengqin and Lai, Zeyu and He, Hongfeng and Shi, Haichao and Li, Fengzhi},
  year={2026}
}
```

## 6. Contact

Fengzhi Li – lifengzhi2018@163.com
```

---

你可以直接将上述内容复制到 `README.md` 文件中。如果后续添加了下游分割（MTANet）的训练脚本，只需在 **Quick Start** 中补充对应的命令即可。
