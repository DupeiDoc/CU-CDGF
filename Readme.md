
# CU‑CDGF: Carotid Ultrasound Conditional Diffusion Generation Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-CC_BY--NC--ND_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

## 1. Overview

CU‑CDGF is a conditional diffusion framework (ControlNet + DDPM) that generates realistic carotid artery ultrasound images from segmentation masks. It alleviates data scarcity and improves downstream segmentation, especially for the intima‑media complex (IMC).

## 2. Quick Start

```bash
# Clone and install
git clone https://github.com/your_username/CU-CDGF.git
cd CU-CDGF
pip install -r requirements.txt
```

**Generate from a mask**  
```python
from diffusion.sample import generate_from_mask
import numpy as np
from PIL import Image

mask = np.array(Image.open("mask.png").convert("L"))
synthetic = generate_from_mask(mask, model_path="checkpoints/controlnet_carotid.pth")
Image.fromarray(synthetic).save("output.png")
```

**Train models**  
```bash
# Base diffusion
python scripts/train_diffusion.py --config config/diffusion_base.yaml
# ControlNet
python scripts/train_controlnet.py --config config/controlnet.yaml
# Segmentation (MTANet)
python scripts/train_segmentation.py --real_data /path/to/real --synthetic_data /path/to/synthetic
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

请直接将上述内容复制到一个新文件中，命名为 `README.md` 即可。如果需要修改任何占位符（如 `your_username`、论文发表信息等），请自行调整。
