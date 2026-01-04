![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)
![Made for ComfyUI](https://img.shields.io/badge/Made%20for-ComfyUI-blueviolet)

SnJake Baikal-Swin-Anime x2 is a custom ComfyUI node for upscaling anime/illustration images with a dedicated restoration model.

---

# Examples

<PLACEHOLDER_IMAGE_1>
<PLACEHOLDER_IMAGE_2>
<PLACEHOLDER_IMAGE_3>

---

# Installation

The installation consists of two steps: installing the node and making the weights available.

## Step 1: Install the Node

1. Open a terminal or command prompt.
2. Navigate to your ComfyUI `custom_nodes` directory.
   ```bash
   # Example for Windows
   cd D:\ComfyUI\custom_nodes\

   # Example for Linux
   cd ~/ComfyUI/custom_nodes/
   ```
3. Clone this repository:
   ```bash
   git clone <GITHUB_REPO_URL>
   ```

## Step 2: Model Weights

On first use the node can automatically download weights from the repository.

- Default weights location: `ComfyUI/models/anime_upscale/`
- Weights selection is a dropdown only.

If you want to download manually:
1. Download the weights from `<HF_REPO_URL>`.
2. Place the file(s) into `ComfyUI/models/anime_upscale/`.

## Step 3: Restart

Restart ComfyUI completely. The node will appear under **`SnJake/Upscale`**.

---

# Usage

The node menu path is **`😎 SnJake/Upscale`**.

## Inputs

- `weights_name`: Select weights from the dropdown (auto-download if missing).
- `image`: Source image.
- `tile`: Tile size for large images. Recommended 256-512. Set 0 to disable tiling.
- `overlap`: Tile overlap for smooth blending. Recommended 32-64.
- `amp`: Precision (`auto`, `bf16`, `fp16`, `none`).
- `device`: Device (`auto`, `cuda`, `cpu`).

## Outputs

- `image`: Upscaled image.

---

# Training Details

- Dataset: 40,000 images from Danbooru2024: https://huggingface.co/datasets/deepghs/danbooru2024
- Validation: 600 images
- Epochs: 70
- Metrics (without degradation) (val):
  - PSNR: 39.517540791829425
  - SSIM: 0.986409981250763
  - DISTS: 0.013307907978693644

Training code is included in `training_code/` for reference.

---

# Disclaimer

This project was made purely for curiosity and personal interest. The code was written by GPT-5.2 Codex.

---

# License

MIT. See `LICENSE.md`.
