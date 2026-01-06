![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)
![Made for ComfyUI](https://img.shields.io/badge/Made%20for-ComfyUI-blueviolet)

SnJake Baikal-Swin-Anime x2 is a custom ComfyUI node for upscaling anime/illustration images with a dedicated restoration model. Model in **experimental** state; V2 is slightly sharper and removes edge noise artifacts.

---

# Examples

<img width="4096" height="4096" alt="Diff 1" src="https://github.com/user-attachments/assets/7e4b7305-c5f3-47a9-8945-9bd9d7db4ced" />
<img width="4096" height="2048" alt="Diff 4" src="https://github.com/user-attachments/assets/c7d0427d-4a65-444e-a317-289c8741c3c8" />
<img width="4096" height="2048" alt="Diff 3" src="https://github.com/user-attachments/assets/6204b5eb-c6fc-4b95-bb80-0c611f9ae5e9" />

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
   git clone https://github.com/SnJake/SnJake_Baikal_Swin_Anime.git
   ```
4. For standard ComfyUI installations (with venv):
    1. Make sure your ComfyUI virtual environment (`venv`) is activated.
    2. Navigate into the new node directory and install the requirements:
       ```bash
       cd SnJake_Baikal_Swin_Anime
       pip install -r requirements.txt
       ```
   For Portable ComfyUI installations:
    1. Navigate back to the **root** of your portable ComfyUI directory (e.g., `D:\ComfyUI_windows_portable`).
    2. Run the following command to use the embedded Python to install the requirements. *Do not activate any venv.*
       ```bash
       python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\SnJake_Baikal_Swin_Anime\requirements.txt
       ```

## Step 2: Model Weights

On first use the node can automatically download weights from the repository.

- Default weights location: `ComfyUI/models/anime_upscale/`

If you want to download manually:
1. Download the weights from [HF REPO](https://huggingface.co/SnJake/Baikal-Swin-Anime).
2. Place the file(s) into `ComfyUI/models/anime_upscale/`.

## Step 3: Restart

Restart ComfyUI completely. The node will appear under **`😎 SnJake/Upscale`**.

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
V1:
- Dataset: 40,000 images from Danbooru2024: https://huggingface.co/datasets/deepghs/danbooru2024
- Validation: 600 images
- Epochs: 70

V2: 
- Slightly sharper output, no edge noise artifacts.
- Epochs: 20 (For now)
- Dataset: 49,606 images from Danbooru2024: https://huggingface.co/datasets/deepghs/danbooru2024

Training code is included in `training_code/` for reference.

---

# Disclaimer

This project was made purely for curiosity and personal interest. The code was written by GPT-5.2 Codex.

---

# License

MIT. See `LICENSE.md`.

