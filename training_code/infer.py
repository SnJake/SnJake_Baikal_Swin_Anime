import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from model import build_model
from utils import load_config


def _pil_to_tensor(img):
    arr = np.array(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _tensor_to_pil(tensor):
    tensor = tensor.clamp(0, 1)
    arr = (tensor.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def _load_image(path):
    img = Image.open(path)
    alpha = None
    if img.mode == "P":
        img = img.convert("RGBA")
    if img.mode in ("RGBA", "LA"):
        alpha = img.getchannel("A")
    img = img.convert("RGB")
    return img, alpha


def _save_image(img, alpha, path, scale):
    if alpha is not None:
        alpha = alpha.resize((img.width, img.height), resample=Image.Resampling.BICUBIC)
        img = img.convert("RGBA")
        img.putalpha(alpha)
    img.save(path)


def _resolve_inputs(input_path, exts):
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    items = []
    for ext in exts:
        items.extend(input_path.rglob(f"*.{ext}"))
    return sorted(items)


@torch.inference_mode()
def _infer_full(model, lr, amp_ctx):
    with amp_ctx:
        return model(lr)


@torch.inference_mode()
def _infer_tiled(model, lr, scale, tile, overlap, amp_ctx):
    b, c, h, w = lr.shape
    out_h = h * scale
    out_w = w * scale
    output = torch.zeros((b, c, out_h, out_w), device=lr.device, dtype=torch.float32)
    weight = torch.zeros((b, 1, out_h, out_w), device=lr.device, dtype=torch.float32)

    tile = min(tile, h, w)
    overlap = max(0, min(overlap, tile // 2))

    for y in range(0, h, tile):
        for x in range(0, w, tile):
            x0 = max(x - overlap, 0)
            y0 = max(y - overlap, 0)
            x1 = min(x + tile + overlap, w)
            y1 = min(y + tile + overlap, h)

            lr_tile = lr[:, :, y0:y1, x0:x1]
            with amp_ctx:
                sr_tile = model(lr_tile).float()

            out_x0 = x * scale
            out_y0 = y * scale
            out_x1 = min(x + tile, w) * scale
            out_y1 = min(y + tile, h) * scale

            tile_x0 = (x - x0) * scale
            tile_y0 = (y - y0) * scale
            tile_x1 = tile_x0 + (out_x1 - out_x0)
            tile_y1 = tile_y0 + (out_y1 - out_y0)

            output[:, :, out_y0:out_y1, out_x0:out_x1] += sr_tile[:, :, tile_y0:tile_y1, tile_x0:tile_x1]
            weight[:, :, out_y0:out_y1, out_x0:out_x1] += 1.0

    return output / weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--weights", type=str, default="")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--tile", type=int, default=0)
    parser.add_argument("--overlap", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", type=str, default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    scale = int(data_cfg.get("scale", 2))
    exts = data_cfg.get("extensions", ["png", "jpg", "jpeg", "webp"])

    model = build_model(cfg.get("model", {}), scale=scale)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    weights_path = args.weights
    if not weights_path:
        out_dir = Path(cfg.get("train", {}).get("output_dir", "runs/anime_sr_x2"))
        weights_path = out_dir / "weights_last.pt"
    state = torch.load(weights_path, map_location="cpu")
    state_dict = state["model"] if "model" in state else state
    model.load_state_dict(state_dict, strict=True)

    amp_mode = args.amp or cfg.get("train", {}).get("amp", "bf16")
    amp_mode = str(amp_mode).lower()
    use_amp = device.type == "cuda" and amp_mode in ("fp16", "bf16")
    amp_dtype = torch.float16 if amp_mode == "fp16" else torch.bfloat16
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()

    inputs = _resolve_inputs(args.input, exts)
    if not inputs:
        raise ValueError("No input images found.")

    output_path = Path(args.output)
    if len(inputs) > 1 or output_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
        out_dir = output_path
    else:
        out_dir = output_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

    for path in tqdm(inputs, desc="Infer", dynamic_ncols=True):
        img, alpha = _load_image(path)
        lr = _pil_to_tensor(img).unsqueeze(0).to(device)
        if args.tile and args.tile > 0:
            sr = _infer_tiled(model, lr, scale, args.tile, args.overlap, amp_ctx)
        else:
            sr = _infer_full(model, lr, amp_ctx)
        sr_img = _tensor_to_pil(sr.squeeze(0))

        if len(inputs) > 1 or output_path.is_dir():
            out_file = out_dir / f"{path.stem}_x{scale}{path.suffix}"
        else:
            out_file = output_path
        _save_image(sr_img, alpha, out_file, scale)


if __name__ == "__main__":
    main()
