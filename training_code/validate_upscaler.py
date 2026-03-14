import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import AnimeSRDataset
from metrics import build_dists, calc_dists, calc_psnr, calc_ssim
from model import build_model
from utils import load_config


@torch.inference_mode()
def infer_tiled(model, lr, scale, tile, overlap, amp_ctx, tile_batch_size=1):
    b, c, h, w = lr.shape
    out_h = h * scale
    out_w = w * scale
    output = torch.zeros((b, c, out_h, out_w), device=lr.device, dtype=torch.float32)
    weight = torch.zeros((b, 1, out_h, out_w), device=lr.device, dtype=torch.float32)

    tile = min(tile, h, w)
    overlap = max(0, min(overlap, tile // 2))
    tile_batch_size = max(1, int(tile_batch_size))

    coords = []
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            coords.append((x, y))

    max_in_h = min(h, tile + 2 * overlap)
    max_in_w = min(w, tile + 2 * overlap)

    for idx in range(0, len(coords), tile_batch_size):
        chunk = coords[idx : idx + tile_batch_size]
        lr_tiles = []
        tile_meta = []
        for x, y in chunk:
            x0 = max(x - overlap, 0)
            y0 = max(y - overlap, 0)
            x1 = min(x + tile + overlap, w)
            y1 = min(y + tile + overlap, h)

            lr_tile = lr[:, :, y0:y1, x0:x1]
            pad_h = max_in_h - lr_tile.shape[-2]
            pad_w = max_in_w - lr_tile.shape[-1]
            if pad_h > 0 or pad_w > 0:
                lr_tile = torch.nn.functional.pad(lr_tile, (0, pad_w, 0, pad_h), mode="replicate")

            out_x0 = x * scale
            out_y0 = y * scale
            out_x1 = min(x + tile, w) * scale
            out_y1 = min(y + tile, h) * scale

            tile_x0 = (x - x0) * scale
            tile_y0 = (y - y0) * scale
            tile_x1 = tile_x0 + (out_x1 - out_x0)
            tile_y1 = tile_y0 + (out_y1 - out_y0)

            lr_tiles.append(lr_tile)
            tile_meta.append((out_x0, out_y0, out_x1, out_y1, tile_x0, tile_y0, tile_x1, tile_y1))

        lr_tiles = torch.cat(lr_tiles, dim=0)
        with amp_ctx:
            sr_tiles = model(lr_tiles).float()

        for i, meta in enumerate(tile_meta):
            out_x0, out_y0, out_x1, out_y1, tile_x0, tile_y0, tile_x1, tile_y1 = meta
            sr_tile = sr_tiles[i : i + 1]
            output[:, :, out_y0:out_y1, out_x0:out_x1] += sr_tile[:, :, tile_y0:tile_y1, tile_x0:tile_x1]
            weight[:, :, out_y0:out_y1, out_x0:out_x1] += 1.0

    return output / weight


def val_collate(batch):
    return batch


def _load_weights(model, path):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    return missing, unexpected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--val_dir", type=str, default="")
    parser.add_argument("--max_images", type=int, default=-1)
    parser.add_argument("--tile", type=int, default=-1)
    parser.add_argument("--overlap", type=int, default=-1)
    parser.add_argument("--tile_batch_size", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=-1)
    parser.add_argument("--amp", type=str, default="")
    parser.add_argument("--use_dists", action="store_true")
    parser.add_argument("--no_dists", action="store_true")
    parser.add_argument("--output_json", type=str, default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    val_cfg = cfg.get("val", {}) if isinstance(cfg.get("val", {}), dict) else {}

    val_dir = args.val_dir or val_cfg.get("val_dir")
    if not val_dir:
        raise ValueError("Validation directory is required: pass --val_dir or set val.val_dir in config.")

    scale = int(data_cfg.get("scale", 2))
    model = build_model(cfg.get("model", {}), scale=scale)

    missing, unexpected = _load_weights(model, args.weights)
    if missing:
        print(f"Warning: missing keys while loading weights: {len(missing)}")
    if unexpected:
        print(f"Warning: unexpected keys while loading weights: {len(unexpected)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    amp_mode = (args.amp or str(cfg.get("train", {}).get("amp", "bf16"))).lower()
    use_amp = amp_mode in ("fp16", "bf16") and device.type == "cuda"
    amp_dtype = torch.float16 if amp_mode == "fp16" else torch.bfloat16

    val_dataset = AnimeSRDataset(
        root=val_dir,
        scale=scale,
        hr_patch_size=int(val_cfg.get("hr_patch_size", 0)),
        exts=data_cfg.get("extensions", ["png", "jpg", "jpeg", "webp"]),
        augment=False,
        degrade_cfg=val_cfg.get("degrade", data_cfg.get("degrade", {})),
        training=False,
        crop_to_scale=bool(val_cfg.get("crop_to_scale", True)),
    )

    batch_size = int(args.batch_size if args.batch_size > 0 else val_cfg.get("batch_size", 1))
    num_workers = int(args.num_workers if args.num_workers >= 0 else val_cfg.get("num_workers", 2))
    pin_memory = bool(val_cfg.get("pin_memory", False))
    persistent_workers = bool(val_cfg.get("persistent_workers", False)) and num_workers > 0

    loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers,
        collate_fn=val_collate,
    )

    tile = int(args.tile if args.tile >= 0 else val_cfg.get("tile", 0))
    overlap = int(args.overlap if args.overlap >= 0 else val_cfg.get("overlap", 16))
    tile_batch_size = int(args.tile_batch_size if args.tile_batch_size > 0 else val_cfg.get("tile_batch_size", 1))
    max_images = int(args.max_images if args.max_images >= 0 else val_cfg.get("max_images", 0))

    use_dists = bool(val_cfg.get("use_dists", True))
    if args.use_dists:
        use_dists = True
    if args.no_dists:
        use_dists = False

    dists_device_name = str(val_cfg.get("dists_device", "cuda")).lower()
    dists_device = torch.device("cuda" if dists_device_name == "cuda" and torch.cuda.is_available() else "cpu")
    dists_model = build_dists(dists_device) if use_dists else None
    if use_dists and dists_model is None:
        print("Warning: DISTS is unavailable, evaluating without DISTS.")

    total_count = 0
    sum_psnr = 0.0
    sum_ssim = 0.0
    sum_dists = 0.0

    progress_total = int(max_images) if max_images and max_images > 0 else len(val_dataset)
    progress = tqdm(total=progress_total, desc="Val", dynamic_ncols=True)
    for batch in loader:
        if max_images and total_count >= max_images:
            break

        samples = batch if isinstance(batch, list) else [batch]
        for lr_img, hr_img in samples:
            if max_images and total_count >= max_images:
                break

            if lr_img.ndim == 3:
                lr_img = lr_img.unsqueeze(0)
            if hr_img.ndim == 3:
                hr_img = hr_img.unsqueeze(0)

            lr_img = lr_img.to(device, non_blocking=True)
            hr_img = hr_img.to(device, non_blocking=True)

            amp_ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
            if tile > 0:
                sr = infer_tiled(model, lr_img, scale, tile, overlap, amp_ctx, tile_batch_size=tile_batch_size)
            else:
                with torch.no_grad(), amp_ctx:
                    sr = model(lr_img).float()

            sr = sr.clamp(0, 1)
            hr_img = hr_img.clamp(0, 1)
            if sr.shape[-2:] != hr_img.shape[-2:]:
                min_h = min(sr.shape[-2], hr_img.shape[-2])
                min_w = min(sr.shape[-1], hr_img.shape[-1])
                sr = sr[..., :min_h, :min_w]
                hr_img = hr_img[..., :min_h, :min_w]

            psnr_vals = calc_psnr(sr, hr_img)
            ssim_vals = calc_ssim(sr, hr_img)
            dists_vals = calc_dists(dists_model, sr, hr_img) if dists_model is not None else None

            bsz = sr.shape[0]
            if max_images and (total_count + bsz) > max_images:
                keep = max_images - total_count
                if keep <= 0:
                    break
                psnr_vals = psnr_vals[:keep]
                ssim_vals = ssim_vals[:keep]
                if dists_vals is not None:
                    dists_vals = dists_vals[:keep]
                bsz = int(keep)

            total_count += bsz
            sum_psnr += float(psnr_vals.sum().cpu())
            sum_ssim += float(ssim_vals.sum().cpu())
            if dists_vals is not None:
                sum_dists += float(dists_vals.sum().cpu())
            progress.update(bsz)

            if total_count > 0:
                postfix = {
                    "psnr": f"{sum_psnr / total_count:.4f}",
                    "ssim": f"{sum_ssim / total_count:.5f}",
                }
                if dists_model is not None:
                    postfix["dists"] = f"{sum_dists / total_count:.5f}"
                progress.set_postfix(postfix)
    progress.close()

    if total_count == 0:
        raise RuntimeError("No validation samples were processed.")

    result = {
        "weights": str(args.weights),
        "config": str(args.config),
        "val_dir": str(val_dir),
        "count": int(total_count),
        "psnr": sum_psnr / total_count,
        "ssim": sum_ssim / total_count,
    }
    if dists_model is not None:
        result["dists"] = sum_dists / total_count

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
