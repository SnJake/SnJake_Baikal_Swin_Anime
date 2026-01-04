import argparse
import gc
import math
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import AnimeSRDataset
from losses import CombinedLoss
from metrics import build_dists, calc_dists, calc_psnr, calc_ssim
from model import build_model
from utils import EMA, append_jsonl, ensure_dir, load_config, set_seed


def build_scheduler(optimizer, cfg, total_steps):
    name = str(cfg.get("name", "cosine")).lower()
    warmup_steps = int(cfg.get("warmup_steps", 0))
    min_lr = float(cfg.get("min_lr", 0.0))
    base_lr = optimizer.param_groups[0]["lr"]
    min_lr_ratio = min_lr / base_lr if base_lr > 0 else 0.0

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        if name == "constant":
            return 1.0
        if total_steps <= warmup_steps:
            return 1.0
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def val_collate(batch):
    return batch


@torch.inference_mode()
def infer_tiled(model, lr, scale, tile, overlap, amp_ctx):
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
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("train", {}).get("seed"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = cfg.get("data", {})
    train_dir = data_cfg.get("train_dir")
    if not train_dir:
        raise ValueError("data.train_dir is required")

    dataset = AnimeSRDataset(
        root=train_dir,
        scale=int(data_cfg.get("scale", 2)),
        hr_patch_size=int(data_cfg.get("hr_patch_size", 192)),
        exts=data_cfg.get("extensions", ["png", "jpg", "jpeg", "webp"]),
        augment=bool(data_cfg.get("augment", True)),
        degrade_cfg=data_cfg.get("degrade", {}),
        training=True,
    )
    num_workers = int(data_cfg.get("num_workers", 4))
    batch_size = int(data_cfg.get("batch_size", 16))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

    model = build_model(cfg.get("model", {}), scale=int(data_cfg.get("scale", 2)))
    model = model.to(device)

    loss_fn = CombinedLoss(cfg.get("loss", {})).to(device)

    optim_cfg = cfg.get("optimizer", {})
    lr = float(optim_cfg.get("lr", 2e-4))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=tuple(optim_cfg.get("betas", (0.9, 0.99))),
        eps=float(optim_cfg.get("eps", 1e-8)),
        weight_decay=float(optim_cfg.get("weight_decay", 1e-2)),
    )

    train_cfg = cfg.get("train", {})
    grad_accum = int(train_cfg.get("grad_accum_steps", 1))
    grad_accum = max(1, grad_accum)
    grad_clip = float(train_cfg.get("grad_clip_norm", 0.0))
    epochs = int(train_cfg.get("epochs", 200))
    log_every = int(train_cfg.get("log_every", 50))
    save_every = int(train_cfg.get("save_every_epochs", 1))

    output_dir = Path(train_cfg.get("output_dir", "runs/anime_sr"))
    ensure_dir(output_dir)
    log_path = output_dir / "train_log.jsonl"

    ema_cfg = cfg.get("ema", {})
    ema = None
    if bool(ema_cfg.get("enabled", True)):
        ema = EMA(model, decay=float(ema_cfg.get("decay", 0.999)))

    amp_mode = str(train_cfg.get("amp", "bf16")).lower()
    use_amp = amp_mode in ("fp16", "bf16") and device.type == "cuda"
    amp_dtype = torch.float16 if amp_mode == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler(enabled=(amp_mode == "fp16" and device.type == "cuda"))

    total_steps = max(1, (len(loader) * epochs) // grad_accum)
    scheduler = build_scheduler(optimizer, cfg.get("scheduler", {}), total_steps=total_steps)

    start_epoch = 0
    global_step = 0
    resume_path = train_cfg.get("resume")
    if resume_path:
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt and scaler is not None:
            scaler.load_state_dict(ckpt["scaler"])
        if ema is not None and "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))

    optimizer.zero_grad(set_to_none=True)

    val_cfg = cfg.get("val", {})
    val_dir = val_cfg.get("val_dir") if isinstance(val_cfg, dict) else None
    val_loader = None
    dists_model = None
    use_dists = False
    dists_device = None
    recreate_dists = False
    empty_cache_after = False
    if val_dir:
        val_dataset = AnimeSRDataset(
            root=val_dir,
            scale=int(data_cfg.get("scale", 2)),
            hr_patch_size=int(val_cfg.get("hr_patch_size", 0)),
            exts=data_cfg.get("extensions", ["png", "jpg", "jpeg", "webp"]),
            augment=False,
            degrade_cfg=val_cfg.get("degrade", data_cfg.get("degrade", {})),
            training=False,
            crop_to_scale=bool(val_cfg.get("crop_to_scale", True)),
        )
        val_bs = int(val_cfg.get("batch_size", 1))
        val_workers = int(val_cfg.get("num_workers", 2))
        val_pin_memory = bool(val_cfg.get("pin_memory", False))
        val_persistent = bool(val_cfg.get("persistent_workers", False)) and val_workers > 0
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_bs,
            shuffle=False,
            num_workers=val_workers,
            pin_memory=val_pin_memory,
            drop_last=False,
            persistent_workers=val_persistent,
            collate_fn=val_collate,
        )
        use_dists = bool(val_cfg.get("use_dists", True))
        dists_device_name = str(val_cfg.get("dists_device", "cuda")).lower()
        if dists_device_name == "cuda" and torch.cuda.is_available():
            dists_device = torch.device("cuda")
        else:
            dists_device = torch.device("cpu")
        recreate_dists = bool(val_cfg.get("recreate_dists_each_val", False))
        empty_cache_after = bool(val_cfg.get("empty_cache_after", False))
        gc_collect = bool(val_cfg.get("gc_collect", False))
        if use_dists and not recreate_dists:
            dists_model = build_dists(dists_device)
            if dists_model is None:
                print("Warning: DISTS is not available. Install dists-pytorch to enable it.")
                use_dists = False

    val_every = int(val_cfg.get("every_epochs", 1)) if isinstance(val_cfg, dict) else 1
    val_tile = int(val_cfg.get("tile", 0)) if isinstance(val_cfg, dict) else 0
    val_overlap = int(val_cfg.get("overlap", 16)) if isinstance(val_cfg, dict) else 16
    val_max_images = int(val_cfg.get("max_images", 0)) if isinstance(val_cfg, dict) else 0

    for epoch in range(start_epoch, epochs):
        model.train()
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", dynamic_ncols=True)
        for step, (lr_img, hr_img) in enumerate(progress):
            lr_img = lr_img.to(device, non_blocking=True)
            hr_img = hr_img.to(device, non_blocking=True)

            if use_amp:
                ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
            else:
                ctx = nullcontext()

            with ctx:
                sr = model(lr_img)
                loss, loss_details = loss_fn(sr, hr_img)
                loss_scaled = loss / grad_accum

            if scaler.is_enabled():
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
                if grad_clip > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                if ema is not None:
                    ema.update(model)

                global_step += 1
                if global_step % log_every == 0:
                    lr_now = optimizer.param_groups[0]["lr"]
                    log_row = {
                        "epoch": epoch + 1,
                        "step": global_step,
                        "lr": lr_now,
                    }
                    log_row.update(loss_details)
                    append_jsonl(log_path, log_row)

            progress.set_postfix(loss=loss_details.get("total", 0.0))

        if val_loader is not None and (epoch + 1) % val_every == 0:
            model.eval()
            total_count = 0
            sum_psnr = 0.0
            sum_ssim = 0.0
            sum_dists = 0.0
            dists_model_epoch = dists_model
            if use_dists and recreate_dists:
                dists_model_epoch = build_dists(dists_device)
                if dists_model_epoch is None:
                    print("Warning: DISTS is not available. Install dists-pytorch to enable it.")
                    use_dists = False

            for batch in tqdm(val_loader, desc="Val", dynamic_ncols=True):
                if val_max_images and total_count >= val_max_images:
                    break
                if isinstance(batch, list):
                    samples = batch
                else:
                    samples = [batch]
                for lr_img, hr_img in samples:
                    if val_max_images and total_count >= val_max_images:
                        break
                    if lr_img.ndim == 3:
                        lr_img = lr_img.unsqueeze(0)
                    if hr_img.ndim == 3:
                        hr_img = hr_img.unsqueeze(0)
                    lr_img = lr_img.to(device, non_blocking=True)
                    hr_img = hr_img.to(device, non_blocking=True)

                    if use_amp:
                        amp_ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
                    else:
                        amp_ctx = nullcontext()

                    if val_tile and val_tile > 0:
                        sr = infer_tiled(
                            model, lr_img, int(data_cfg.get("scale", 2)), val_tile, val_overlap, amp_ctx
                        )
                    else:
                        with amp_ctx:
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
                    dists_vals = calc_dists(dists_model_epoch, sr, hr_img) if dists_model_epoch is not None else None

                    batch_size = sr.shape[0]
                    total_count += batch_size
                    sum_psnr += float(psnr_vals.sum().cpu())
                    sum_ssim += float(ssim_vals.sum().cpu())
                    if dists_vals is not None:
                        sum_dists += float(dists_vals.sum().cpu())

                    if val_max_images and total_count >= val_max_images:
                        break

            if total_count > 0:
                val_metrics = {
                    "epoch": epoch + 1,
                    "phase": "val",
                    "psnr": sum_psnr / total_count,
                    "ssim": sum_ssim / total_count,
                }
                if dists_model_epoch is not None:
                    val_metrics["dists"] = sum_dists / total_count
                append_jsonl(log_path, val_metrics)
            if recreate_dists and dists_model_epoch is not None:
                del dists_model_epoch
            if empty_cache_after and device.type == "cuda":
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
            if gc_collect:
                gc.collect()

        if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
            ckpt = {
                "epoch": epoch,
                "global_step": global_step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            if scaler.is_enabled():
                ckpt["scaler"] = scaler.state_dict()
            if ema is not None:
                ckpt["ema"] = ema.state_dict()

            ckpt_path = output_dir / f"ckpt_epoch_{epoch + 1:04d}.pt"
            torch.save(ckpt, ckpt_path)
            torch.save(ckpt, output_dir / "ckpt_last.pt")

            weights_state = ema.get_model_state(model) if ema is not None else model.state_dict()
            weights_path = output_dir / f"weights_epoch_{epoch + 1:04d}.pt"
            torch.save({"model": weights_state}, weights_path)
            torch.save({"model": weights_state}, output_dir / "weights_last.pt")


if __name__ == "__main__":
    main()
