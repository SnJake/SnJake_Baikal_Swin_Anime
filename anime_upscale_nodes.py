import os
import shutil
import urllib.request
from contextlib import nullcontext
from typing import Dict, Tuple

import torch

try:
    import folder_paths
except Exception:
    folder_paths = None

try:
    from comfy import utils as comfy_utils
except Exception:
    comfy_utils = None

from .training_code.model import build_model
from .training_code.utils import load_config


_EMOJI = "\U0001F60E"
_CATEGORY = f"{_EMOJI} SnJake/Upscale"
_VALID_EXTS = (".pt", ".pth", ".ckpt", ".safetensors")
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "training_code", "config.yaml")
_WEIGHTS_BASE_URL = "https://huggingface.co/<USER>/<REPO>/resolve/main"
_REMOTE_WEIGHTS = {
    "Baikal_Swin_Anime_x2.safetensors": f"{_WEIGHTS_BASE_URL}/Baikal_Swin_Anime_x2.safetensors",
}


class AnimeUpscaleModel:
    def __init__(self, model, scale: int, amp_mode: str, config_path: str, weights_path: str):
        self.model = model
        self.scale = int(scale)
        self.amp_mode = amp_mode
        self.config_path = config_path
        self.weights_path = weights_path

    def to(self, device):
        self.model = self.model.to(device, memory_format=torch.channels_last)
        return self

    def __call__(self, lr):
        with torch.inference_mode():
            return self.model(lr)


_MODEL_CACHE: Dict[Tuple[str, str], AnimeUpscaleModel] = {}


def _resolve_models_dir() -> str:
    if folder_paths is not None and hasattr(folder_paths, "models_dir"):
        base = folder_paths.models_dir
    else:
        base = os.path.join(os.getcwd(), "models")
    path = os.path.join(base, "anime_upscale")
    os.makedirs(path, exist_ok=True)
    try:
        if folder_paths is not None and hasattr(folder_paths, "add_model_search_path"):
            folder_paths.add_model_search_path("anime_upscale", path)
    except Exception:
        pass
    return path


def _list_models() -> list[str]:
    root = _resolve_models_dir()
    try:
        names = [
            name
            for name in os.listdir(root)
            if name.lower().endswith(_VALID_EXTS) and os.path.isfile(os.path.join(root, name))
        ]
    except Exception:
        names = []
    names.sort()
    return names


def _weight_choices() -> list[str]:
    local_names = _list_models()
    remote_names = list(_REMOTE_WEIGHTS.keys())
    choices = local_names + [name for name in remote_names if name not in local_names]
    return choices if choices else ["<none found>"]


def _download_weights(url: str, dst_path: str) -> None:
    tmp_path = f"{dst_path}.tmp"
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ComfyUI-SnJake-AnimeUpscale"},
    )
    try:
        with urllib.request.urlopen(request) as response, open(tmp_path, "wb") as handle:
            shutil.copyfileobj(response, handle)
        os.replace(tmp_path, dst_path)
    except Exception:
        try:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise


def _resolve_weights_path(weights_name: str) -> str:
    root = _resolve_models_dir()

    if not weights_name or weights_name in ("<none found>",):
        raise FileNotFoundError("Anime upscale weights are not configured.")

    if os.path.basename(weights_name) != weights_name:
        raise ValueError("Weights name must be a filename without directories.")

    candidate = os.path.join(root, weights_name)
    if os.path.isfile(candidate):
        return candidate

    url = _REMOTE_WEIGHTS.get(weights_name)
    if url:
        if "<USER>" in url or "<REPO>" in url:
            raise FileNotFoundError(
                "Weights URL placeholders are not configured. Update _WEIGHTS_BASE_URL/_REMOTE_WEIGHTS "
                "or download weights manually to the models/anime_upscale folder."
            )
        print(f"Downloading anime upscale weights: {weights_name}")
        _download_weights(url, candidate)
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        f"Anime upscale weights not found: '{weights_name}'."
    )


def _load_checkpoint_any(path: str):
    if comfy_utils is not None:
        return comfy_utils.load_torch_file(path, safe_load=True)
    return torch.load(path, map_location="cpu", weights_only=False)


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            return ckpt["model"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
    return ckpt


def _resolve_amp_dtype(mode: str, default_mode: str):
    mode_in = (mode or "auto").lower()
    fallback = (default_mode or "auto").lower()
    if mode_in == "auto":
        mode_in = fallback if fallback in ("bf16", "fp16", "auto", "none") else "auto"

    if mode_in == "none" or not torch.cuda.is_available():
        return None
    if mode_in == "bf16":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if mode_in == "fp16":
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _infer_full(model, lr, amp_ctx):
    with amp_ctx:
        return model(lr)


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


class SnJakeAnimeUpscaleCheckpointLoader:
    @classmethod
    def INPUT_TYPES(cls):
        names = _weight_choices()
        default_name = names[0]

        return {
            "required": {
                "weights_name": (
                    names,
                    {"default": default_name, "tooltip": "Select weights (auto-download if missing)."},
                ),
                "force_reload": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("ANIME_UPSCALE_MODEL", "UPSCALE_MODEL")
    RETURN_NAMES = ("upscale_model_custom", "upscale_model")
    FUNCTION = "load"
    CATEGORY = _CATEGORY

    def load(self, weights_name, force_reload):
        config_path = _CONFIG_PATH
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config file not found: '{config_path}'")

        weights_path = _resolve_weights_path(weights_name)
        cache_key = (weights_path, config_path)

        if not force_reload and cache_key in _MODEL_CACHE:
            cached = _MODEL_CACHE[cache_key]
            return (cached, cached)

        cfg = load_config(config_path)
        model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
        scale = int(cfg.get("data", {}).get("scale", 2)) if isinstance(cfg, dict) else 2
        amp_mode = str(cfg.get("train", {}).get("amp", "auto")).lower() if isinstance(cfg, dict) else "auto"

        model = build_model(model_cfg, scale=scale)
        ckpt = _load_checkpoint_any(weights_path)
        state_dict = _extract_state_dict(ckpt)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model = model.to("cpu", memory_format=torch.channels_last)

        wrapper = AnimeUpscaleModel(
            model=model,
            scale=scale,
            amp_mode=amp_mode,
            config_path=config_path,
            weights_path=weights_path,
        )
        _MODEL_CACHE[cache_key] = wrapper
        return (wrapper, wrapper)


class SnJakeAnimeUpscaleInference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscale_model_custom": ("ANIME_UPSCALE_MODEL",),
                "image": ("IMAGE",),
                "tile": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 16,
                        "tooltip": "Recommended tile size: 256-512 (0 disables tiling).",
                    },
                ),
                "overlap": (
                    "INT",
                    {
                        "default": 16,
                        "min": 0,
                        "max": 1024,
                        "step": 4,
                        "tooltip": "Recommended overlap: 32-64.",
                    },
                ),
                "amp": (["auto", "bf16", "fp16", "none"], {"default": "auto"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale"
    CATEGORY = _CATEGORY

    def upscale(self, upscale_model_custom, image, tile, overlap, amp, device):
        if upscale_model_custom is None:
            raise ValueError("Upscale model is missing.")

        if device == "auto":
            device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device_t = torch.device(device)

        model = upscale_model_custom.model.to(device_t, memory_format=torch.channels_last)
        model.eval()

        amp_dtype = _resolve_amp_dtype(amp, upscale_model_custom.amp_mode)
        amp_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
            if amp_dtype is not None and device_t.type == "cuda"
            else nullcontext()
        )

        if image.dim() != 4:
            raise ValueError("Expected image tensor with shape [B, H, W, C].")

        b, h, w, c = image.shape
        if c != 3:
            raise ValueError("Anime upscale expects 3-channel RGB images.")

        outputs = []
        with torch.inference_mode():
            for i in range(b):
                lr = image[i].permute(2, 0, 1).contiguous().unsqueeze(0)
                lr = lr.to(device_t, memory_format=torch.channels_last)

                if tile and tile > 0:
                    sr = _infer_tiled(model, lr, upscale_model_custom.scale, int(tile), int(overlap), amp_ctx)
                else:
                    sr = _infer_full(model, lr, amp_ctx)

                sr = sr.float().clamp(0, 1)
                out = sr.squeeze(0).permute(1, 2, 0).contiguous().to(image.device)
                outputs.append(out.unsqueeze(0))

        result = torch.cat(outputs, dim=0)
        result = result.to(image.dtype)
        return (result,)
