import argparse
from pathlib import Path
import sys

import torch

try:
    from safetensors.torch import save_file
except Exception:
    raise SystemExit("safetensors is required. Install with: python -m pip install safetensors")


def _extract_state_dict(obj, key):
    if key:
        if not isinstance(obj, dict) or key not in obj:
            raise SystemExit(f"Key '{key}' not found in checkpoint.")
        return obj[key]
    if isinstance(obj, dict):
        for candidate in ("model", "state_dict", "model_state_dict"):
            if candidate in obj and isinstance(obj[candidate], dict):
                return obj[candidate]
    return obj


def _resolve_output_path(input_path, output_path):
    input_path = Path(input_path)
    if output_path:
        output_path = Path(output_path)
        if output_path.is_dir():
            return output_path / f"{input_path.stem}.safetensors"
        return output_path
    return input_path.with_suffix(".safetensors")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to .pt/.pth checkpoint or weights file.")
    parser.add_argument("--output", type=str, default="", help="Output .safetensors path or directory.")
    parser.add_argument("--key", type=str, default="", help="Checkpoint key with state dict (e.g. model).")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    state = torch.load(input_path, map_location="cpu")
    state_dict = _extract_state_dict(state, args.key.strip() or None)
    if not isinstance(state_dict, dict):
        raise SystemExit("Loaded object is not a state dict. Use --key to select one from the checkpoint.")

    tensors = {}
    skipped = []
    for name, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            tensors[name] = value.detach().cpu()
        else:
            skipped.append((name, type(value).__name__))

    if not tensors:
        raise SystemExit("No tensors found to save.")

    if skipped:
        details = ", ".join([f"{name}({kind})" for name, kind in skipped[:5]])
        if len(skipped) > 5:
            details += f", ... +{len(skipped) - 5} more"
        print(f"Warning: skipped non-tensor entries: {details}", file=sys.stderr)

    output_path = _resolve_output_path(input_path, args.output.strip() or None)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output_path))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
