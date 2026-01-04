import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def append_jsonl(path, data):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=True) + "\n")


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    def update(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    if self.shadow[name].device != param.device:
                        self.shadow[name] = self.shadow[name].to(param.device)
                    new = (1.0 - self.decay) * param.detach()
                    self.shadow[name].mul_(self.decay).add_(new)

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state):
        self.decay = state["decay"]
        self.shadow = state["shadow"]

    def get_model_state(self, model):
        state = model.state_dict()
        for name, value in self.shadow.items():
            if name in state:
                state[name] = value.clone()
        return state
