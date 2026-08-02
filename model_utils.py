"""Shared model construction, transforms, and checkpoint helpers.

Used by train.py, evaluate.py, and predict.py so the three entry points stay
consistent on how the model is built and how a checkpoint is loaded for
inference.
"""

import copy
import os

import torch
import torch.nn as nn
from timm.models import vit_small_patch16_224
from torchvision import transforms

IMAGE_SIZE = 224
NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD = [0.5, 0.5, 0.5]


def build_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Construct ViT-Small/patch16/224 with a fresh binary classification head."""
    model = vit_small_patch16_224(pretrained=pretrained)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model


def build_eval_transform() -> transforms.Compose:
    """Resize/tensor/normalize pipeline shared by validation, evaluation, and inference."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])


def load_inference_model(checkpoint_path: str, device: torch.device, num_classes: int = 2) -> nn.Module:
    """Load a checkpoint for inference/evaluation, preferring EMA weights when present."""
    model = build_model(num_classes=num_classes, pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = ckpt.get("ema_state_dict")
    if state is None:
        state = ckpt.get("model_state_dict")
    if state is None:
        state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def build_param_groups(model: nn.Module, weight_decay: float) -> list:
    """Split parameters into decay/no-decay groups (no weight decay on biases or 1-D norm params)."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def validate_paths(paths: dict) -> None:
    """Raise a clear error up front listing every path in `paths` (name -> path) that doesn't exist."""
    missing = [f"  --{name}: {path}" for name, path in paths.items() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("The following paths do not exist:\n" + "\n".join(missing))


class ModelEma:
    """Exponential moving average of a model's floating-point parameters and buffers."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.module = copy.deepcopy(model)
        self.module.eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay

    def to(self, device: torch.device) -> "ModelEma":
        self.module.to(device)
        return self

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_sd = self.module.state_dict()
        for k, v in model.state_dict().items():
            ema_v = ema_sd[k]
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                ema_v.copy_(v)

    def state_dict(self) -> dict:
        return self.module.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.module.load_state_dict(state_dict)
