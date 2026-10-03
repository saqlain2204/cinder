"""A small vision-language policy.

The tablet is a picture. The prompt is a row of characters. The reply is one
character.
Training uses the environment reward only. There is no target sentence in the loss.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from cinder.script import TRAIN_SIZE

ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789 \n.,?'-"
PAD = 0
EOS = 1
LEFT = 2 + len(ALPHABET)
RIGHT = LEFT + 1
VOCAB = RIGHT + 1
D_MODEL = 32
MAX_PROMPT = 72


def token_id(char: str) -> int:
    return 2 + ALPHABET.index(char)


def digit_ids() -> list[int]:
    return [token_id(str(i)) for i in range(10)]


def pin_ids() -> list[int]:
    return [token_id(str(i)) for i in range(5)]


def side_ids() -> list[int]:
    return [LEFT, RIGHT]


def legal_ids(task: str) -> list[int]:
    if task == "pins":
        return pin_ids()
    if task == "pair":
        return side_ids()
    if task == "batch":
        return digit_ids()
    raise ValueError(f"unknown task {task}")


def encode_prompt(text: str, length: int = MAX_PROMPT) -> list[int]:
    ids = [token_id(char) for char in text.lower() if char in ALPHABET]
    ids = ids[:length]
    return ids + [PAD] * (length - len(ids))


def decode_token(token: int) -> str:
    if token == LEFT:
        return "left"
    if token == RIGHT:
        return "right"
    index = token - 2
    if 0 <= index < len(ALPHABET):
        return ALPHABET[index]
    return ""


class CinderVLM(nn.Module):
    """A small conv net over the tablet, conditioned on the characters of the prompt."""

    def __init__(self) -> None:
        super().__init__()
        d = D_MODEL
        self.conv = nn.Conv2d(3, 8, kernel_size=5, stride=4, padding=2)
        self.prompt_embed = nn.Embedding(VOCAB, d)
        view = 8 * 16 * 16
        self.head = nn.Sequential(
            nn.Linear(view + d, 128),
            nn.ReLU(),
            nn.Linear(128, VOCAB),
        )
        nn.init.normal_(self.head[-1].weight, std=0.02)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, image: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        """Return reply logits. ``image`` is B,3,64,64 and ``prompt`` is B,P."""
        if image.shape[-2:] != (TRAIN_SIZE, TRAIN_SIZE):
            raise ValueError(f"expected a {TRAIN_SIZE}px tablet, got {tuple(image.shape[-2:])}")
        centered = image - image.mean(dim=(2, 3), keepdim=True)
        centered = centered / centered.std(dim=(2, 3), keepdim=True).clamp_min(1e-3)
        features = torch.nn.functional.relu(self.conv(centered)).flatten(1)
        mask = prompt.ne(PAD).float().unsqueeze(-1)
        summary = (self.prompt_embed(prompt) * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.head(torch.cat([features, summary], dim=-1))


def load_image(png: bytes) -> torch.Tensor:
    """PNG bytes to a 3,H,W float tensor in 0..1."""
    from io import BytesIO

    import numpy as np
    from PIL import Image

    image = Image.open(BytesIO(png)).convert("RGB")
    array = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255.0
    return array.contiguous()


def save_policy(model: CinderVLM, path: str | Path, metrics: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "metrics": metrics}, path)


def load_policy(path: str | Path, device: torch.device | None = None) -> tuple[CinderVLM, dict]:
    device = device or torch.device("cpu")
    blob = torch.load(Path(path), map_location=device, weights_only=False)
    model = CinderVLM()
    model.load_state_dict(blob["state"])
    model.to(device)
    model.eval()
    return model, blob.get("metrics", {})
