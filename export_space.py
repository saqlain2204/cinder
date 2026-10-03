"""Write the static Space catalog from the real renderer and the trained policy."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch

from cinder.grpo import greedy_reply
from cinder.policy import load_policy
from cinder.script import TRAIN_SIZE, make_tablet, render, render_pair
from cinder.sim import BATCH_PROMPT, PAIR_PROMPT, PINS_PROMPT, CinderSim

ROOT = Path(__file__).resolve().parent / "space"
TABLETS = ROOT / "tablets"


def main() -> None:
    TABLETS.mkdir(parents=True, exist_ok=True)
    model, _ = load_policy(Path(__file__).resolve().parent / "cinder" / "weights" / "policy.pt")
    device = torch.device("cpu")
    rng = random.Random(7)
    tablets = []
    batch_hit = pins_hit = 0
    for index in range(24):
        tablet = make_tablet(rng)
        name = f"t{index:02d}.png"
        (TABLETS / name).write_bytes(render(tablet, 360))
        small = render(tablet, TRAIN_SIZE)
        batch = greedy_reply(model, small, BATCH_PROMPT, "batch", device)
        pins = greedy_reply(model, small, PINS_PROMPT, "pins", device)
        batch_hit += batch == str(tablet.value)
        pins_hit += pins == str(tablet.pins)
        tablets.append(
            {
                "image": f"tablets/{name}",
                "value": str(tablet.value),
                "pins": str(tablet.pins),
                "batchModel": batch,
                "pinsModel": pins,
            }
        )
    sim = CinderSim()
    pairs = []
    for index in range(8):
        outcome = sim.reset(seed=4000 + index, task="pair", size=300)
        name = f"p{index:02d}.png"
        (TABLETS / name).write_bytes(outcome.image)
        pairs.append({"image": f"tablets/{name}", "reading": sim.answer})
    payload = {"tablets": tablets, "pairs": pairs}
    (ROOT / "catalog.json").write_text(json.dumps(payload), encoding="utf-8")
    print(f"tablets {len(tablets)} batch {batch_hit}/{len(tablets)} pins {pins_hit}/{len(tablets)} pairs {len(pairs)}")


if __name__ == "__main__":
    main()
