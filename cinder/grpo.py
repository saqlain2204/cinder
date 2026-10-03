"""Reward-weighted updates on the fork tally.

Each step shows a few tablets and scores every legal reply. The update raises
the probability of a reply in proportion to the reward it just received.
The correct digit is not a target sentence. It matters because it is the
reply that scores 1.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from cinder.policy import (
    CinderVLM,
    decode_token,
    encode_prompt,
    legal_ids,
    load_image,
    save_policy,
)
from cinder.script import TRAIN_SIZE
from cinder.sim import CinderSim, reward_for


def group_advantages(rewards: list[float]) -> torch.Tensor:
    """Standardize rewards inside one group. A tied group gets advantage 0."""
    scores = torch.tensor(rewards, dtype=torch.float32)
    if float(scores.std(unbiased=False)) < 1e-6:
        return torch.zeros_like(scores)
    return (scores - scores.mean()) / scores.std(unbiased=False).clamp_min(1e-6)


def _masked_logits(logits: torch.Tensor, task: str) -> torch.Tensor:
    allowed = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
    allowed[legal_ids(task)] = True
    return logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)


@torch.no_grad()
def greedy_reply(model: CinderVLM, png: bytes, prompt: str, task: str, device: torch.device) -> str:
    model.eval()
    image = load_image(png).unsqueeze(0).to(device)
    tokens = torch.tensor([encode_prompt(prompt)], dtype=torch.long, device=device)
    logits = _masked_logits(model(image, tokens), task)
    return decode_token(int(logits.argmax(dim=-1).item()))


@torch.no_grad()
def evaluate(model: CinderVLM, n: int = 300, seed: int = 50_000, device: torch.device | None = None) -> dict:
    """Greedy exact-match on held-out tablets, for the batch question and the pin question."""
    device = device or next(model.parameters()).device
    model.eval()
    sim = CinderSim()
    images = []
    tablets = []
    for index in range(n):
        sim.reset(seed=seed + index, task="batch", size=TRAIN_SIZE)
        assert sim.tablet is not None
        tablets.append(sim.tablet)
        images.append(load_image(sim.model_image()))
    picture = torch.stack(images, dim=0).to(device)
    replies: dict[str, list[str]] = {}
    for task in ("batch", "pins"):
        sim.reset(tablet=tablets[0], task=task, size=TRAIN_SIZE)
        prompt = torch.tensor([encode_prompt(sim.reset(tablet=tablets[0], task=task, size=TRAIN_SIZE).prompt)] * n, device=device)
        chosen = []
        for start in range(0, n, 64):
            logits = _masked_logits(model(picture[start : start + 64], prompt[start : start + 64]), task)
            chosen.extend(int(token) for token in logits.argmax(dim=-1).tolist())
        replies[task] = [decode_token(token) for token in chosen]
    batch_hit = pins_hit = follow_hit = follow_n = 0
    batch_reward = pins_reward = 0.0
    for tablet, batch_text, pins_text in zip(tablets, replies["batch"], replies["pins"]):
        batch_hit += int(batch_text == str(tablet.value))
        pins_hit += int(pins_text == str(tablet.pins))
        sim.reset(tablet=tablet, task="batch", size=TRAIN_SIZE)
        batch_reward += float(sim.step(batch_text).reward or 0.0)
        sim.reset(tablet=tablet, task="pins", size=TRAIN_SIZE)
        pins_reward += float(sim.step(pins_text).reward or 0.0)
        if tablet.value != tablet.pins:
            follow_n += 1
            follow_hit += int(batch_text != pins_text)
    return {
        "n": n,
        "seed": seed,
        "batch_exact": batch_hit / n,
        "pins_exact": pins_hit / n,
        "batch_reward": batch_reward / n,
        "pins_reward": pins_reward / n,
        "prompt_follows": (follow_hit / follow_n) if follow_n else 0.0,
    }


def train(
    steps: int = 700,
    groups_per_step: int = 8,
    lr: float = 3e-3,
    entropy_coef: float = 0.02,
    temperature: float = 1.0,
    seed: int = 0,
    eval_every: int = 100,
    eval_n: int = 160,
    output: str | Path = "cinder/weights/policy.pt",
    focus: str | None = None,
) -> dict:
    """Train from reward. Returns the before and after scores on held-out tablets."""
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    model = CinderVLM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sim = CinderSim()
    initial_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    glimpse = evaluate(model, n=min(80, eval_n), device=device)
    print(
        f"before  batch {glimpse['batch_exact']:.3f}  pins {glimpse['pins_exact']:.3f}  "
        f"follows {glimpse['prompt_follows']:.3f}"
    )
    history: list[float] = []
    best = glimpse
    best_state = {key: value.detach().cpu().clone() for key, value in initial_state.items()}
    tasks = ("batch", "pins")
    for step in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        coef = entropy_coef * (0.15 + 0.85 * (1.0 - step / max(steps - 1, 1)))
        task = focus or tasks[step % 2]
        pictures = []
        reward_rows: list[list[float]] = []
        prompt_text = ""
        allowed = legal_ids(task)
        replies = [decode_token(token) for token in allowed]
        for group_index in range(groups_per_step):
            outcome = sim.reset(
                seed=seed + step * groups_per_step + group_index,
                task=task,
                size=TRAIN_SIZE,
            )
            prompt_text = outcome.prompt
            tablet = sim.tablet
            assert tablet is not None
            pictures.append(load_image(sim.model_image()))
            reward_rows.append([reward_for(task, tablet, reply) for reply in replies])
        picture = torch.stack(pictures, dim=0).to(device)
        prompt = torch.tensor([encode_prompt(prompt_text)] * groups_per_step, dtype=torch.long, device=device)
        logits = _masked_logits(model(picture, prompt), task)
        dist = torch.distributions.Categorical(logits=logits / max(temperature, 1e-6))
        log_prob = torch.stack(
            [dist.log_prob(torch.full((groups_per_step,), token, device=device)) for token in allowed],
            dim=1,
        )
        rewards = torch.tensor(reward_rows, dtype=torch.float32, device=device)
        loss = -(rewards * log_prob).sum(dim=1).mean() - coef * dist.entropy().mean()
        loss.backward()
        flat_rewards = [score for row in reward_rows for score in row]
        step_reward = sum(flat_rewards) / len(flat_rewards)
        step_best = sum(max(row) for row in reward_rows) / groups_per_step
        step_entropy = float(dist.entropy().mean().detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        history.append(step_reward)
        if step % 25 == 0 or step + 1 == steps:
            print(
                f"step {step + 1:4d}  reward {step_reward:.3f}  "
                f"best {step_best:.3f}  entropy {step_entropy:.2f}"
            )
        if eval_every and (step + 1) % eval_every == 0:
            scores = evaluate(model, n=eval_n, device=device)
            print(
                f"eval    batch {scores['batch_exact']:.3f}  pins {scores['pins_exact']:.3f}  "
                f"follows {scores['prompt_follows']:.3f}"
            )
            mark = scores["batch_exact"] + scores["pins_exact"]
            if mark >= best["batch_exact"] + best["pins_exact"]:
                best = scores
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            if scores["batch_exact"] >= 0.95 and scores["pins_exact"] >= 0.95 and scores["prompt_follows"] >= 0.90:
                print("held-out scores cleared the bar")
                break
    model.load_state_dict(best_state)
    after = evaluate(model, n=400, seed=50_000, device=device)
    model.load_state_dict(initial_state)
    before = evaluate(model, n=400, seed=50_000, device=device)
    model.load_state_dict(best_state)
    metrics = {"before": before, "after": after, "steps": len(history), "history_tail": history[-10:]}
    destination = Path(output)
    save_policy(model, destination, metrics)
    log_path = destination.parent / "metrics.json"
    log_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"after   batch {after['batch_exact']:.3f}  pins {after['pins_exact']:.3f}  "
        f"follows {after['prompt_follows']:.3f}"
    )
    print(f"saved {destination}")
    return metrics


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train the cinder policy from reward")
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--out", default="cinder/weights/policy.pt")
    args = parser.parse_args()
    train(steps=args.steps, output=args.out)


if __name__ == "__main__":
    main()
