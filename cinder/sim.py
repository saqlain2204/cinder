"""One tablet, one question, one reward.

The observation is a picture and a prompt. The batch number is not in the prompt.
``batch`` asks for the fork tally, including the turn line. ``pins`` asks for the
pin count only. ``pair`` shows two tablets and asks which lower batch leaves first.
"""

from __future__ import annotations

import base64
import random
from dataclasses import dataclass

from cinder.parsing import parse_number, parse_side
from cinder.script import TRAIN_SIZE, Tablet, make_tablet, render, render_pair, tablet_from_parts

BATCH_PROMPT = "Read this annealing tablet.\nReply with the batch number."
PINS_PROMPT = "Read this annealing tablet.\nReply with the number of pins."
PAIR_PROMPT = "Two tablets sit on the fork.\nWhich one leaves the kiln first?\nReply left or right."


@dataclass
class Outcome:
    prompt: str
    task: str
    image: bytes
    reward: float | None
    done: bool
    legal_actions: tuple[str, ...]

    @property
    def image_base64(self) -> str:
        return base64.b64encode(self.image).decode("ascii")


def reward_for(task: str, tablet: Tablet, text: str) -> float:
    """Score a reply without drawing. Pair is scored through the simulator."""
    if task == "pins":
        target = tablet.pins
    elif task == "batch":
        target = tablet.value
    else:
        raise ValueError("reward_for scores a single tablet")
    guess = parse_number(text)
    if guess is None:
        return 0.0
    gap = abs(guess - target)
    if gap == 0:
        return 1.0
    if gap == 1:
        return 0.2
    return 0.0


class CinderSim:
    """In-process environment. ``reset`` then ``step`` once."""

    def __init__(self) -> None:
        self.task = "batch"
        self.steps = 0
        self.done = False
        self._ready = False
        self._rng = random.Random()
        self._tablet: Tablet | None = None
        self._left: Tablet | None = None
        self._right: Tablet | None = None
        self._size = TRAIN_SIZE

    @property
    def tablet(self) -> Tablet | None:
        return self._tablet

    @property
    def answer(self) -> str:
        """The scoring key. Training must not feed this back into the prompt."""
        if self.task == "pins":
            if self._tablet is None:
                raise RuntimeError("Call reset() first.")
            return str(self._tablet.pins)
        if self.task == "pair":
            if self._left is None or self._right is None:
                raise RuntimeError("Call reset() first.")
            return "left" if self._left.value < self._right.value else "right"
        if self._tablet is None:
            raise RuntimeError("Call reset() first.")
        return str(self._tablet.value)

    def reset(
        self,
        seed: int | None = None,
        task: str = "batch",
        size: int = TRAIN_SIZE,
        tablet: Tablet | None = None,
        left: Tablet | None = None,
        right: Tablet | None = None,
    ) -> Outcome:
        if task not in ("batch", "pins", "pair"):
            raise ValueError("task must be 'batch', 'pins', or 'pair'")
        self._rng = random.Random(seed)
        self.task = task
        self._size = size
        self.steps = 0
        self.done = False
        self._ready = True
        self._tablet = None
        self._left = None
        self._right = None
        if task == "pair":
            self._left = left or make_tablet(self._rng)
            self._right = right or make_tablet(self._rng)
            guard = 0
            while self._left.value == self._right.value and guard < 8:
                self._right = make_tablet(self._rng)
                guard += 1
            if self._left.value == self._right.value:
                other = (self._left.value + 3) % 10
                self._right = tablet_from_parts(other, quench=not self._left.quench, seed=self._left.seed + 1)
            image = render_pair(self._left, self._right, size)
            return self._view(image, reward=None, done=False)
        self._tablet = tablet or make_tablet(self._rng)
        return self._view(render(self._tablet, size), reward=None, done=False)

    def step(self, text: str) -> Outcome:
        if not self._ready:
            raise RuntimeError("Call reset() before step().")
        if self.done:
            raise RuntimeError("This episode is finished. Call reset().")
        self.steps += 1
        self.done = True
        reward = self._score(text)
        image = self._current_image()
        return self._view(image, reward=reward, done=True)

    def model_image(self) -> bytes:
        """The 64px picture the trained policy reads. Pair is not a trained task."""
        if self.task == "pair":
            raise RuntimeError("The policy is trained on a single tablet.")
        if self._tablet is None:
            raise RuntimeError("Call reset() first.")
        return render(self._tablet, TRAIN_SIZE)

    def _score(self, text: str) -> float:
        if self.task == "pair":
            side = parse_side(text)
            return 1.0 if side == self.answer else 0.0
        guess = parse_number(text)
        if guess is None:
            return 0.0
        target = int(self.answer)
        gap = abs(guess - target)
        if gap == 0:
            return 1.0
        if gap == 1:
            return 0.2
        return 0.0

    def _current_image(self) -> bytes:
        if self.task == "pair":
            assert self._left is not None and self._right is not None
            return render_pair(self._left, self._right, self._size)
        assert self._tablet is not None
        return render(self._tablet, self._size)

    def _view(self, image: bytes, reward: float | None, done: bool) -> Outcome:
        if self.task == "pair":
            prompt = PAIR_PROMPT
            legal = ("left", "right")
        elif self.task == "pins":
            prompt = PINS_PROMPT
            legal = tuple(str(i) for i in range(5))
        else:
            prompt = BATCH_PROMPT
            legal = tuple(str(i) for i in range(10))
        return Outcome(
            prompt=prompt,
            task=self.task,
            image=image,
            reward=reward,
            done=done,
            legal_actions=legal,
        )
