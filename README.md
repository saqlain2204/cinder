# Cinder

A clay tablet from an annealing shed. The batch number is not written in digits. You reply, and the environment scores you.

![A tablet with one notch and one pin](docs/tablet.png)

The night shift at one glassworks marks batches in a tally used inside that shed and nowhere else.

- A notch in the top of the tablet is five.
- A pin in the lower field is one.
- That sum is the raw tally.
- A copper line across the middle means the tablet was turned on the fork, and the batch number is nine minus the raw tally.
- Specks along the sides are ash. They are not pins.

The picture never contains a decimal digit, and the prompt does not contain the number. A finished batch episode is one digit, from 0 to 9.

## The environment

The model does not own the shed. It cannot pick the tablet, and it must not be handed the reading. Another program chooses the clay, draws the picture, writes the question, and scores the reply. That program is the environment.

An episode is three calls.

`reset` starts it and returns an observation: the picture and the prompt. The action is the reply. `step` sends that reply and returns a reward. Then the episode is over. The reading is part of the score, not part of the observation.

The tally is not a sentence sitting in the weights at the start. Those weights begin random. A convolution can see pixels, and a row of characters can see the question, but a notch and a pin are not in there yet. The only teacher is the number that comes back after the reply.

An [OpenEnv](https://github.com/huggingface/OpenEnv) server exposes the same `reset` and `step`, so a trainer can run the episode without holding the renderer.

## Three questions

**Batch.** One step. Reply with the batch number, turn line included. An exact digit scores 1. A neighbor scores 0.2. Anything else scores 0. There is one correct digit, so the reward points at it.

```
Read this annealing tablet.
Reply with the batch number.
```

**Pins.** The same tablet and a different sentence. Reply with how many pins are in the lower field. Notches and the copper line do not count. The picture can stay still while the words change the answer.

**Fork.** Two tablets. The lower batch leaves the kiln first. Reply left or right. The score is 1 on the lower batch and 0 otherwise. The policy below was trained on a single tablet. Fork is there to play.

## Training

The policy is a small vision-language model. A short convolution reads the tablet. The prompt is summed in as characters. The reply is one token.

Each step lays out every legal reply for one tablet, scores them in the environment, and raises the probability of a reply in proportion to the reward it received. A tied group would teach nothing. Here the scores differ, so the weights move toward the higher ones. The digit is never a sentence to copy.

On 400 held-out tablets, before any update and after 400 steps:

| Question | Before | After |
| --- | ---: | ---: |
| Batch number | 10.8% | 88.5% |
| Pin count | 10.3% | 95.3% |

When the batch number and the pin count are different numbers, the two prompts get different replies 77.3% of the time after training, and 10.5% before.

```bash
pip install -e ".[train]"
python -m cinder.grpo
```

## Play

```bash
pip install -e .
python -m cinder.play
```

Open `http://127.0.0.1:8091`. Type a reply, or ask the trained policy. The reading is shown after the reward.

## OpenEnv server

```bash
pip install -e ".[openenv]"
python -m cinder.server.app --port 8000
```

```python
from cinder.client import CinderEnv
from cinder.models import CinderAction

with CinderEnv(base_url="http://127.0.0.1:8000").sync() as env:
    result = env.reset()
    result = env.step(CinderAction(text="4"))
    print(result.reward)
```

## Layout

| Path | What it is |
| --- | --- |
| `cinder/script.py` | The tally and the tablet renderer |
| `cinder/sim.py` | `reset` and `step` |
| `cinder/grpo.py` | Reward-weighted training |
| `cinder/policy.py` | The vision-language policy |
| `cinder/play.py` | The local page |
| `cinder/server/app.py` | The OpenEnv server |
| `space/` | The static essay and the playable tally |
