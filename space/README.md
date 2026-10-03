---
title: Cinder
emoji: 🔥
colorFrom: gray
colorTo: red
sdk: static
pinned: false
license: mit
---

# The tally on the clay

A clay tablet from an annealing shed. The batch number is not written in digits. You reply, and the environment scores you.

The night shift at one glassworks marks annealing batches on clay. A notch is five. A pin is one. A copper line across the middle means the batch number is nine minus that sum. Specks along the sides are ash. The picture has no decimal digit, and the prompt does not contain the number.

## What an environment is

The model does not own the shed. Another program chooses the clay, draws the picture, writes the question, and scores the reply. That program is the environment.

`reset` returns the picture and the prompt. The action is one reply. `step` returns a reward, and the episode ends. The reading arrives with the score.

The weights start random. A notch and a pin are not in them yet. The only teacher is the reward.

## Three questions

**Batch.** One step. An exact digit scores 1. A neighbor scores 0.2. Anything else scores 0.

**Pins.** The same tablet, a different sentence: how many pins. Notches and the copper line do not count.

**Fork.** Two tablets. The lower batch leaves the kiln first. Reply left or right. The policy was trained on a single tablet.

## Why the update is a reward

The policy is a small vision-language model: a convolution over the tablet, plus the characters of the prompt. Each training step scores every legal reply and raises the probability of a reply in proportion to that reward. The digit is never a sentence to copy.

An OpenEnv server is the same `reset` and `step` for another program.

On 400 held-out tablets the batch number moved from 10.8% to 88.5%, and the pin count from 10.3% to 95.3%. When those two answers differ, the two prompts get different replies 77.3% of the time after training, and 10.5% before.

The playable tally is on the page above.
