"""The fork tally, the reward, and one policy step."""

from __future__ import annotations

import pytest

from cinder.grpo import group_advantages
from cinder.parsing import parse_number, parse_side
from cinder.script import make_tablet, render, tablet_from_parts
from cinder.sim import BATCH_PROMPT, PAIR_PROMPT, PINS_PROMPT, CinderSim


def test_every_batch_number_roundtrips():
    for value in range(10):
        for quench in (False, True):
            tablet = tablet_from_parts(value, quench, seed=value + 10)
            raw = 5 * tablet.notches + tablet.pins
            assert tablet.notches in (0, 1)
            assert 0 <= tablet.pins <= 4
            assert (9 - raw if quench else raw) == value


def test_render_is_a_deterministic_png():
    tablet = tablet_from_parts(7, True, seed=4)
    first = render(tablet, 64)
    second = render(tablet, 64)
    assert first.startswith(b"\x89PNG")
    assert first == second
    assert render(tablet_from_parts(7, False, seed=4), 64) != first


def test_prompts_hide_the_reading():
    sim = CinderSim()
    prompts = set()
    for seed in range(20):
        outcome = sim.reset(seed=seed, task="batch", size=64)
        prompts.add(outcome.prompt)
        assert outcome.prompt == BATCH_PROMPT
        assert str(sim.answer) not in outcome.prompt
    assert prompts == {BATCH_PROMPT}
    pins = sim.reset(seed=3, task="pins", size=64)
    assert pins.prompt == PINS_PROMPT
    assert pins.prompt != BATCH_PROMPT


def test_reward_peaks_on_the_exact_digit():
    sim = CinderSim()
    tablet = tablet_from_parts(6, False, seed=1)
    sim.reset(tablet=tablet, task="batch", size=64)
    assert sim.step("6").reward == 1.0
    sim.reset(tablet=tablet, task="batch", size=64)
    assert sim.step("The batch is 7.").reward == 0.2
    sim.reset(tablet=tablet, task="batch", size=64)
    assert sim.step("2").reward == 0.0
    sim.reset(tablet=tablet, task="batch", size=64)
    assert sim.step("no number").reward == 0.0
    with pytest.raises(RuntimeError):
        sim.step("6")


def test_pins_ignore_the_turn_line():
    turned = tablet_from_parts(0, True, seed=8)
    assert turned.raw == 9
    assert turned.pins == 4
    sim = CinderSim()
    sim.reset(tablet=turned, task="pins", size=64)
    assert sim.step("4").reward == 1.0
    sim.reset(tablet=turned, task="batch", size=64)
    assert sim.step("0").reward == 1.0
    sim.reset(tablet=turned, task="batch", size=64)
    assert sim.step("4").reward == 0.0


def test_pair_picks_the_lower_batch_and_hides_it():
    sim = CinderSim()
    left = tablet_from_parts(2, False, seed=1)
    right = tablet_from_parts(8, True, seed=2)
    first = sim.reset(task="pair", left=left, right=right, size=64)
    second = sim.reset(task="pair", left=right, right=left, size=64)
    assert first.prompt == second.prompt == PAIR_PROMPT
    assert "2" not in first.prompt and "8" not in first.prompt
    sim.reset(task="pair", left=left, right=right, size=64)
    assert sim.step("left").reward == 1.0
    sim.reset(task="pair", left=left, right=right, size=64)
    assert sim.step("take the right one").reward == 0.0


def test_same_seed_repeats_the_tablet():
    sim = CinderSim()
    sim.reset(seed=19, task="batch", size=64)
    first = sim.tablet
    sim.reset(seed=19, task="batch", size=64)
    assert sim.tablet == first
    labels = {make_tablet(__import__("random").Random(i)).value for i in range(200)}
    assert labels == set(range(10))


def test_parsers_use_the_last_mention():
    assert parse_number("batch 4, or maybe 8") == 8
    assert parse_number("ten") is None
    assert parse_number("10") is None
    assert parse_side("not right, left") == "left"


def test_tied_group_has_no_advantage():
    assert group_advantages([1.0, 1.0, 1.0]).tolist() == [0.0, 0.0, 0.0]
    scores = group_advantages([0.0, 1.0])
    assert float(scores[1]) > 0 > float(scores[0])


def test_openenv_observation_hides_the_number():
    pytest.importorskip("openenv")
    try:
        from cinder.environment import CinderEnvironment
    except ImportError:
        pytest.skip("OpenEnv imports, but its server extras are not installed")

    env = CinderEnvironment()
    observation = env.reset(seed=2, task="batch", size=64)
    assert observation.reward is None
    assert observation.image_base64
    dumped = observation.model_dump()
    assert "value" not in dumped
    assert str(env._sim.answer) not in observation.prompt


def test_policy_reply_is_a_legal_digit():
    torch = pytest.importorskip("torch")
    from cinder.policy import CinderVLM, digit_ids, encode_prompt, legal_ids

    torch.manual_seed(0)
    model = CinderVLM()
    image = torch.rand(2, 3, 64, 64)
    prompt = torch.tensor([encode_prompt("Reply with the batch number.")] * 2)
    logits = model(image, prompt)
    assert logits.shape == (2, logits.shape[-1])
    assert torch.isfinite(logits).all()
    allowed = legal_ids("batch")
    assert allowed == digit_ids()
    assert len(allowed) == 10
