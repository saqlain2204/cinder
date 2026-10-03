"""OpenEnv adapter around CinderSim."""

from __future__ import annotations

from uuid import uuid4

try:
    from openenv.core.env_server.interfaces import Environment
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        'OpenEnv is not installed. Install this project with: pip install -e ".[openenv]"'
    ) from exc

from cinder.models import CinderAction, CinderObservation, CinderState
from cinder.sim import CinderSim, Outcome


class CinderEnvironment(Environment):
    """One session. Each instance keeps its own tablet."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._sim = CinderSim()
        self._episode_id = str(uuid4())

    def reset(self, seed=None, episode_id=None, **kwargs) -> CinderObservation:
        self._episode_id = episode_id or str(uuid4())
        return self._observation(self._sim.reset(seed=seed, **kwargs))

    def step(self, action: CinderAction, timeout_s=None, **kwargs) -> CinderObservation:
        del timeout_s, kwargs
        text = action.text if isinstance(action, CinderAction) else str(getattr(action, "text", action))
        return self._observation(self._sim.step(text))

    @property
    def state(self) -> CinderState:
        return CinderState(
            episode_id=self._episode_id,
            step_count=self._sim.steps,
            task=self._sim.task,
        )

    def _observation(self, outcome: Outcome) -> CinderObservation:
        return CinderObservation(
            prompt=outcome.prompt,
            task=outcome.task,
            image_base64=outcome.image_base64,
            legal_actions=list(outcome.legal_actions),
            done=outcome.done,
            reward=outcome.reward,
        )
