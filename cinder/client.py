"""Client for a running Cinder OpenEnv server."""

from __future__ import annotations

from typing import Dict

try:
    from openenv.core import EnvClient
    from openenv.core.client_types import StepResult
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        'OpenEnv is not installed. Install this project with: pip install -e ".[openenv]"'
    ) from exc

from cinder.models import CinderAction, CinderObservation, CinderState


class CinderEnv(EnvClient[CinderAction, CinderObservation, CinderState]):
    """WebSocket client. Start the server with ``python -m cinder.server.app``."""

    def _step_payload(self, action: CinderAction) -> Dict:
        return {"text": action.text}

    def _parse_result(self, payload: Dict) -> StepResult[CinderObservation]:
        obs = payload.get("observation", {})
        observation = CinderObservation(
            prompt=obs.get("prompt", ""),
            task=obs.get("task", "batch"),
            image_base64=obs.get("image_base64", ""),
            legal_actions=list(obs.get("legal_actions") or []),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=payload.get("metadata") or obs.get("metadata") or {},
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=payload.get("metadata"),
        )

    def _parse_state(self, payload: Dict) -> CinderState:
        return CinderState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task=payload.get("task", "batch"),
        )
