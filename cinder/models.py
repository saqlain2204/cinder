"""Action and observation types for the OpenEnv server."""

from __future__ import annotations

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        'OpenEnv is not installed. Install this project with: pip install -e ".[openenv]"'
    ) from exc

from pydantic import Field


class CinderAction(Action):
    """A free-text reply. Batch and pins want a digit. Pair wants left or right."""

    text: str = Field(default="", description="The reply for this step")


class CinderObservation(Observation):
    """What the agent sees. The batch number is not included."""

    prompt: str = Field(default="")
    task: str = Field(default="batch", description="batch, pins, or pair")
    image_base64: str = Field(default="", description="PNG tablet, base64")
    legal_actions: list[str] = Field(default_factory=list)


class CinderState(State):
    task: str = "batch"
