"""Serve Cinder over the OpenEnv HTTP and WebSocket API."""

from __future__ import annotations

try:
    from openenv.core.env_server.http_server import create_app
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        'OpenEnv is not installed. Install this project with: pip install -e ".[openenv]"'
    ) from exc

from cinder.environment import CinderEnvironment
from cinder.models import CinderAction, CinderObservation

app = create_app(
    CinderEnvironment,
    CinderAction,
    CinderObservation,
    env_name="cinder",
    max_concurrent_envs=8,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Cinder OpenEnv server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(host=args.host, port=args.port)
