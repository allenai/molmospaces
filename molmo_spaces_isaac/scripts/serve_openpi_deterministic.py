#!/usr/bin/env python3
"""Serve an OpenPI checkpoint with reproducible RNG resets for parity debugging.

This is a diagnostic wrapper around the OpenPI websocket protocol. The stock
OpenPI server advances a JAX sampling RNG on every inference and does not expose
an episode reset over the websocket client. For MolmoSpaces/Arena parity checks,
resetting the RNG when a new client connects makes paired runs easier to compare.

Run this from an environment where the forked OpenPI repo is importable, e.g.:

    cd /path/to/openpi-omarrayyann
    uv run /path/to/molmo_spaces_isaac/scripts/serve_openpi_deterministic.py \
      --port 8081 \
      --policy-config pi05_droid_jointpos \
      --policy-dir /path/to/checkpoints/pi05_droid_jointpos \
      --rng-seed 0
"""

from __future__ import annotations

import argparse
import asyncio
import http
import logging
import os
import socket
import time
import traceback
from typing import Any


log = logging.getLogger(__name__)


def _reset_policy_rng(policy: Any, seed: int) -> bool:
    """Reset OpenPI JAX policy RNG if the wrapped policy exposes ``_rng``."""
    target = getattr(policy, "_policy", policy)
    if not hasattr(target, "_rng"):
        return False
    import jax

    target._rng = jax.random.key(int(seed))
    return True


def _create_policy(args: argparse.Namespace):
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    sample_kwargs = {"temperature": float(args.temperature)}
    return _policy_config.create_trained_policy(
        _config.get_config(args.policy_config),
        args.policy_dir,
        default_prompt=args.default_prompt,
        sample_kwargs=sample_kwargs,
    )


class DeterministicWebsocketPolicyServer:
    """Small websocket server compatible with ``openpi_client``."""

    def __init__(
        self,
        policy,
        *,
        host: str,
        port: int,
        metadata: dict[str, Any],
        rng_seed: int | None,
        rng_seed_step: int,
        reset_each_infer: bool,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = int(port)
        self._metadata = metadata
        self._rng_seed = rng_seed
        self._rng_seed_step = int(rng_seed_step)
        self._reset_each_infer = reset_each_infer
        self._connection_count = 0

    def serve_forever(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        import websockets.asyncio.server as ws_server

        async with ws_server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket) -> None:
        from openpi_client import msgpack_numpy
        import websockets
        import websockets.frames

        log.info("Connection from %s opened", websocket.remote_address)
        connection_index = self._connection_count
        self._connection_count += 1
        connection_seed = (
            None
            if self._rng_seed is None
            else int(self._rng_seed + connection_index * self._rng_seed_step)
        )
        if connection_seed is not None:
            did_reset = _reset_policy_rng(self._policy, connection_seed)
            log.info("Reset policy RNG on connection to seed=%s: %s", connection_seed, did_reset)

        packer = msgpack_numpy.Packer()
        metadata = dict(self._metadata)
        metadata["rng_seed_used"] = connection_seed
        metadata["rng_connection_index"] = connection_index
        await websocket.send(packer.pack(metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())
                if self._reset_each_infer and connection_seed is not None:
                    _reset_policy_rng(self._policy, connection_seed)

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                if isinstance(action, tuple):
                    action = action[0]
                action["server_timing"] = {"infer_ms": infer_time * 1000}
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time
            except websockets.ConnectionClosed:
                log.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection, request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--policy-config", default="pi05_droid_jointpos")
    parser.add_argument("--policy-dir", required=True)
    parser.add_argument("--default-prompt", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument(
        "--rng-seed-step",
        type=int,
        default=0,
        help="Add this value per websocket connection. Use 1 to sweep seeds without restarting.",
    )
    parser.add_argument(
        "--no-reset-on-connect",
        action="store_true",
        help="Keep OpenPI's normal advancing RNG sequence after initial policy creation.",
    )
    parser.add_argument(
        "--reset-each-infer",
        action="store_true",
        help="Reset to --rng-seed before every inference, not only per client connection.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, force=True)
    args = _parse_args()
    policy = _create_policy(args)
    metadata = dict(getattr(policy, "metadata", {}) or {})
    metadata["policy_name"] = os.path.basename(str(args.policy_dir))
    metadata["rng_seed"] = None if args.no_reset_on_connect else int(args.rng_seed)
    metadata["rng_seed_step"] = int(args.rng_seed_step)
    metadata["rng_reset"] = (
        "none"
        if args.no_reset_on_connect
        else "each_infer"
        if args.reset_each_infer
        else "per_connection"
    )

    hostname = socket.gethostname()
    log.info("Creating deterministic OpenPI server (host=%s, ip=%s)", hostname, socket.gethostbyname(hostname))
    server = DeterministicWebsocketPolicyServer(
        policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
        rng_seed=None if args.no_reset_on_connect else int(args.rng_seed),
        rng_seed_step=int(args.rng_seed_step),
        reset_each_infer=bool(args.reset_each_infer),
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
