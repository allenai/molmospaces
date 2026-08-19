#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from types import SimpleNamespace

from molmo_spaces.evaluation.molmoact2_server_policy import Molmoact2ServerPolicy
from molmo_spaces.evaluation.policy_server import WebsocketPolicyServer

log = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a shared MolmoAct2 policy over websocket.")
    parser.add_argument("--checkpoint_path", required=True, help="Path to MolmoAct2 checkpoint.")
    parser.add_argument("--mm_olmo_path", required=True, help="Path to the molmoact2 repo root.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, required=True, help="Bind port.")
    parser.add_argument("--device", default="cuda", help="Torch device for the served model.")
    parser.add_argument("--exo_camera_key", default="", help="Preferred exocentric camera key.")
    parser.add_argument(
        "--seq_len", type=int, default=None, help="Optional sequence length override."
    )
    parser.add_argument("--num_steps", type=int, default=None, help="Optional flow-step override.")
    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=None,
        help="Optional action-chunk length override.",
    )
    parser.add_argument(
        "--action_mode",
        default="continuous",
        choices=("continuous", "discrete"),
        help="MolmoAct2 action mode.",
    )
    parser.add_argument(
        "--discrete_action_tokenizer",
        default=None,
        help="Discrete tokenizer path/name when action_mode=discrete.",
    )
    parser.add_argument(
        "--discrete_generation_max_steps",
        type=int,
        default=128,
        help="Maximum discrete token generation length.",
    )
    parser.add_argument("--style", default="", help="Optional style override.")
    parser.add_argument("--norm_tag", default="", help="Optional normalization tag override.")
    parser.add_argument(
        "--grasping_type",
        default="binary",
        choices=("binary", "continuous"),
        help="How to decode the gripper output.",
    )
    parser.add_argument(
        "--grasping_threshold",
        type=float,
        default=0.5,
        help="Threshold for binary gripper decoding.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose MolmoAct2 logging.",
    )
    return parser


def build_policy(args: argparse.Namespace) -> Molmoact2ServerPolicy:
    chunk_size = args.n_action_steps if args.n_action_steps is not None else 8
    policy_config = SimpleNamespace(
        checkpoint_path=str(Path(args.checkpoint_path).expanduser()),
        mm_olmo_path=str(Path(args.mm_olmo_path).expanduser()),
        device=args.device,
        exo_camera_key=str(args.exo_camera_key or ""),
        seq_len=args.seq_len,
        num_steps=args.num_steps,
        n_action_steps=args.n_action_steps,
        action_mode=args.action_mode,
        discrete_action_tokenizer=args.discrete_action_tokenizer,
        discrete_generation_max_steps=int(args.discrete_generation_max_steps),
        style=str(args.style or ""),
        norm_tag=str(args.norm_tag or ""),
        verbose=bool(args.verbose),
        grasping_type=args.grasping_type,
        grasping_threshold=float(args.grasping_threshold),
        chunk_size=int(chunk_size),
        # BasePolicy.__init__ reads policy_config.force_enable_depth unconditionally.
        # This stub stands in for BasePolicyConfig but does not inherit it, so the
        # field has to be supplied explicitly; False matches BasePolicyConfig's default
        # and keeps the depth/camera_config branch (which this stub also lacks) unused.
        force_enable_depth=False,
    )
    exp_config = SimpleNamespace(policy_config=policy_config, task_type="pick")
    return Molmoact2ServerPolicy(exp_config, task_type="pick")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = build_arg_parser().parse_args()

    policy = build_policy(args)
    model_name = Path(args.checkpoint_path).name or "molmoact2"
    metadata = {
        "policy_name": "molmoact2",
        "checkpoint_path": str(Path(args.checkpoint_path).expanduser()),
        "device": args.device,
        "style": args.style,
        "norm_tag": args.norm_tag,
        "action_mode": args.action_mode,
        "n_action_steps": args.n_action_steps,
    }
    log.info(
        "Starting MolmoAct2 websocket server on %s:%s for checkpoint %s",
        args.host,
        args.port,
        args.checkpoint_path,
    )
    log.info(
        "Effective MolmoAct2 inference config: action_mode=%s style=%s norm_tag=%s "
        "n_action_steps=%s exo_camera_key=%s",
        args.action_mode,
        args.style or "<checkpoint default>",
        args.norm_tag or "<checkpoint default>",
        args.n_action_steps if args.n_action_steps is not None else "<checkpoint default>",
        args.exo_camera_key or "<auto>",
    )
    server = WebsocketPolicyServer(
        policies=policy,
        model_name=model_name,
        host=args.host,
        port=args.port,
        metadata=metadata,
        max_concurrency=1024,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
