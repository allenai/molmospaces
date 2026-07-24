"""Upload commercial_episodes.parquet to allenai/molmobot-data (repo root)."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

HF_REPO_ID = "allenai/molmobot-data"
HF_REPO_TYPE = "dataset"
DEFAULT_PATH_IN_REPO = "commercial_episodes.parquet"


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Upload commercial_episodes.parquet to the root of "
            f"{HF_REPO_ID} on Hugging Face."
        )
    )
    parser.add_argument(
        "--local-path",
        type=Path,
        default=here / "commercial_episodes.parquet",
        help="Local parquet file to upload (default: nc_filter/commercial_episodes.parquet).",
    )
    parser.add_argument(
        "--path-in-repo",
        default=DEFAULT_PATH_IN_REPO,
        help=f"Destination path in the HF repo (default: {DEFAULT_PATH_IN_REPO}).",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="HF branch or revision to commit to (default: main).",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload commercial_episodes.parquet",
        help="Commit message for the HF upload.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_path = args.local_path.resolve()
    if not local_path.is_file():
        raise FileNotFoundError(f"Local file not found: {local_path}")

    size_mb = local_path.stat().st_size / (1024 * 1024)
    print(
        f"Uploading {local_path} ({size_mb:.1f} MB) "
        f"to {HF_REPO_ID}:{args.path_in_repo} ({args.revision})"
    )

    api = HfApi()
    url = api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=args.path_in_repo,
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        revision=args.revision,
        commit_message=args.commit_message,
    )
    print(f"Done: {url}")


if __name__ == "__main__":
    main()
