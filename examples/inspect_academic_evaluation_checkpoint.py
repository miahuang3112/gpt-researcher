from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {path}"
        )

    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(payload, dict):
        raise ValueError(
            "Checkpoint must contain a JSON object."
        )

    return payload


def print_checkpoint_summary(
    payload: dict[str, Any],
) -> None:
    results = payload.get("dimension_results", [])

    if not isinstance(results, list):
        raise ValueError(
            "dimension_results must be a list."
        )

    print(f"Paper ID: {payload.get('paper_id')}")
    print(f"Completed: {payload.get('completed', False)}")
    print(f"Saved dimensions: {len(results)}/9")
    print()

    if not results:
        print("No dimension results have been saved.")
        return

    for index, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            print(f"{index}. Invalid result entry")
            continue

        verification = item.get("verification") or {}
        metrics = verification.get("metrics") or {}

        print(
            f"{index}. {item.get('dimension')}\n"
            f"   status: {item.get('status')}\n"
            f"   score: {item.get('score')}\n"
            f"   verification: "
            f"{verification.get('status')}\n"
            f"   claim support rate: "
            f"{metrics.get('claim_support_rate')}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect an academic evaluation checkpoint."
        )
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the checkpoint JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(
        args.checkpoint
    ).expanduser().resolve()

    payload = load_checkpoint(checkpoint_path)
    print_checkpoint_summary(payload)


if __name__ == "__main__":
    main()