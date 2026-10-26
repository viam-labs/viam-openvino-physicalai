"""Viam custom training script: Viam sequence dataset -> LeRobot -> physicalai ACT -> OpenVINO.

Viam runs `python3 -m model.training` with the sequence-dataset file arguments
plus custom `--key=value` arguments. Everything except tmp/ in
--model_output_directory is published as the model version.
"""

from __future__ import annotations

import argparse
from pathlib import Path

POLICIES = ("act",)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse Viam's arguments strictly: an unknown or misspelled custom arg exits 2."""
    # allow_abbrev=False: otherwise a truncated typo like --max_epoch silently matches --max_epochs.
    parser = argparse.ArgumentParser(prog="model.training", allow_abbrev=False)
    # Passed by Viam for sequence datasets.
    parser.add_argument("--binary_data_file", type=Path, required=True)
    parser.add_argument("--tabular_data_file", type=Path, required=True)
    parser.add_argument("--sequences_file", type=Path, required=True)
    parser.add_argument("--model_output_directory", type=Path, required=True)
    # Custom arguments.
    parser.add_argument("--arm", required=True, help="follower arm component; source of the state")
    parser.add_argument(
        "--cameras", type=str.split, required=True,
        help="space-separated camera components; the first is the frame clock",
    )
    parser.add_argument(
        "--action_source", default="next_state",
        help="next_state, or an arm component (e.g. a teleop leader) whose JointPositions is the action",
    )
    parser.add_argument("--policy", choices=POLICIES, default="act")
    parser.add_argument("--fps", type=int, default=10, help="must match the capture frequency_hz")
    parser.add_argument("--image_size", type=int, default=None, help="downscale longest image side")
    parser.add_argument("--vcodec", default="auto", help="libsvtav1 if hardware encoding fails")
    parser.add_argument("--task", default=None, help="instruction for sequences without a cmd: tag")
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    # ACT's library defaults; at 10 fps, 100 steps is ~10 s of open-loop actions.
    parser.add_argument("--chunk_size", type=int, default=100, help="actions predicted per inference")
    parser.add_argument(
        "--n_action_steps", type=int, default=None, help="actions executed per inference (default: chunk_size)"
    )
    args = parser.parse_args(argv)
    if not args.cameras:
        parser.error("--cameras must name at least one camera component")
    if args.n_action_steps is None:
        args.n_action_steps = args.chunk_size
    if args.action_source == args.arm:
        parser.error("--action_source must differ from --arm (use next_state for the follower's next pose)")
    if not 1 <= args.n_action_steps <= args.chunk_size:
        parser.error("--n_action_steps must be between 1 and --chunk_size")
    return args


def main(argv: list[str] | None = None) -> None:
    parse_args(argv)


if __name__ == "__main__":
    main()
