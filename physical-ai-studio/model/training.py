"""Viam custom training script: Viam sequence dataset -> LeRobot -> physicalai ACT -> OpenVINO.

Viam runs `python3 -m model.training` with the sequence-dataset file arguments
plus custom `--key=value` arguments. Everything except tmp/ in
--model_output_directory is published as the model version.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

POLICIES = ("act",)
# Names the converter's load_export() requires, in Viam's argument order below.
PARQUET_FILES = ("sequences.parquet", "tabular_data.parquet", "binary_data.parquet")


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


def resolve_export_dir(args: argparse.Namespace, work: Path) -> Path:
    """Return a directory holding the three Parquet files under the converter's names.

    A local `viam dataset export` already has them together, next to binary_data/,
    and its relative image paths only resolve against that directory, so it is
    used as-is. Otherwise (a cloud job) the files are symlinked into work/export,
    which is only safe because cloud image paths are absolute (/gcs/...).
    """
    given = (args.sequences_file, args.tabular_data_file, args.binary_data_file)
    files = dict(zip(PARQUET_FILES, given))
    parents = {path.resolve().parent for path in files.values()}
    if len(parents) == 1 and all(path.name == name for name, path in files.items()):
        return parents.pop()

    image_paths = pq.read_table(args.binary_data_file, columns=["path"]).column("path").to_pylist()
    if any(not path.startswith("/") for path in image_paths):
        raise SystemExit(
            "binary_data paths are relative, so the three Parquet files must sit together in the "
            f"export directory, next to binary_data/, named {', '.join(PARQUET_FILES)}"
        )
    stage = work / "export"
    stage.mkdir()
    for name, path in files.items():
        (stage / name).symlink_to(path.resolve())
    return stage


def check_components(args: argparse.Namespace, export_dir: Path) -> None:
    """Exit early when a named component never appears in the export (most likely a typo).

    Without this, the converter skips every sequence and fails with a vaguer error.
    """
    cameras = set(
        pq.read_table(export_dir / "binary_data.parquet", columns=["component_name"])
        .column("component_name").to_pylist()
    )
    readings = {
        (row["component_name"], row["method_name"])
        for row in pq.read_table(
            export_dir / "tabular_data.parquet", columns=["component_name", "method_name"]
        ).to_pylist()
    }
    arms = [args.arm] if args.action_source == "next_state" else [args.arm, args.action_source]
    missing = [c for c in args.cameras if c not in cameras]
    missing += [a for a in arms if (a, "JointPositions") not in readings]
    if missing:
        present = sorted({f"{c} (GetImages)" for c in cameras} | {f"{c} ({m})" for c, m in readings})
        raise SystemExit(
            f"Components not found in the export: {', '.join(missing)}. Present: {', '.join(present)}"
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="physicalai-") as tmp:
        work = Path(tmp)
        export_dir = resolve_export_dir(args, work)
        check_components(args, export_dir)


if __name__ == "__main__":
    main()
