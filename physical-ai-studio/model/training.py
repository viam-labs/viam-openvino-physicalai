"""Viam custom training script: Viam sequence dataset -> LeRobot -> physicalai ACT -> OpenVINO.

Viam runs `python3 -m model.training` with the sequence-dataset file arguments
plus custom `--key=value` arguments. Everything except tmp/ in
--model_output_directory is published as the model version.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import tempfile
import time
from importlib import metadata
from pathlib import Path

import pyarrow.parquet as pq

logger = logging.getLogger("physical-ai-studio")

# LeRobot dataset name; the converter and the datamodule must agree on it.
DATASET_REPO_ID = "viam/dataset"

POLICIES = ("act",)
# Names the converter's load_export() requires, in Viam's argument order below.
PARQUET_FILES = ("sequences.parquet", "tabular_data.parquet", "binary_data.parquet")


def unquote(value: str) -> str:
    """Viam passes `--args=cameras="'a b'"` values through with the shell quotes still on."""
    return value.strip("'\"")


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
        "--cameras", type=lambda s: unquote(s).split(), required=True,
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
    parser.add_argument("--task", type=unquote, default=None, help="instruction for sequences without a cmd: tag")
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


def convert_dataset(args: argparse.Namespace, export_dir: Path, dataset_root: Path):
    """Convert the Viam export into a LeRobot v3 dataset at dataset_root (must not exist)."""
    # Imported here: lerobot must not load before logging is configured (converter's streaming.py).
    from viam_sequence_to_lerobot.convert import ConversionConfig, convert

    return convert(
        ConversionConfig(
            export_dir=export_dir,
            output_root=dataset_root,
            repo_id=DATASET_REPO_ID,
            task=args.task,
            camera_components=tuple(args.cameras),
            arm_component=args.arm,
            action_arm=None if args.action_source == "next_state" else args.action_source,
            fps=args.fps,
            image_size=args.image_size,
            vcodec=args.vcodec,
        )
    )


def train(args: argparse.Namespace, dataset_root: Path, ckpt_dir: Path):
    """Train the policy; return it and the device type it trained on."""
    from physicalai.data import LeRobotDataModule
    from physicalai.policies import ACT
    from physicalai.train import Trainer

    # Docker's default 64 MB /dev/shm kills DataLoader workers with a bus error, and Viam
    # sets the job's --shm-size, not us. num_workers=0 also disables persistent_workers.
    shm = shutil.disk_usage("/dev/shm").total
    workers = {}
    if shm < 1 << 30:
        logger.warning("/dev/shm is only %d MiB: loading data in the main process (num_workers=0)", shm >> 20)
        workers["num_workers"] = 0
    datamodule = LeRobotDataModule(
        repo_id=DATASET_REPO_ID, root=dataset_root, train_batch_size=args.batch_size, **workers
    )
    policy = ACT(chunk_size=args.chunk_size, n_action_steps=args.n_action_steps)
    # devices=1: on a multi-GPU host, "auto" picks DDP, whose launcher re-runs this whole
    # script per rank (re-converting, and every rank exporting to out).
    trainer = Trainer(
        max_epochs=args.max_epochs, accelerator="auto", devices=1, default_root_dir=ckpt_dir
    )
    device = trainer.strategy.root_device.type
    if device == "cpu":
        logger.warning("Training on CPU: expected for local runs, but in a cloud job no GPU was found")
    else:
        logger.info("Training on %s", trainer.strategy.root_device)
    trainer.fit(model=policy, datamodule=datamodule)
    return policy, device


def model_inputs(policy, cameras: list[str]) -> tuple[list[dict], dict]:
    """Map each camera, and the state, to the exported model's input name and shape."""
    from physicalai.inference.constants import IMAGES, STATE

    schema = {feature.name: list(feature.shape) for feature in policy.inputs_schema}
    image_inputs = [name for name in schema if name == IMAGES or name.startswith(f"{IMAGES}.")]
    entries = []
    for camera in cameras:
        if len(cameras) == 1 and image_inputs == [IMAGES]:
            name = IMAGES
        else:
            safe = camera.replace("-", "_").replace(".", "_")  # converter's camera_feature_key()
            matches = [n for n in image_inputs if n.endswith(f".{safe}")]
            if len(matches) != 1:
                raise RuntimeError(f"Cannot map camera {camera!r} to one exported input in {image_inputs}")
            name = matches[0]
        # The graph normalizes internally; it takes frames as LeRobot decodes them.
        entries.append({
            "component": camera, "input": name, "shape": schema[name], "method": "GetImages",
            "dtype": "float32", "layout": "CHW", "color": "RGB", "range": [0, 1],
        })
    return entries, {"input": STATE, "shape": schema[STATE]}


def package_version(name: str) -> str:
    """Installed version, with the commit for VCS installs (the converter)."""
    dist = metadata.distribution(name)
    direct_url = json.loads(dist.read_text("direct_url.json") or "{}")
    commit = direct_url.get("vcs_info", {}).get("commit_id")
    return f"{dist.version}+{commit[:12]}" if commit else dist.version


def write_manifest(
    args, out: Path, summary, policy, device: str, wall_time_s: float, versions: dict[str, str]
) -> dict:
    """Write viam.json: what the inference module needs beyond the export's manifest.json."""
    cameras, state = model_inputs(policy, args.cameras)
    dim = state["shape"][0]  # the converter guarantees action dim == state dim in joints mode
    manifest = {
        "schema_version": 1,
        "policy": args.policy,
        "fps": args.fps,
        "cameras": cameras,
        "image_size": args.image_size,
        "state": {**state, "component": args.arm, "method": "JointPositions", "units": "degrees", "dim": dim},
        "action": {
            "source": args.action_source,
            "apply": "MoveToJointPositions",
            "component": args.arm,  # always the follower: that's what gets commanded at inference
            "units": "degrees",
            "dim": dim,
        },
        "tasks": sorted(summary.tasks_written),
        "training": {
            "args": {"max_epochs": args.max_epochs, "batch_size": args.batch_size,
                     "chunk_size": args.chunk_size, "n_action_steps": args.n_action_steps,
                     "vcodec": args.vcodec, "task": args.task},
            "episodes": summary.episodes_written,
            "frames": summary.frames_written,
            "skipped_sequences": len(summary.skipped),
            "device": device,
            "wall_time_s": round(wall_time_s),
        },
        "versions": versions,
    }
    (out / "viam.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    start = time.monotonic()
    import torch

    # Logged first so a cloud job's device is visible before the long conversion step.
    if torch.cuda.is_available():
        logger.info("GPU available: %s", torch.cuda.get_device_name(0))
    else:
        logger.warning("No GPU available: training will run on CPU")
    # Up front: a missing package must not throw away a finished training run.
    versions = {
        name: package_version(name)
        for name in ("physicalai-train", "physicalai", "lerobot", "viam-sequence-to-lerobot")
    }
    out = args.model_output_directory
    out.mkdir(parents=True, exist_ok=True)
    # Dataset and checkpoints go to local disk: the output dir may be a slow /gcs/ mount.
    with tempfile.TemporaryDirectory(prefix="physicalai-") as tmp:
        work = Path(tmp)
        export_dir = resolve_export_dir(args, work)
        check_components(args, export_dir)
        try:
            summary = convert_dataset(args, export_dir, work / "dataset")
        except ValueError as exc:
            # load_export silently drops images whose files are missing, so a misplaced
            # binary_data/ only shows up here, as every sequence lacking camera frames.
            if not str(exc).startswith("No convertible episodes"):
                raise
            raise SystemExit(
                f"{exc}\nIf every sequence reports 'no frames from camera', check that "
                "binary_data/ sits next to the Parquet files."
            ) from exc
        policy, device = train(args, work / "dataset", work / "ckpt")
        policy.export(out, backend="openvino")
    manifest = write_manifest(args, out, summary, policy, device, time.monotonic() - start, versions)
    logger.info(
        "Done: %d episodes, %d frames, %d sequences skipped, device %s, %ds",
        summary.episodes_written,
        summary.frames_written,
        len(summary.skipped),
        device,
        manifest["training"]["wall_time_s"],
    )


if __name__ == "__main__":
    main()
