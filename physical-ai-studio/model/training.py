"""Viam custom training script: Viam sequence dataset -> LeRobot -> physicalai ACT -> OpenVINO.

Viam runs `python3 -m model.training` with the sequence-dataset file arguments
plus custom `--key=value` arguments. Everything except tmp/ in
--model_output_directory is published as the model version.
"""

from __future__ import annotations

import argparse
import ctypes
import faulthandler
import glob
import io
import json
import logging
import os
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
# --vcodec auto tries these in order and uses the first that can actually encode here:
# lerobot's own auto picks NVENC whenever FFmpeg lists it, even where opening it fails.
VCODEC_CANDIDATES = ("h264_nvenc", "h264", "libsvtav1")
# Cloud Storage transfer-manager threads for copying /gcs/ images to local disk. A process pool
# hung on Viam, likely from forking after av, torch and google-auth had started threads.
DOWNLOAD_WORKERS = 24
DOWNLOAD_BATCH = 1000


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
    parser.add_argument(
        "--vcodec", default="auto", help="auto: first of " + ", ".join(VCODEC_CANDIDATES) + " that works here"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="conversion processes (default: half the cores with --image_size, else the converter's choice)",
    )
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
    used as-is. Otherwise the files are staged in work/export, which is only safe
    because cloud image paths are absolute (/gcs/...). Images under /gcs/ are first
    copied to local disk, even when a cloud job hands over the three files together:
    the converter checks and reads each one in turn, a network round trip apiece
    through the mount, which took hours for 70k images.
    """
    given = (args.sequences_file, args.tabular_data_file, args.binary_data_file)
    files = dict(zip(PARQUET_FILES, given))
    parents = {path.resolve().parent for path in files.values()}
    together = len(parents) == 1 and all(path.name == name for name, path in files.items())
    logger.info("Reading image paths from %s", args.binary_data_file)
    image_paths = pq.read_table(args.binary_data_file, columns=["path"]).column("path").to_pylist()
    if together and not any(p.startswith("/gcs/") for p in image_paths):
        return parents.pop()
    if any(not path.startswith("/") for path in image_paths):
        raise SystemExit(
            "binary_data paths are relative, so the three Parquet files must sit together in the "
            f"export directory, next to binary_data/, named {', '.join(PARQUET_FILES)}"
        )
    stage = work / "export"
    stage.mkdir()
    for name, path in files.items():
        if name == "binary_data.parquet" and any(p.startswith("/gcs/") for p in image_paths):
            table = pq.read_table(path)
            local = download_images(image_paths, work / "images")
            pq.write_table(table.set_column(table.schema.get_field_index("path"), "path", [local]), stage / name)
        else:
            (stage / name).symlink_to(path.resolve())
    return stage


def download_images(paths: list[str], dest: Path) -> list[str]:
    """Copy /gcs/<bucket>/<object> images to dest with the Cloud Storage transfer manager."""
    from google.cloud import storage
    from google.cloud.storage import transfer_manager

    try:
        client = storage.Client()
    except OSError:  # no project in the environment; downloads don't need one
        client = storage.Client(project="unused")
    logger.info("Copying %d images from /gcs as %s (project %s)",
                len(paths), type(client._credentials).__name__, client.project)
    dest.mkdir()
    local = [str(dest / f"{i:07d}{Path(p).suffix}") for i, p in enumerate(paths)]
    # One download without retries first: an auth or permission problem otherwise hides
    # behind per-file retries until a whole batch finishes, which can take an hour.
    _, _, bucket, name = paths[0].split("/", 3)
    probe = time.monotonic()
    client.bucket(bucket).blob(name).download_to_filename(local[0], timeout=30, retry=None)
    logger.info("Copied the first image in %.1fs", time.monotonic() - probe)
    buckets = {}
    start = time.monotonic()
    for k in range(0, len(paths), DOWNLOAD_BATCH):
        pairs = []
        for src, dst in zip(paths[k:k + DOWNLOAD_BATCH], local[k:k + DOWNLOAD_BATCH]):
            _, _, bucket, name = src.split("/", 3)
            if bucket not in buckets:
                buckets[bucket] = client.bucket(bucket)
            pairs.append((buckets[bucket].blob(name), dst))
        transfer_manager.download_many(
            pairs, max_workers=DOWNLOAD_WORKERS, worker_type=transfer_manager.THREAD, raise_exception=True
        )
        logger.info("Copied %d/%d images from /gcs in %ds", k + len(pairs), len(paths), time.monotonic() - start)
    return local


def choose_vcodec(requested: str) -> str:
    """Resolve auto to the first candidate codec that can encode a few frames here."""
    if requested != "auto":
        return requested
    import av
    import numpy as np

    for codec in VCODEC_CANDIDATES:
        try:
            with av.open(io.BytesIO(), mode="w", format="mp4") as out:
                stream = out.add_stream(codec, rate=10)
                stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
                frame = av.VideoFrame.from_ndarray(np.zeros((64, 64, 3), np.uint8), format="rgb24")
                for _ in range(3):
                    out.mux(stream.encode(frame))
                out.mux(stream.encode())
        except Exception as exc:  # any failure means "not usable here"
            logger.info("Video codec %s unusable here: %s", codec, exc)
            continue
        logger.info("Video codec: %s", codec)
        return codec
    raise SystemExit("No usable video codec among " + ", ".join(VCODEC_CANDIDATES))


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
            workers=args.workers,
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
    # ACT letterboxes frames to image_size; its default (384) would pad 224px frames back up.
    size = {"image_size": (args.image_size, args.image_size)} if args.image_size else {}
    policy = ACT(chunk_size=args.chunk_size, n_action_steps=args.n_action_steps, **size)
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
    # Every thread's stack to stderr every 10 minutes, so a cloud job that goes quiet shows where.
    faulthandler.dump_traceback_later(600, repeat=True)
    start = time.monotonic()
    import torch

    # Logged first so a cloud job's device is visible before the long conversion step.
    if torch.cuda.is_available():
        logger.info("GPU available: %s", torch.cuda.get_device_name(0))
    else:
        logger.warning("No GPU available: training will run on CPU")
        try:
            ctypes.CDLL("libcuda.so.1")
            libcuda = "loads"
        except OSError as exc:
            libcuda = str(exc)
        logger.warning(
            "GPU diagnostics: torch CUDA %s, /dev/nvidia* %s, /usr/local/nvidia %s, LD_LIBRARY_PATH %s, libcuda.so.1 %s",
            torch.version.cuda,
            sorted(glob.glob("/dev/nvidia*")),
            sorted(str(p) for p in Path("/usr/local/nvidia").glob("*/*"))[:20],
            os.environ.get("LD_LIBRARY_PATH"),
            libcuda,
        )
    args.vcodec = choose_vcodec(args.vcodec)
    if args.workers is None and args.image_size:
        # Downscaled frames make one encoder cheap; decoding the full-size JPEGs is the bottleneck.
        args.workers = max(1, (os.cpu_count() or 2) // 2)
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
