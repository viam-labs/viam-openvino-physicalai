"""Viam custom training script: Viam sequence dataset -> LeRobot -> physicalai ACT, SmolVLA or Pi0.5 -> OpenVINO or ONNX.

Viam runs `python3 -m model.training` with the sequence-dataset file arguments
plus custom `--key=value` arguments. Everything except tmp/ in
--model_output_directory is published as the model version.
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
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

POLICIES = ("act", "smolvla", "pi05")
# Vision-language-action policies: fine-tuned from a checkpoint that fixes their chunk size
# (50 for both bases), conditioned on task text, exported to OpenVINO only.
VLAS = ("smolvla", "pi05")
PRETRAINED = {"smolvla": "lerobot/smolvla_base", "pi05": "lerobot/pi05_base"}
# Pi0.5 is ~3.6B parameters: a full fine-tune with Adam does not fit one 40 GB A100, LoRA does.
LORA_RANK = 32
# ACT's library default; at 10 fps, 100 steps is ~10 s of open-loop actions.
ACT_CHUNK_SIZE = 100
# physicalai ExportBackend values offered here; ACT supports both.
EXPORT_FORMATS = ("openvino", "onnx")
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
    parser.add_argument(
        "--arm", type=lambda s: unquote(s).split(), required=True,
        help="space-separated follower arm components; their joints (each followed by its gripper) form the state",
    )
    parser.add_argument(
        "--grippers", type=lambda s: unquote(s).split(), default=[],
        help="space-separated gripper components captured with DoCommand, one per arm in the same order",
    )
    parser.add_argument(
        "--gripper_key", default="position", help="numeric field of each gripper's captured DoCommand response"
    )
    parser.add_argument(
        "--cameras", type=lambda s: unquote(s).split(), required=True,
        help="space-separated camera components; the first is the frame clock",
    )
    parser.add_argument(
        "--action_source", default="next_state",
        help="next_state, or an arm component (e.g. a teleop leader) whose JointPositions is the action",
    )
    parser.add_argument("--policy", choices=POLICIES, default="act")
    parser.add_argument("--export_format", choices=EXPORT_FORMATS, default="openvino")
    parser.add_argument(
        "--pretrained", default=None,
        help="smolvla/pi05: checkpoint to fine-tune (Hugging Face id or local directory; "
             "default " + ", ".join(f"{p}: {c}" for p, c in PRETRAINED.items()) + ")",
    )
    parser.add_argument(
        "--num_steps", type=int, default=10, help="smolvla/pi05: flow-matching denoising steps per inference"
    )
    parser.add_argument(
        "--lora_rank", type=int, default=None,
        help=f"pi05 only: LoRA adapter rank (default {LORA_RANK}); 0 fine-tunes every weight",
    )
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
    parser.add_argument(
        "--chunk_size", type=int, default=None,
        help=f"act only: actions predicted per inference (default {ACT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--n_action_steps", type=int, default=None,
        help="actions executed per inference (default: the chunk size)",
    )
    args = parser.parse_args(argv)
    if not args.cameras:
        parser.error("--cameras must name at least one camera component")
    if not args.arm:
        parser.error("--arm must name at least one arm component")
    if args.grippers and len(args.grippers) != len(args.arm):
        parser.error(f"--grippers needs one gripper per arm: got {len(args.grippers)} for {len(args.arm)} arms")
    if args.action_source in args.arm:
        parser.error("--action_source must differ from --arm (use next_state for the follower's next pose)")
    if args.action_source != "next_state" and (len(args.arm) > 1 or args.grippers):
        parser.error("--action_source: leader actions are single-arm only; use next_state")
    if args.lora_rank is not None and args.policy != "pi05":
        parser.error("--lora_rank applies to pi05 only")
    if args.policy in VLAS:
        if args.chunk_size is not None:
            parser.error(
                f"--chunk_size is fixed by the --pretrained checkpoint for {args.policy}; set --n_action_steps instead"
            )
        if args.export_format != "openvino":
            parser.error(f"{args.policy} exports to openvino only")
        if args.num_steps < 1:
            parser.error("--num_steps must be at least 1")
        # The upper bound is the checkpoint's chunk size, checked in check_checkpoint().
        if args.n_action_steps is not None and args.n_action_steps < 1:
            parser.error("--n_action_steps must be at least 1")
        if args.pretrained is None:
            args.pretrained = PRETRAINED[args.policy]
        if args.policy == "pi05":
            if args.lora_rank is None:
                args.lora_rank = LORA_RANK
            if args.lora_rank < 0:
                parser.error("--lora_rank must be 0 (full fine-tune) or more")
    else:
        if args.chunk_size is None:
            args.chunk_size = ACT_CHUNK_SIZE
        if args.n_action_steps is None:
            args.n_action_steps = args.chunk_size
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
    expected = [(arm, "JointPositions") for arm in args.arm] + [(g, "DoCommand") for g in args.grippers]
    if args.action_source != "next_state":
        expected.append((args.action_source, "JointPositions"))
    missing = [c for c in args.cameras if c not in cameras]
    missing += [component for component, method in expected if (component, method) not in readings]
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
            arm_components=tuple(args.arm),
            grippers=tuple(args.grippers),
            gripper_key=args.gripper_key,
            action_arm=None if args.action_source == "next_state" else args.action_source,
            fps=args.fps,
            image_size=args.image_size,
            vcodec=args.vcodec,
            workers=args.workers,
        )
    )


def feature_key(camera: str) -> str:
    """The converter's camera_feature_key(): a camera's LeRobot image feature suffix."""
    return camera.replace("-", "_").replace(".", "_")


def checkpoint_config(pretrained: str) -> dict:
    """A SmolVLA or Pi0.5 checkpoint's parsed config.json, from a local directory or the Hub."""
    try:
        local = Path(pretrained)
        if local.is_dir():
            config = local / "config.json"
        else:
            from huggingface_hub import hf_hub_download

            config = Path(hf_hub_download(pretrained, "config.json"))
        return json.loads(config.read_text())
    except Exception as exc:
        raise SystemExit(f"Cannot read config.json for --pretrained={pretrained}: {exc}") from exc


def check_checkpoint(args: argparse.Namespace) -> None:
    """Fail before the slow dataset conversion if the VLA checkpoint or --n_action_steps is unusable."""
    config = checkpoint_config(args.pretrained)
    kind = config.get("type")
    if kind is not None and kind != args.policy:
        raise SystemExit(f"--pretrained={args.pretrained} is a {kind} checkpoint, not {args.policy}")
    chunk = config.get("chunk_size")
    if not isinstance(chunk, int):
        raise SystemExit(f"--pretrained={args.pretrained} config.json has no integer chunk_size: not a {args.policy} checkpoint?")
    if args.n_action_steps is not None and args.n_action_steps > chunk:
        raise SystemExit(f"--n_action_steps={args.n_action_steps} exceeds the {args.pretrained} chunk size ({chunk})")


def build_policy(args: argparse.Namespace):
    """Construct the untrained policy named by --policy."""
    if args.policy == "act":
        from physicalai.policies import ACT

        # ACT letterboxes frames to image_size; its default (384) would pad 224px frames back up.
        size = {"image_size": (args.image_size, args.image_size)} if args.image_size else {}
        return ACT(chunk_size=args.chunk_size, n_action_steps=args.n_action_steps, **size)

    if args.policy == "pi05":
        from physicalai.policies import Pi05

        # Pi05 takes no camera arguments: fine-tuning swaps the checkpoint's stats (3 cameras,
        # 32-dim state) for the dataset's, and the export names inputs as model_inputs() expects.
        # Unlike SmolVLA it applies the constructor's n_action_steps over the checkpoint's, and its
        # default (50) would win if omitted; Pi05Config rejects a value above the chunk size.
        chunk = checkpoint_config(args.pretrained)["chunk_size"]
        lora = {"lora_enabled": True, "lora_rank": args.lora_rank} if args.lora_rank else {}
        return Pi05(
            pretrained_name_or_path=args.pretrained,
            num_inference_steps=args.num_steps,
            n_action_steps=args.n_action_steps or chunk,
            **lora,
        )

    from physicalai.policies import SmolVLA

    keys = [feature_key(camera) for camera in args.cameras]
    features = checkpoint_config(args.pretrained).get("input_features", {})
    slots = sum(1 for feature in features.values() if feature.get("type") == "VISUAL")
    policy = SmolVLA(
        pretrained_name_or_path=args.pretrained,
        num_steps=args.num_steps,
        # --cameras fill the checkpoint's camera slots in order; unused slots are masked.
        num_cameras=max(len(keys), slots),
        # A single camera is exported as plain "images" and needs no map.
        image_key_reorder_map={key: i for i, key in enumerate(keys)} if len(keys) > 1 else None,
    )
    # A pretrained SmolVLA ignores the constructor's chunk_size and n_action_steps: both come
    # from the checkpoint. Its config is frozen, so n_action_steps is set by replacing it; the
    # export then trims each chunk to it. Only policy.config (and so the export) gets the new
    # n_action_steps; the base class's own select_action queue keeps the checkpoint default,
    # which nothing here uses. The upper bound was checked up front by check_checkpoint().
    chunk = policy.config.chunk_size
    steps = args.n_action_steps or chunk
    policy.config = dataclasses.replace(policy.config, n_action_steps=steps)
    return policy


def progress_callback(every_s: float = 60):
    """A callback logging speed, time spent waiting for data and loss: cloud logs show no progress bar.

    Every `every_s` seconds it logs the window since the last line; at each epoch's end, the epoch's
    mean loss. Times are wall-clock between hooks, so they include the GPU sync the progress bar forces.
    """
    from lightning.pytorch.callbacks import Callback

    class Progress(Callback):
        def on_train_epoch_start(self, trainer, pl_module):
            self.batch_end = None  # the first batch's wait is worker startup, not loading
            self.epoch_loss, self.epoch_steps, self.epoch_start = 0.0, 0, time.monotonic()
            self.window(self.epoch_start)

        def window(self, now):
            self.start, self.steps, self.wait, self.loss = now, 0, 0.0, 0.0

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            if self.batch_end is not None:
                self.wait += time.monotonic() - self.batch_end

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            loss = (outputs["loss"] if isinstance(outputs, dict) else outputs).detach()
            # Summed as tensors: reading .item() every step would add a GPU sync.
            self.loss, self.epoch_loss = self.loss + loss, self.epoch_loss + loss
            self.steps += 1
            self.epoch_steps += 1
            self.batch_end = now = time.monotonic()
            if now - self.start < every_s:
                return
            per_step = (now - self.start) / self.steps
            left = (trainer.estimated_stepping_batches - trainer.global_step) * per_step
            logger.info(
                "Epoch %d batch %d/%d: %.2f s/step, %.0f%% waiting for data, loss %.4f, ~%.0f min left",
                trainer.current_epoch, batch_idx + 1, trainer.num_training_batches, per_step,
                100 * self.wait / (now - self.start), float(self.loss) / self.steps, left / 60,
            )
            self.window(now)

        def on_train_epoch_end(self, trainer, pl_module):
            if self.epoch_steps:
                logger.info(
                    "Epoch %d done: mean loss %.4f over %d steps in %.0f min", trainer.current_epoch,
                    float(self.epoch_loss) / self.epoch_steps, self.epoch_steps,
                    (time.monotonic() - self.epoch_start) / 60,
                )

    return Progress()


def train(args: argparse.Namespace, dataset_root: Path, ckpt_dir: Path):
    """Train the policy; return it and the device type it trained on."""
    from physicalai.data import LeRobotDataModule
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
    policy = build_policy(args)
    # devices=1: on a multi-GPU host, "auto" picks DDP, whose launcher re-runs this whole
    # script per rank (re-converting, and every rank exporting to out).
    trainer = Trainer(
        max_epochs=args.max_epochs, accelerator="auto", devices=1, default_root_dir=ckpt_dir,
        callbacks=[progress_callback()],
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
            matches = [n for n in image_inputs if n.endswith(f".{feature_key(camera)}")]
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
    train_args = {
        "max_epochs": args.max_epochs, "batch_size": args.batch_size,
        # The trained policy's own values: VLA chunk sizes come from their checkpoint.
        "chunk_size": policy.config.chunk_size, "n_action_steps": policy.config.n_action_steps,
        "vcodec": args.vcodec, "task": args.task, "export_format": args.export_format,
    }
    if args.policy in VLAS:
        train_args |= {"pretrained": args.pretrained, "num_steps": args.num_steps}
    if args.policy == "pi05":
        train_args["lora_rank"] = args.lora_rank
    manifest = {
        "schema_version": 1,
        "policy": args.policy,
        "format": args.export_format,
        "fps": args.fps,
        "cameras": cameras,
        "image_size": args.image_size,
        "state": {**state, "component": args.arm[0], "method": "JointPositions", "units": "degrees", "dim": dim},
        "action": {
            "source": args.action_source,
            "apply": "MoveToJointPositions",
            "component": args.arm[0],  # always the follower: that's what gets commanded at inference
            "units": "degrees",
            "dim": dim,
        },
        "tasks": sorted(summary.tasks_written),
        "training": {
            "args": train_args,
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
    # Stacks to stderr on a fatal signal. Not dump_traceback_later: its timed dumps of running
    # threads segfaulted Pi0.5 training twice on Viam cloud.
    faulthandler.enable()
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
        if args.policy in VLAS:
            check_checkpoint(args)
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
        if args.policy == "act" and len(summary.tasks_written) > 1:
            logger.warning(
                "ACT ignores task text, so these %d tasks will blend into one behavior: %s. "
                "Use --policy=smolvla or --policy=pi05 to condition on the task.",
                len(summary.tasks_written), "; ".join(sorted(summary.tasks_written)),
            )
        policy, device = train(args, work / "dataset", work / "ckpt")
        policy.export(out, backend=args.export_format)
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
