"""End-to-end and failure tests for model.training, run inside the training container."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

import model.training as training
from model.training import VCODEC_CANDIDATES, choose_vcodec, main, parse_args, resolve_export_dir

CAM, WRIST, ARM, LEADER = "overhead-cam", "wrist-cam", "follower-arm", "leader-arm"
N_JOINTS, TICKS = 6, 15
BASE = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
ARG_FOR_FILE = {
    "sequences_file": "sequences.parquet",
    "tabular_data_file": "tabular_data.parquet",
    "binary_data_file": "binary_data.parquet",
}


def _row(table: dict, seq: str, ts: datetime, component: str, method: str, **data) -> None:
    table["sequence_id"].append(seq)
    table["timestamp"].append(ts)
    table["part_id"].append("part-1")
    table["component_name"].append(component)
    table["method_name"].append(method)
    for key, value in data.items():
        table[key].append(value)


def build_export(root: Path, cameras: list[str], *, leader_joints=N_JOINTS, absolute=False,
                 tasks=("pick the cube", "pick the cube")) -> Path:
    """Write a two-sequence Viam sequence export (follower + leader arm, 10 Hz) and return its dir."""
    export = root / "export"
    seqs = ["seq-a", "seq-b"]
    starts = [BASE, BASE + timedelta(minutes=1)]
    sequences = {
        "sequence_id": seqs,
        "tags": [[f"cmd:{task}"] for task in tasks],
        "start_at": starts,
        "end_at": [s + timedelta(seconds=2) for s in starts],
    }
    binary: dict = defaultdict(list)
    tabular: dict = defaultdict(list)
    rng = np.random.default_rng(0)
    for seq, start in zip(seqs, starts):
        for tick in range(TICKS):
            ts = start + timedelta(seconds=0.1 * tick)
            for cam in cameras:
                rel = f"binary_data/{seq}/{cam}/{tick:04d}.jpeg"
                (export / rel).parent.mkdir(parents=True, exist_ok=True)
                pixels = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(export / rel)
                _row(binary, seq, ts, cam, "GetImages", path=str(export / rel) if absolute else rel)
            for arm, n, offset in ((ARM, N_JOINTS, 0.0), (LEADER, leader_joints, 0.5)):
                payload = json.dumps({"positions": {"values": [tick + offset + j for j in range(n)]}})
                _row(tabular, seq, ts + timedelta(milliseconds=2), arm, "JointPositions", payload=payload)
    pq.write_table(pa.table(sequences), export / "sequences.parquet")
    pq.write_table(pa.table(dict(tabular)), export / "tabular_data.parquet")
    pq.write_table(pa.table(dict(binary)), export / "binary_data.parquet")
    return export


def local_files(export: Path) -> dict[str, Path]:
    """Files as a local `viam dataset export` lays them out: together, canonical names."""
    return {arg: export / name for arg, name in ARG_FOR_FILE.items()}


def cloud_files(export: Path) -> dict[str, Path]:
    """Files as a cloud job may pass them: separate dirs, arbitrary names."""
    files = {}
    for arg, name in ARG_FOR_FILE.items():
        dest = export.parent / arg / "data.parquet"
        dest.parent.mkdir()
        (export / name).rename(dest)
        files[arg] = dest
    return files


def run(files: dict[str, Path], out: Path, **custom) -> None:
    argv = [f"--{arg}={path}" for arg, path in files.items()]
    argv.append(f"--model_output_directory={out}")
    argv += [f"--{key}={value}" for key, value in custom.items()]
    main(argv)


def required_argv(tmp_path: Path) -> list[str]:
    """Viam's file args plus the required custom args; parse_args doesn't open the files."""
    argv = [f"--{arg}={tmp_path / name}" for arg, name in ARG_FOR_FILE.items()]
    return argv + [f"--model_output_directory={tmp_path / 'out'}", f"--arm={ARM}", f"--cameras={CAM}"]


@pytest.mark.parametrize(
    "bad, msg",
    [
        ({"max_epoch": 1}, "--max_epoch"),
        ({"policy": "diffusion"}, "invalid choice: 'diffusion'"),
        ({"export_format": "torch"}, "invalid choice: 'torch'"),
        ({"policy": "smolvla", "chunk_size": 20}, "--chunk_size is fixed"),
        ({"policy": "smolvla", "export_format": "onnx"}, "openvino only"),
        ({"policy": "smolvla", "n_action_steps": 0}, "--n_action_steps must be at least 1"),
        ({"policy": "smolvla", "num_steps": 0}, "--num_steps must be at least 1"),
        ({"policy": "pi05", "chunk_size": 20}, "--chunk_size is fixed"),
        ({"policy": "pi05", "export_format": "onnx"}, "openvino only"),
        ({"policy": "pi05", "num_steps": 0}, "--num_steps must be at least 1"),
        ({"policy": "pi05", "lora_rank": -1}, "--lora_rank must be 0"),
        ({"lora_rank": 8}, "--lora_rank applies to pi05 only"),
        ({"policy": "smolvla", "lora_rank": 8}, "--lora_rank applies to pi05 only"),
        ({"chunk_size": 5, "n_action_steps": 10}, "--n_action_steps"),
        ({"action_source": ARM}, "--action_source"),
    ],
)
def test_bad_custom_args_exit_2(tmp_path, capsys, bad, msg):
    export = build_export(tmp_path, [CAM])
    with pytest.raises(SystemExit) as exc:
        run(local_files(export), tmp_path / "out", arm=ARM, cameras=CAM, **bad)
    assert exc.value.code == 2
    assert msg in capsys.readouterr().err


def test_export_format_defaults_to_openvino(tmp_path):
    base = required_argv(tmp_path)
    assert parse_args(base).export_format == "openvino"
    assert parse_args([*base, "--export_format=onnx"]).export_format == "onnx"


def test_chunk_defaults_per_policy(tmp_path):
    act = parse_args(required_argv(tmp_path))
    assert (act.chunk_size, act.n_action_steps) == (100, 100)
    assert (act.pretrained, act.lora_rank) == (None, None)
    smol = parse_args([*required_argv(tmp_path), "--policy=smolvla"])
    # VLA chunk sizes come from the checkpoint, resolved in build_policy().
    assert (smol.chunk_size, smol.n_action_steps) == (None, None)
    assert (smol.pretrained, smol.num_steps, smol.lora_rank) == ("lerobot/smolvla_base", 10, None)
    pi = parse_args([*required_argv(tmp_path), "--policy=pi05"])
    assert (pi.chunk_size, pi.n_action_steps) == (None, None)
    assert (pi.pretrained, pi.num_steps, pi.lora_rank) == ("lerobot/pi05_base", 10, 32)
    full = parse_args([*required_argv(tmp_path), "--policy=pi05", "--lora_rank=0", "--pretrained=me/pi05"])
    assert (full.pretrained, full.lora_rank) == ("me/pi05", 0)


def test_misspelled_camera_lists_present_components(tmp_path):
    export = build_export(tmp_path, [CAM])
    with pytest.raises(SystemExit) as exc:
        run(local_files(export), tmp_path / "out", arm=ARM, cameras="overhead-cma")
    message = str(exc.value.code)
    assert "overhead-cma" in message
    assert "overhead-cam (GetImages)" in message and "leader-arm (JointPositions)" in message


def test_missing_action_source_component_exits(tmp_path):
    export = build_export(tmp_path, [CAM])
    with pytest.raises(SystemExit, match="leader-amr"):
        run(local_files(export), tmp_path / "out", arm=ARM, cameras=CAM, action_source="leader-amr")


def test_staging_with_relative_image_paths_exits(tmp_path):
    export = build_export(tmp_path, [CAM], absolute=False)
    with pytest.raises(SystemExit, match="next to binary_data/"):
        run(cloud_files(export), tmp_path / "out", arm=ARM, cameras=CAM)


@pytest.mark.parametrize(
    "layout, export_format", [("local", "openvino"), ("cloud", "openvino"), ("local", "onnx")]
)
def test_trains_exports_and_runs_inference(tmp_path, layout, export_format):
    # local: one camera, relative paths, next-state actions.
    # cloud: two cameras, absolute paths, staged inputs, leader-arm actions.
    from physicalai.inference import InferenceModel

    cloud = layout == "cloud"
    cameras = [CAM, WRIST] if cloud else [CAM]
    action_source = LEADER if cloud else "next_state"
    export = build_export(tmp_path, cameras, absolute=cloud)
    files = cloud_files(export) if cloud else local_files(export)
    out = tmp_path / "out"

    run(files, out, arm=ARM, cameras=" ".join(cameras), action_source=action_source,
        max_epochs=1, batch_size=2, vcodec="libsvtav1", chunk_size=10, n_action_steps=5,
        export_format=export_format)

    artifact = "act.onnx" if export_format == "onnx" else "act.xml"
    assert (out / artifact).is_file() and (out / "manifest.json").is_file()
    manifest = json.loads((out / "viam.json").read_text())
    assert manifest["schema_version"] == 1 and manifest["policy"] == "act"
    assert manifest["format"] == export_format
    assert manifest["training"]["args"]["export_format"] == export_format
    assert [c["component"] for c in manifest["cameras"]] == cameras
    for cam in manifest["cameras"]:
        assert {k: cam[k] for k in ("dtype", "layout", "color", "range")} == {
            "dtype": "float32", "layout": "CHW", "color": "RGB", "range": [0, 1],
        }
    assert manifest["state"]["component"] == ARM and manifest["state"]["dim"] == N_JOINTS
    assert manifest["action"] == {
        "source": action_source, "apply": "MoveToJointPositions", "component": ARM,
        "units": "degrees", "dim": N_JOINTS,
    }
    assert manifest["tasks"] == ["pick the cube"]
    assert manifest["training"]["args"]["chunk_size"] == 10
    assert manifest["training"]["args"]["n_action_steps"] == 5
    assert "pretrained" not in manifest["training"]["args"]
    frames_per_episode = TICKS if cloud else TICKS - 1  # next_state consumes the last tick
    assert manifest["training"]["episodes"] == 2
    assert manifest["training"]["frames"] == 2 * frames_per_episode

    # The manifest's input names and shapes must be exactly what the exported model takes.
    model = InferenceModel(out)
    obs = {manifest["state"]["input"]: np.zeros((1, *manifest["state"]["shape"]), np.float32)}
    for cam in manifest["cameras"]:
        obs[cam["input"]] = np.zeros((1, *cam["shape"]), np.float32)
    assert model.select_action(obs).shape == (N_JOINTS,)


@pytest.mark.skipif(
    not os.environ.get("PHYSICALAI_TEST_SMOLVLA"),
    reason="set PHYSICALAI_TEST_SMOLVLA=1: downloads lerobot/smolvla_base (~1 GB) and fine-tunes on CPU for minutes",
)
def test_smolvla_trains_exports_and_runs_inference(tmp_path):
    from physicalai.inference import InferenceModel

    export = build_export(tmp_path, [CAM, WRIST])
    out = tmp_path / "out"
    run(local_files(export), out, arm=ARM, cameras=f"{CAM} {WRIST}", policy="smolvla",
        max_epochs=1, batch_size=8, vcodec="libsvtav1", n_action_steps=10)

    assert (out / "smolvla.xml").is_file() and (out / "tokenizer.xml").is_file()
    manifest = json.loads((out / "viam.json").read_text())
    assert (manifest["policy"], manifest["format"]) == ("smolvla", "openvino")
    assert [c["input"] for c in manifest["cameras"]] == ["images.overhead_cam", "images.wrist_cam"]
    train_args = manifest["training"]["args"]
    assert (train_args["chunk_size"], train_args["n_action_steps"]) == (50, 10)
    assert (train_args["pretrained"], train_args["num_steps"]) == ("lerobot/smolvla_base", 10)
    assert manifest["tasks"] == ["pick the cube"]

    # The tokenizer is part of the export: the model takes the task as a string.
    model = InferenceModel(out)
    batch = {cam["input"]: np.zeros((1, *cam["shape"]), np.float32) for cam in manifest["cameras"]}
    batch[manifest["state"]["input"]] = np.zeros((1, *manifest["state"]["shape"]), np.float32)
    batch["task"] = ["pick the cube"]
    assert np.asarray(model.predict_action_chunk(batch)).shape == (10, N_JOINTS)


@pytest.mark.skipif(
    not os.environ.get("PHYSICALAI_TEST_PI05"),
    reason="set PHYSICALAI_TEST_PI05=1: downloads lerobot/pi05_base (~15 GB), needs the image's cached "
           "PaliGemma tokenizer and a large GPU or ~60 GB of RAM",
)
def test_pi05_trains_exports_and_runs_inference(tmp_path):
    from physicalai.inference import InferenceModel

    export = build_export(tmp_path, [CAM, WRIST])
    out = tmp_path / "out"
    run(local_files(export), out, arm=ARM, cameras=f"{CAM} {WRIST}", policy="pi05",
        max_epochs=1, batch_size=2, vcodec="libsvtav1", n_action_steps=10)

    assert (out / "pi05.xml").is_file() and (out / "tokenizer.xml").is_file()
    manifest = json.loads((out / "viam.json").read_text())
    assert (manifest["policy"], manifest["format"]) == ("pi05", "openvino")
    assert [c["input"] for c in manifest["cameras"]] == ["images.overhead_cam", "images.wrist_cam"]
    train_args = manifest["training"]["args"]
    assert (train_args["chunk_size"], train_args["n_action_steps"], train_args["lora_rank"]) == (50, 10, 32)
    assert (train_args["pretrained"], train_args["num_steps"]) == ("lerobot/pi05_base", 10)
    assert manifest["tasks"] == ["pick the cube"]

    model = InferenceModel(out)
    batch = {cam["input"]: np.zeros((1, *cam["shape"]), np.float32) for cam in manifest["cameras"]}
    batch[manifest["state"]["input"]] = np.zeros((1, *manifest["state"]["shape"]), np.float32)
    batch["task"] = ["pick the cube"]
    assert np.asarray(model.predict_action_chunk(batch)).shape == (10, N_JOINTS)


def test_check_checkpoint_rejects_oversized_n_action_steps(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text(json.dumps({"chunk_size": 50, "input_features": {
        "observation.state": {"type": "STATE"},
        "observation.images.camera1": {"type": "VISUAL"},
        "observation.images.camera2": {"type": "VISUAL"},
    }}))
    assert training.checkpoint_config(str(ckpt))["chunk_size"] == 50

    base = [*required_argv(tmp_path), "--policy=smolvla", f"--pretrained={ckpt}"]
    with pytest.raises(SystemExit, match="exceeds"):
        training.check_checkpoint(parse_args([*base, "--n_action_steps=60"]))
    assert training.check_checkpoint(parse_args([*base, "--n_action_steps=10"])) is None


def test_zero_episodes_hints_at_export_layout(tmp_path):
    export = build_export(tmp_path, [CAM])
    (export / "binary_data").rename(tmp_path / "misplaced")  # images no longer resolve
    with pytest.raises(SystemExit, match="binary_data/ sits next to"):
        run(local_files(export), tmp_path / "out", arm=ARM, cameras=CAM)


def test_leader_joint_mismatch_fails_before_training(tmp_path):
    export = build_export(tmp_path, [CAM], leader_joints=N_JOINTS + 1)
    with pytest.raises(ValueError, match=f"has {N_JOINTS + 1} joints"):
        run(local_files(export), tmp_path / "out", arm=ARM, cameras=CAM, action_source=LEADER)
    assert not (tmp_path / "out" / "viam.json").exists()


def test_viam_quoted_values_are_unquoted(tmp_path):
    files = [f"--{arg}=x" for arg in ARG_FOR_FILE] + [f"--model_output_directory={tmp_path}"]
    args = parse_args(files + [f"--arm={ARM}", f"--cameras='{CAM} {WRIST}'", "--task='open the box'"])
    assert args.cameras == [CAM, WRIST] and args.task == "open the box"


@pytest.mark.parametrize("together", [False, True])
def test_gcs_images_are_staged_with_local_paths(tmp_path, monkeypatch, together):
    export = build_export(tmp_path, [CAM], absolute=True)
    table = pq.read_table(export / "binary_data.parquet")
    gcs = [p.replace(str(export), "/gcs/bucket") for p in table.column("path").to_pylist()]
    pq.write_table(table.set_column(table.schema.get_field_index("path"), "path", [gcs]),
                   export / "binary_data.parquet")
    seen = []
    monkeypatch.setattr(training, "download_images", lambda paths, dest: seen.extend(paths) or
                        [str(dest / f"{i}.jpeg") for i in range(len(paths))])
    # A Viam cloud job hands over the three files together under the converter's names.
    files = {arg: export / name for arg, name in ARG_FOR_FILE.items()} if together else cloud_files(export)
    args = parse_args([f"--{a}={p}" for a, p in files.items()] +
                      [f"--model_output_directory={tmp_path / 'out'}", f"--arm={ARM}", f"--cameras={CAM}"])
    (tmp_path / "work").mkdir()
    stage = resolve_export_dir(args, tmp_path / "work")
    assert seen == gcs
    staged = pq.read_table(stage / "binary_data.parquet").column("path").to_pylist()
    assert staged[0] == str(tmp_path / "work" / "images" / "0.jpeg") and len(staged) == len(gcs)


def test_auto_vcodec_picks_a_codec_that_encodes():
    assert choose_vcodec("libsvtav1") == "libsvtav1"
    # No GPU in the test container, so NVENC fails to open and software H.264 wins.
    assert choose_vcodec("auto") == "h264"


@pytest.mark.parametrize("tasks, warns", [
    (("pick the cube", "pick the cube"), False),
    (("pick the cube", "open the box"), True),
])
def test_act_warns_when_tasks_differ(tmp_path, monkeypatch, caplog, tasks, warns):
    export = build_export(tmp_path, [CAM], tasks=tasks)

    def stop(*_args):
        raise SystemExit("stopped before training")

    monkeypatch.setattr(training, "train", stop)
    with caplog.at_level("WARNING", logger="physical-ai-studio"), pytest.raises(SystemExit, match="stopped"):
        run(local_files(export), tmp_path / "out", arm=ARM, cameras=CAM, vcodec="libsvtav1")
    assert ("ACT ignores task text" in caplog.text) == warns


def _checkpoint(tmp_path: Path, **config) -> Path:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text(json.dumps(config))
    return ckpt


def test_check_checkpoint_rejects_another_policys_checkpoint(tmp_path):
    ckpt = _checkpoint(tmp_path, type="smolvla", chunk_size=50)
    args = parse_args([*required_argv(tmp_path), "--policy=pi05", f"--pretrained={ckpt}"])
    with pytest.raises(SystemExit, match="is a smolvla checkpoint, not pi05"):
        training.check_checkpoint(args)


class FakePi05:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.mark.parametrize("extra, lora, steps", [
    ([], {"lora_enabled": True, "lora_rank": 32}, 50),
    (["--lora_rank=0", "--n_action_steps=10"], {}, 10),
])
def test_build_policy_pi05(tmp_path, monkeypatch, extra, lora, steps):
    import physicalai.policies

    monkeypatch.setattr(physicalai.policies, "Pi05", FakePi05)
    ckpt = _checkpoint(tmp_path, type="pi05", chunk_size=50)
    args = parse_args([*required_argv(tmp_path), "--policy=pi05", f"--pretrained={ckpt}", "--num_steps=5", *extra])
    policy = training.build_policy(args)
    # n_action_steps is always passed: Pi05's own default (50) would override the checkpoint's.
    assert policy.kwargs == {
        "pretrained_name_or_path": str(ckpt), "num_inference_steps": 5, "n_action_steps": steps, **lora,
    }


def test_manifest_records_pi05_training_args(tmp_path):
    args = parse_args([*required_argv(tmp_path), "--policy=pi05", "--lora_rank=8"])
    feature = lambda name, shape: SimpleNamespace(name=name, shape=shape)  # noqa: E731
    policy = SimpleNamespace(
        inputs_schema=[feature("images", (3, 8, 8)), feature("state", (N_JOINTS,)), feature("task", ())],
        config=SimpleNamespace(chunk_size=50, n_action_steps=10),
    )
    summary = SimpleNamespace(tasks_written={"open the box"}, episodes_written=1, frames_written=2, skipped=[])
    out = tmp_path / "out"
    out.mkdir()
    manifest = training.write_manifest(args, out, summary, policy, "cuda", 1.0, {})
    train_args = manifest["training"]["args"]
    assert manifest["policy"] == "pi05"
    assert (train_args["pretrained"], train_args["num_steps"], train_args["lora_rank"]) == ("lerobot/pi05_base", 10, 8)
    assert (train_args["chunk_size"], train_args["n_action_steps"]) == (50, 10)
