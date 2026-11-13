"""End-to-end and failure tests for model.training, run inside the training container."""

from __future__ import annotations

import json
import os
import subprocess
import sys
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
LEFT, RIGHT, LEFT_GRIPPER, RIGHT_GRIPPER = "left-arm", "right-arm", "left-gripper", "right-gripper"
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
                 tasks=("pick the cube", "pick the cube"), bimanual=False) -> Path:
    """Write a two-sequence Viam sequence export (follower + leader arm, 10 Hz) and return its dir;
    bimanual=True writes two arms and two DoCommand grippers instead of follower + leader."""
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
            arms = ((LEFT, N_JOINTS, 0.0), (RIGHT, N_JOINTS, 10.0)) if bimanual else (
                (ARM, N_JOINTS, 0.0), (LEADER, leader_joints, 0.5))
            for arm, n, offset in arms:
                payload = json.dumps({"positions": {"values": [tick + offset + j for j in range(n)]}})
                _row(tabular, seq, ts + timedelta(milliseconds=2), arm, "JointPositions", payload=payload)
            if bimanual:
                for gripper, offset in ((LEFT_GRIPPER, 0.0), (RIGHT_GRIPPER, 50.0)):
                    payload = json.dumps({"docommand_output": {"position": tick + offset}})
                    _row(tabular, seq, ts + timedelta(milliseconds=3), gripper, "DoCommand", payload=payload)
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
        ({"arm": f"{ARM} other-arm", "grippers": "g1"}, "--grippers needs one gripper per arm"),
        ({"arm": f"{ARM} other-arm", "action_source": LEADER}, "leader actions are single-arm only"),
        ({"grippers": "g1", "action_source": LEADER}, "leader actions are single-arm only"),
        ({"arm": f"other-arm {ARM}", "action_source": ARM}, "--action_source"),
        ({"export_format": "onnx", "weight_compression": "int8"}, "openvino export only"),
        ({"weight_compression": "int2"}, "invalid choice: 'int2'"),
        ({"precision": "8-bit"}, "invalid choice: '8-bit'"),
        ({"devices": 0}, "--devices"),
        ({"devices": -1}, "--devices"),
        ({"devices": "two"}, "--devices"),
    ],
)
def test_bad_custom_args_exit_2(tmp_path, capsys, bad, msg):
    export = build_export(tmp_path, [CAM])
    with pytest.raises(SystemExit) as exc:
        run(local_files(export), tmp_path / "out", **({"arm": ARM, "cameras": CAM} | bad))
    assert exc.value.code == 2
    assert msg in capsys.readouterr().err


def test_export_format_defaults_to_openvino(tmp_path):
    base = required_argv(tmp_path)
    assert parse_args(base).export_format == "openvino"
    assert parse_args([*base, "--export_format=onnx"]).export_format == "onnx"


def test_precision_and_compression_defaults(tmp_path):
    args = parse_args(required_argv(tmp_path))
    assert args.precision is None and args.weight_compression == "none"
    args = parse_args([*required_argv(tmp_path), "--precision=bf16-mixed", "--weight_compression=int4"])
    assert args.precision == "bf16-mixed" and args.weight_compression == "int4"


def test_compression_targets_the_model_not_the_tokenizer(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"model": {"artifacts": {"openvino": "pi05.xml"}}}))
    for name in ("pi05.xml", "tokenizer.xml"):
        (tmp_path / name).touch()
    assert training.openvino_model_path(tmp_path) == tmp_path / "pi05.xml"


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
        "components": [{"component": ARM, "offset": 0, "dim": N_JOINTS, "method": "JointPositions",
                        "apply": "MoveToJointPositions", "units": "degrees"}],
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
    summary = SimpleNamespace(tasks_written={"open the box"}, episodes_written=1, frames_written=2, skipped=[],
                              segments=((ARM, N_JOINTS),))
    out = tmp_path / "out"
    out.mkdir()
    manifest = training.write_manifest(args, out, summary, policy, "cuda", 1.0, {})
    assert manifest["action"]["components"] == [
        {"component": ARM, "offset": 0, "dim": N_JOINTS, "method": "JointPositions",
         "apply": "MoveToJointPositions", "units": "degrees"}]
    train_args = manifest["training"]["args"]
    assert manifest["policy"] == "pi05"
    assert (train_args["pretrained"], train_args["num_steps"], train_args["lora_rank"]) == ("lerobot/pi05_base", 10, 8)
    assert (train_args["chunk_size"], train_args["n_action_steps"]) == (50, 10)


def test_manifest_components_bimanual(tmp_path):
    files = [f"--{arg}=x" for arg in ARG_FOR_FILE] + [f"--model_output_directory={tmp_path}", f"--cameras={CAM}"]
    args = parse_args(files + ["--arm=left-arm right-arm", "--grippers=left-gripper right-gripper"])
    dim = 2 * (N_JOINTS + 1)
    feature = lambda name, shape: SimpleNamespace(name=name, shape=shape)  # noqa: E731
    policy = SimpleNamespace(
        inputs_schema=[feature("images", (3, 8, 8)), feature("state", (dim,))],
        config=SimpleNamespace(chunk_size=10, n_action_steps=5),
    )
    segments = (("left-arm", N_JOINTS), ("left-gripper", 1), ("right-arm", N_JOINTS), ("right-gripper", 1))
    summary = SimpleNamespace(tasks_written={"t"}, episodes_written=1, frames_written=2, skipped=[], segments=segments)
    out = tmp_path / "out"
    out.mkdir()
    manifest = training.write_manifest(args, out, summary, policy, "cpu", 1.0, {})

    arm = {"method": "JointPositions", "apply": "MoveToJointPositions", "units": "degrees"}
    grip = {"method": "DoCommand", "key": "position", "apply": "DoCommand", "units": "as captured"}
    assert manifest["action"]["components"] == [
        {"component": "left-arm", "offset": 0, "dim": N_JOINTS, **arm},
        {"component": "left-gripper", "offset": N_JOINTS, "dim": 1, **grip},
        {"component": "right-arm", "offset": N_JOINTS + 1, "dim": N_JOINTS, **arm},
        {"component": "right-gripper", "offset": 2 * N_JOINTS + 1, "dim": 1, **grip},
    ]
    assert all("apply" not in c for c in manifest["state"]["components"])
    assert [c["component"] for c in manifest["state"]["components"]] == [name for name, _ in segments]
    assert manifest["action"]["component"] == manifest["state"]["component"] == "left-arm"
    assert manifest["action"]["dim"] == manifest["state"]["dim"] == dim


def test_progress_callback_logs_speed_wait_and_loss(caplog):
    import torch

    cb = training.progress_callback(every_s=0)
    trainer = SimpleNamespace(current_epoch=0, num_training_batches=2, estimated_stepping_batches=2, global_step=0,
                              is_global_zero=True)
    caplog.set_level("INFO", logger="physical-ai-studio")
    cb.on_train_epoch_start(trainer, None)
    for i, loss in enumerate([1.0, 3.0]):
        cb.on_train_batch_start(trainer, None, None, i)
        trainer.global_step += 1
        cb.on_train_batch_end(trainer, None, {"loss": torch.tensor(loss)}, None, i)
    cb.on_train_epoch_end(trainer, None)
    lines = [r.getMessage() for r in caplog.records]
    assert "Epoch 0 batch 1/2:" in lines[0] and "waiting for data, loss 1.0000" in lines[0]
    assert "loss 3.0000, ~0 min left" in lines[1]
    assert all(int(line.split("%")[0].rsplit(" ", 1)[1]) <= 100 for line in lines[:2])
    assert lines[2].startswith("Epoch 0 done: mean loss 2.0000 over 2 steps")


def test_progress_callback_is_silent_off_rank_zero(caplog):
    import torch

    cb = training.progress_callback(every_s=0)
    trainer = SimpleNamespace(current_epoch=0, num_training_batches=1, estimated_stepping_batches=1,
                              global_step=0, is_global_zero=False)
    caplog.set_level("INFO", logger="physical-ai-studio")
    cb.on_train_epoch_start(trainer, None)
    cb.on_train_batch_start(trainer, None, None, 0)
    trainer.global_step += 1
    cb.on_train_batch_end(trainer, None, {"loss": torch.tensor(1.0)}, None, 0)
    cb.on_train_epoch_end(trainer, None)
    assert caplog.records == []


def test_arm_and_grippers_parse_as_lists(tmp_path):
    args = parse_args(required_argv(tmp_path))
    assert (args.arm, args.grippers, args.gripper_key) == ([ARM], [], "position")
    files = [f"--{arg}=x" for arg in ARG_FOR_FILE] + [f"--model_output_directory={tmp_path}", f"--cameras={CAM}"]
    args = parse_args(files + ["--arm='left-arm right-arm'", "--grippers='lg rg'", "--gripper_key=width"])
    assert (args.arm, args.grippers, args.gripper_key) == (["left-arm", "right-arm"], ["lg", "rg"], "width")


def test_missing_gripper_lists_present_components(tmp_path):
    export = build_export(tmp_path, [CAM])
    with pytest.raises(SystemExit) as exc:
        run(local_files(export), tmp_path / "out", arm=ARM, cameras=CAM, grippers="follower-gripper")
    message = str(exc.value.code)
    assert "follower-gripper" in message and "follower-arm (JointPositions)" in message


def test_bimanual_trains_exports_and_runs_inference(tmp_path):
    from physicalai.inference import InferenceModel

    export = build_export(tmp_path, [CAM, WRIST], bimanual=True)
    out = tmp_path / "out"
    run(local_files(export), out, arm=f"{LEFT} {RIGHT}", grippers=f"{LEFT_GRIPPER} {RIGHT_GRIPPER}",
        cameras=f"{CAM} {WRIST}", max_epochs=1, batch_size=2, vcodec="libsvtav1", chunk_size=10, n_action_steps=5)

    manifest = json.loads((out / "viam.json").read_text())
    dim = 2 * (N_JOINTS + 1)
    assert manifest["state"]["dim"] == manifest["action"]["dim"] == dim
    assert [(c["component"], c["offset"], c["dim"]) for c in manifest["action"]["components"]] == [
        (LEFT, 0, N_JOINTS), (LEFT_GRIPPER, N_JOINTS, 1), (RIGHT, N_JOINTS + 1, N_JOINTS),
        (RIGHT_GRIPPER, 2 * N_JOINTS + 1, 1),
    ]
    assert manifest["training"]["frames"] == 2 * (TICKS - 1)

    model = InferenceModel(out)
    obs = {manifest["state"]["input"]: np.zeros((1, *manifest["state"]["shape"]), np.float32)}
    for cam in manifest["cameras"]:
        obs[cam["input"]] = np.zeros((1, *cam["shape"]), np.float32)
    assert model.select_action(obs).shape == (dim,)


def test_devices_defaults_to_auto(tmp_path):
    assert parse_args(required_argv(tmp_path)).devices == "auto"
    assert parse_args([*required_argv(tmp_path), "--devices=2"]).devices == 2
    assert parse_args([*required_argv(tmp_path), "--devices='auto'"]).devices == "auto"


@pytest.mark.parametrize("requested, gpus, expected", [(2, 0, 2), ("auto", 0, 1), ("auto", 2, 2), (1, 2, 1)])
def test_resolve_devices_returns_an_int(monkeypatch, requested, gpus, expected):
    import torch

    monkeypatch.setattr(torch.cuda, "device_count", lambda: gpus)
    assert training.resolve_devices(requested) == expected


def test_extra_rank_without_work_dir_exits_before_staging(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.delenv("PHYSICALAI_WORK", raising=False)
    monkeypatch.setattr(training, "resolve_export_dir", lambda *a: pytest.fail("rank 1 staged the export"))
    with pytest.raises(SystemExit, match="PHYSICALAI_WORK is unset"):
        main(required_argv(tmp_path))


def test_extra_rank_only_trains_on_rank_zeros_work_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("PHYSICALAI_WORK", str(tmp_path / "work"))
    monkeypatch.setattr(training, "resolve_export_dir", lambda *a: pytest.fail("rank 1 staged the export"))
    calls = []
    monkeypatch.setattr(training, "train", lambda args, dataset, ckpt: calls.append((dataset, ckpt)))
    main(required_argv(tmp_path))
    assert calls == [(tmp_path / "work" / "dataset", tmp_path / "work" / "ckpt")]
    assert not (tmp_path / "out" / "viam.json").exists()


def test_two_process_ddp_trains_and_exports_once(tmp_path):
    # A subprocess, not main(): under pytest, Lightning would relaunch pytest itself for rank 1.
    export = build_export(tmp_path, [CAM])
    out = tmp_path / "out"
    argv = [f"--{arg}={path}" for arg, path in local_files(export).items()] + [
        f"--model_output_directory={out}", f"--arm={ARM}", f"--cameras={CAM}", "--devices=2",
        "--max_epochs=1", "--batch_size=2", "--vcodec=libsvtav1", "--chunk_size=10", "--n_action_steps=5",
    ]
    env = {k: v for k, v in os.environ.items() if k not in ("LOCAL_RANK", "PHYSICALAI_WORK")}
    result = subprocess.run([sys.executable, "-m", "model.training", *argv],
                            capture_output=True, text=True, timeout=1800, env=env)
    log = result.stdout + result.stderr
    assert result.returncode == 0, log[-4000:]
    assert log.count("Training on 2 × cpu, effective batch 4 (batch_size 2 per device)") == 1
    assert log.count("Rank 1: training only") == 1
    assert log.count("sequences skipped, device") == 1  # main()'s "Done:" line: rank 1 never exports
    assert (out / "act.xml").is_file()
    assert json.loads((out / "viam.json").read_text())["training"]["frames"] == 2 * (TICKS - 1)


def test_gripper_open_rescales_raw_grippers_to_unit_open_zero(tmp_path):
    """--gripper_open turns a raw reading (840 = open) into the vla controller's 0..1, 0 = open."""
    export = tmp_path / "export"
    export.mkdir()
    tabular = pa.table({
        "component_name": ["gripper", "gripper", "gripper", "arm"],
        "method_name": ["DoCommand", "DoCommand", "DoCommand", "JointPositions"],
        "payload": [json.dumps({"docommand_output": {"pos": v}}) for v in (840, 0, 900)] + ['{"joints": [1]}'],
    })
    pq.write_table(tabular, export / "tabular_data.parquet")
    pq.write_table(pa.table({"path": ["binary_data/a.jpg"]}), export / "binary_data.parquet")
    pq.write_table(pa.table({"sequence_id": ["s"]}), export / "sequences.parquet")
    args = parse_args([
        "--binary_data_file=x", "--tabular_data_file=x", "--sequences_file=x", "--model_output_directory=x",
        f"--arm={ARM}", "--grippers=gripper", "--gripper_key=pos", f"--cameras={CAM}", "--gripper_open=840",
    ])
    out = training.normalize_grippers(args, export, tmp_path)
    payloads = pq.read_table(out / "tabular_data.parquet").column("payload").to_pylist()
    assert [json.loads(p)["docommand_output"]["pos"] for p in payloads[:3]] == [0.0, 1.0, 0.0]
    assert payloads[3] == '{"joints": [1]}'
    assert pq.read_table(out / "binary_data.parquet").column("path").to_pylist() == [
        str((export / "binary_data/a.jpg").resolve())
    ]
    assert training.manifest_components(args, [(ARM, 6), ("gripper", 1)])[1]["units"] == "0..1, 0 = open"
