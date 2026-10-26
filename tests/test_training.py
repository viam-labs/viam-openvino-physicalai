"""End-to-end and failure tests for model.training, run inside the training container."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from model.training import main

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


def build_export(root: Path, cameras: list[str], *, leader_joints=N_JOINTS, absolute=False) -> Path:
    """Write a two-sequence Viam sequence export (follower + leader arm, 10 Hz) and return its dir."""
    export = root / "export"
    seqs = ["seq-a", "seq-b"]
    starts = [BASE, BASE + timedelta(minutes=1)]
    sequences = {
        "sequence_id": seqs,
        "tags": [["cmd:pick the cube"]] * 2,
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


@pytest.mark.parametrize(
    "bad, msg",
    [
        ({"max_epoch": 1}, "--max_epoch"),
        ({"policy": "pi05"}, "invalid choice: 'pi05'"),
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


@pytest.mark.parametrize("layout", ["local", "cloud"])
def test_trains_exports_and_runs_inference(tmp_path, layout):
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
        max_epochs=1, batch_size=2, vcodec="libsvtav1", chunk_size=10, n_action_steps=5)

    assert (out / "act.xml").is_file() and (out / "manifest.json").is_file()
    manifest = json.loads((out / "viam.json").read_text())
    assert manifest["schema_version"] == 1 and manifest["policy"] == "act"
    assert [c["component"] for c in manifest["cameras"]] == cameras
    assert manifest["state"]["component"] == ARM and manifest["state"]["dim"] == N_JOINTS
    assert manifest["action"] == {
        "source": action_source, "apply": "MoveToJointPositions", "component": ARM,
        "units": "degrees", "dim": N_JOINTS,
    }
    assert manifest["tasks"] == ["pick the cube"]
    assert manifest["training"]["args"]["chunk_size"] == 10
    assert manifest["training"]["args"]["n_action_steps"] == 5
    frames_per_episode = TICKS if cloud else TICKS - 1  # next_state consumes the last tick
    assert manifest["training"]["episodes"] == 2
    assert manifest["training"]["frames"] == 2 * frames_per_episode

    # The manifest's input names and shapes must be exactly what the exported model takes.
    model = InferenceModel(out)
    obs = {manifest["state"]["input"]: np.zeros((1, *manifest["state"]["shape"]), np.float32)}
    for cam in manifest["cameras"]:
        obs[cam["input"]] = np.zeros((1, *cam["shape"]), np.float32)
    assert model.select_action(obs).shape == (N_JOINTS,)


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
