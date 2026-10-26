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
