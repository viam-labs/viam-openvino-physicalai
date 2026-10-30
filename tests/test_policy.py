import json

import numpy as np
import pytest

from src.policy import ConfigError, LoadError, PolicyConfig, read_package

CAMS = [("cam-a", "images.cam_a", [3, 8, 16]), ("cam-b", "images.cam_b", [3, 16, 8])]
N_STEPS, DIM = 4, 6


def write_package(path, **overrides):
    meta = {
        "schema_version": 1,
        "policy": "act",
        "fps": 10,
        "cameras": [{"component": c, "input": i, "shape": s} for c, i, s in CAMS],
        "state": {"input": "state", "shape": [DIM], "component": "arm", "dim": DIM},
        "action": {"component": "arm", "dim": DIM},
        "tasks": ["open the box"],
        "training": {"args": {"chunk_size": 10, "n_action_steps": N_STEPS}},
    } | overrides
    path.mkdir(parents=True, exist_ok=True)
    (path / "viam.json").write_text(json.dumps(meta))
    return path


def test_config_defaults_and_coercion():
    cfg = PolicyConfig.parse({"model_path": "/m", "warmup_inferences": 3.0})
    assert (cfg.model_path, cfg.device, cfg.warmup_inferences, cfg.load_timeout_s) == ("/m", "AUTO", 3, 600.0)


@pytest.mark.parametrize(
    "attrs, msg",
    [
        ({}, "model_path"),
        ({"model_path": ""}, "model_path"),
        ({"model_path": "/m", "device": "cuda"}, "device"),
        ({"model_path": "/m", "warmup_inferences": 1.5}, "warmup_inferences"),
        ({"model_path": "/m", "warmup_inferences": -1}, "warmup_inferences"),
        ({"model_path": "/m", "load_timeout_s": 0}, "load_timeout_s"),
    ],
)
def test_bad_config_raises(attrs, msg):
    with pytest.raises(ConfigError, match=msg):
        PolicyConfig.parse(attrs)


def test_read_package_builds_vla_specs(tmp_path):
    pkg = read_package(write_package(tmp_path / "m"), device="NPU")
    s = pkg.specs
    assert s["policy_type"] == "act"
    assert s["image_feature_keys"] == s["declared_image_feature_keys"] == ["images.cam_a", "images.cam_b"]
    assert s["input_features"] == {"images.cam_a": [3, 8, 16], "images.cam_b": [3, 16, 8], "state": [DIM]}
    assert s["output_features"] == {"action": [DIM]}
    assert (s["action_dim"], s["state_dim"], s["n_action_steps"]) == (DIM, DIM, N_STEPS)
    assert s["preprocess_image_size"] is None and s["num_steps"] is None
    assert (s["supports_rtc"], s["rtc_enabled"], s["relative_actions"]) == (False, False, False)
    assert (s["device"], s["dtype"]) == ("NPU", "float32")
    assert s["cameras"] == {"images.cam_a": "cam-a", "images.cam_b": "cam-b"} and s["arm"] == "arm"
    assert (pkg.state_input, pkg.default_task) == ("state", "open the box")


@pytest.mark.parametrize(
    "overrides, msg",
    [({"schema_version": 2}, "schema_version"), ({"policy": "pi05"}, "policy"), ({"cameras": "x"}, "invalid")],
)
def test_unsupported_package_raises(tmp_path, overrides, msg):
    with pytest.raises(LoadError, match=msg):
        read_package(write_package(tmp_path / "m", **overrides), device="CPU")


def test_missing_viam_json_raises(tmp_path):
    with pytest.raises(LoadError, match="viam.json"):
        read_package(tmp_path, device="CPU")


def test_malformed_viam_json_raises(tmp_path):
    (tmp_path / "viam.json").write_text("{not json")
    with pytest.raises(LoadError, match="viam.json"):
        read_package(tmp_path, device="CPU")
