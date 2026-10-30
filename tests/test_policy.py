import asyncio
import json
import os
import time

import numpy as np
import pytest
from viam.proto.app.robot import ServiceConfig
from viam.utils import dict_to_struct

from src.policy import ConfigError, LoadError, PhysicalAIPolicy, PolicyConfig, read_package
from src.wire import WireError
from tests.vla_encode import encode_image

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


class FakeModel:
    """Stands in for physicalai's InferenceModel: records batches, returns a fixed chunk."""

    instances: list = []

    def __init__(self, path, device):
        self.path, self.device, self.batches, self.resets = path, device, [], 0
        FakeModel.instances.append(self)

    def predict_action_chunk(self, batch):
        self.batches.append(batch)
        return np.arange(N_STEPS * DIM, dtype=np.float32).reshape(N_STEPS, DIM)

    def reset(self):
        self.resets += 1


async def make_policy(tmp_path, factory=FakeModel, **attrs):
    svc = PhysicalAIPolicy("policy")
    svc._model_factory = factory
    attrs = {"model_path": str(write_package(tmp_path / "m")), "device": "CPU"} | attrs
    svc.reconfigure(ServiceConfig(name="policy", attributes=dict_to_struct(attrs)), {})
    await svc._load_task
    return svc


def infer_request(**overrides):
    images = {key: encode_image(np.zeros((s[1], s[2], 3), np.uint8), encoding="jpeg") for _, key, s in CAMS}
    return {"command": "infer", "images": images, "state": {"values": [0.0] * DIM}, "task": "open the box"} | overrides


async def test_ready_after_load_with_warmups(tmp_path):
    FakeModel.instances.clear()
    svc = await make_policy(tmp_path, warmup_inferences=3)
    assert await svc.do_command({"command": "status"}) == {"state": "ready", "error": ""}
    model = FakeModel.instances[-1]
    assert (model.path, model.device, len(model.batches)) == (str(tmp_path / "m"), "CPU", 3)


async def test_specs_and_reset(tmp_path):
    FakeModel.instances.clear()
    svc = await make_policy(tmp_path)
    specs = await svc.do_command({"command": "specs"})
    assert specs["n_action_steps"] == N_STEPS and specs["image_feature_keys"] == ["images.cam_a", "images.cam_b"]
    assert await svc.do_command({"command": "reset"}) == {"ok": True}
    assert FakeModel.instances[-1].resets == 1


async def test_infer_converts_inputs_and_returns_chunk(tmp_path):
    FakeModel.instances.clear()
    svc = await make_policy(tmp_path, warmup_inferences=0)
    out = await svc.do_command(infer_request())
    rows = np.array(out["actions"]["rows"])
    assert rows.shape == (N_STEPS, DIM) and out["raw_actions"] == out["actions"] and out["latency_s"] >= 0
    batch = FakeModel.instances[-1].batches[-1]
    assert batch["images.cam_a"].shape == (1, 3, 8, 16) and batch["images.cam_a"].dtype == np.float32
    assert batch["images.cam_b"].shape == (1, 3, 16, 8)
    assert batch["state"].shape == (1, DIM) and batch["task"] == ["open the box"]


async def test_infer_defaults_task_to_viam_json(tmp_path):
    FakeModel.instances.clear()
    svc = await make_policy(tmp_path, warmup_inferences=0)
    request = infer_request()
    del request["task"]
    await svc.do_command(request)
    assert FakeModel.instances[-1].batches[-1]["task"] == ["open the box"]


@pytest.mark.parametrize(
    "overrides, msg",
    [
        ({"images": {}}, "image_feature_keys"),
        ({"state": {"values": [0.0] * (DIM + 1)}}, "state_dim"),
        ({"rtc": {"inference_delay": 1}}, "RTC"),
        ({"task": 3}, "task"),
    ],
)
async def test_bad_infer_requests_raise(tmp_path, overrides, msg):
    svc = await make_policy(tmp_path, warmup_inferences=0)
    with pytest.raises(WireError, match=msg):
        await svc.do_command(infer_request(**overrides))


async def test_load_failure_reports_status(tmp_path):
    svc = await make_policy(tmp_path, model_path=str(tmp_path / "missing"))
    status = await svc.do_command({"command": "status"})
    assert status["state"] == "failed" and "viam.json" in status["error"]
    with pytest.raises(RuntimeError, match="not ready"):
        await svc.do_command({"command": "specs"})


async def test_model_construction_error_reports_status(tmp_path):
    def broken(path, device):
        raise RuntimeError("NPU plugin not available")

    svc = await make_policy(tmp_path, factory=broken)
    status = await svc.do_command({"command": "status"})
    assert status["state"] == "failed" and "NPU plugin not available" in status["error"]


async def test_load_timeout_reports_status(tmp_path):
    def slow(path, device):
        time.sleep(0.5)
        return FakeModel(path, device)

    svc = await make_policy(tmp_path, factory=slow, load_timeout_s=0.05)
    status = await svc.do_command({"command": "status"})
    assert status["state"] == "failed" and "timed out" in status["error"]


async def test_unknown_command_and_close(tmp_path):
    svc = await make_policy(tmp_path)
    with pytest.raises(ValueError, match="unknown command"):
        await svc.do_command({"command": "nope"})
    await svc.close()
    with pytest.raises(RuntimeError, match="closed"):
        await svc.do_command({"command": "status"})


def test_validate_config_rejects_bad_device():
    cfg = ServiceConfig(name="policy", attributes=dict_to_struct({"model_path": "/m", "device": "TPU"}))
    with pytest.raises(ConfigError, match="device"):
        PhysicalAIPolicy.validate_config(cfg)


async def test_infer_layout_scaling_and_encodings(tmp_path):
    FakeModel.instances.clear()
    svc = await make_policy(tmp_path, warmup_inferences=0)
    img = np.zeros((8, 16, 3), np.uint8)
    img[0, 0] = (255, 0, 51)
    request = infer_request(state={"values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    request["images"]["images.cam_a"] = encode_image(img, encoding="raw")
    request["images"]["images.cam_b"] = encode_image(np.zeros((16, 8, 3), np.uint8), encoding="png")
    await svc.do_command(request)
    batch = FakeModel.instances[-1].batches[-1]
    np.testing.assert_allclose(batch["images.cam_a"][0, :, 0, 0], [1.0, 0.0, 0.2], atol=1e-6)
    assert not batch["images.cam_a"][0, :, 1, 0].any()
    np.testing.assert_array_equal(batch["state"][0], [1, 2, 3, 4, 5, 6])


async def test_infer_refused_before_ready(tmp_path):
    svc = await make_policy(tmp_path, model_path=str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="not ready"):
        await svc.do_command(infer_request())


async def test_reconfigure_during_load_discards_first_load(tmp_path):
    def slow(path, device):
        time.sleep(0.3)
        return FakeModel(path, device)

    svc = await make_policy(tmp_path, factory=slow)
    svc.reconfigure(
        ServiceConfig(name="policy", attributes=dict_to_struct({"model_path": str(tmp_path / "missing")})), {}
    )
    await svc._load_task
    await asyncio.sleep(0.5)
    status = await svc.do_command({"command": "status"})
    assert status["state"] == "failed" and "viam.json" in status["error"]


REAL_MODEL = os.environ.get("PHYSICALAI_TEST_MODEL")


@pytest.mark.integration
@pytest.mark.skipif(not REAL_MODEL, reason="set PHYSICALAI_TEST_MODEL to an exported model directory")
async def test_real_model_serves_a_chunk():
    svc = PhysicalAIPolicy("policy")
    attrs = {"model_path": REAL_MODEL, "device": "CPU", "warmup_inferences": 1}
    svc.reconfigure(ServiceConfig(name="policy", attributes=dict_to_struct(attrs)), {})
    await svc._load_task
    assert (await svc.do_command({"command": "status"}))["state"] == "ready"
    specs = await svc.do_command({"command": "specs"})
    images = {
        key: encode_image(np.zeros((*specs["input_features"][key][1:], 3), np.uint8), encoding="jpeg")
        for key in specs["image_feature_keys"]
    }
    out = await svc.do_command(
        {"command": "infer", "images": images, "state": {"values": [0.0] * specs["state_dim"]}}
    )
    rows = np.array(out["actions"]["rows"])
    assert rows.shape == (specs["n_action_steps"], specs["action_dim"]) and np.isfinite(rows).all()
