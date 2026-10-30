"""viam-labs:openvino-physicalai:policy — OpenVINO ACT inference over the viam-labs:vla policy protocol.

The vla controller (viam-labs/vla-inference-service) owns cameras, the arm, scheduling and
safety; this service only answers its status/specs/infer/reset DoCommands.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEVICES = ("AUTO", "NPU", "GPU", "CPU")


class ConfigError(ValueError):
    """Invalid service configuration."""


class LoadError(Exception):
    """The model package cannot be served; the message is shown in `status`."""


def _int(value: Any, name: str, minimum: int, maximum: int) -> int:
    # Struct numbers arrive as floats: accept 2.0, reject 2.5 and booleans.
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value):
        raise ConfigError(f"{name} must be an integer, got {value!r}")
    if not minimum <= value <= maximum:
        raise ConfigError(f"{name} must be between {minimum} and {maximum}, got {value!r}")
    return int(value)


@dataclass(frozen=True)
class PolicyConfig:
    model_path: str
    device: str = "AUTO"
    warmup_inferences: int = 2
    load_timeout_s: float = 600.0

    @staticmethod
    def parse(attrs: dict[str, Any]) -> PolicyConfig:
        path = attrs.get("model_path")
        if not isinstance(path, str) or not path:
            raise ConfigError("model_path is required: a directory holding viam.json and the OpenVINO export")
        device = attrs.get("device", "AUTO")
        if device not in DEVICES:
            raise ConfigError(f"device must be one of {DEVICES}, got {device!r}")
        timeout = attrs.get("load_timeout_s", 600.0)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout < math.inf:
            raise ConfigError(f"load_timeout_s must be a positive number, got {timeout!r}")
        return PolicyConfig(
            model_path=path,
            device=device,
            warmup_inferences=_int(attrs.get("warmup_inferences", 2), "warmup_inferences", 0, 100),
            load_timeout_s=float(timeout),
        )


@dataclass(frozen=True)
class Package:
    specs: dict[str, Any]
    state_input: str
    default_task: str


def read_package(path: Path | str, *, device: str) -> Package:
    """Build vla-protocol specs from the package's viam.json (written by the training script)."""
    path = Path(path)
    try:
        meta = json.loads((path / "viam.json").read_text())
    except OSError as exc:
        raise LoadError(f"cannot read viam.json in {path}: {exc}") from exc
    if meta.get("schema_version") != 1:
        raise LoadError(f"unsupported viam.json schema_version {meta.get('schema_version')!r} (expected 1)")
    if meta.get("policy") != "act":
        raise LoadError(f"unsupported policy {meta.get('policy')!r} (only act)")
    try:
        cameras = [(c["input"], [int(n) for n in c["shape"]], c["component"]) for c in meta["cameras"]]
        state, action = meta["state"], meta["action"]
        keys = [key for key, _, _ in cameras]
        specs = {
            "policy_type": "act",
            "action_dim": int(action["dim"]),
            "state_dim": int(state["dim"]),
            # What predict_action_chunk returns: ACT's export trims the chunk to n_action_steps.
            "n_action_steps": int(meta["training"]["args"]["n_action_steps"]),
            "input_features": {**{key: shape for key, shape, _ in cameras}, state["input"]: [int(state["dim"])]},
            "output_features": {"action": [int(action["dim"])]},
            "image_feature_keys": keys,
            "declared_image_feature_keys": list(keys),
            # InferenceModel letterboxes to the graph input itself, as in training.
            "preprocess_image_size": None,
            "num_steps": None,
            "supports_rtc": False,
            "rtc_enabled": False,
            "relative_actions": False,
            "device": device,
            "dtype": "float32",
            # Not read by the controller: helpers for writing its `cameras` and `arm` config.
            "cameras": {key: component for key, _, component in cameras},
            "arm": action["component"],
        }
        tasks = meta.get("tasks") or [""]
        return Package(specs=specs, state_input=state["input"], default_task=tasks[0])
    except (KeyError, TypeError, ValueError) as exc:
        raise LoadError(f"invalid viam.json in {path}: {type(exc).__name__}: {exc}") from exc
