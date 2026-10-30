"""viam-labs:openvino-physicalai:policy — OpenVINO ACT inference over the viam-labs:vla policy protocol.

The vla controller (viam-labs/vla-inference-service) owns cameras, the arm, scheduling and
safety; this service only answers its status/specs/infer/reset DoCommands.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ClassVar, Mapping, Self, Sequence

import numpy as np
from viam.proto.app.robot import ServiceConfig
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.generic import Generic
from viam.utils import struct_to_dict

from .wire import WireError, decode_image, decode_vector, encode_matrix

LOGGER = logging.getLogger(__name__)

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
    except (OSError, ValueError) as exc:
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


def _inference_model(path: str, device: str) -> Any:
    # Imported here so config validation and the protocol tests never load OpenVINO.
    from physicalai.inference import InferenceModel

    return InferenceModel(path, backend="openvino", device=device)


def _require_dict(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WireError(f"{field} must be an object, got {type(value).__name__}")
    return value


class PhysicalAIPolicy(Generic, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily("viam-labs", "openvino-physicalai"), "policy")

    def __init__(self, name: str):
        super().__init__(name)
        # Tests inject a fake with InferenceModel's predict_action_chunk/reset.
        self._model_factory: Callable[[str, str], Any] = _inference_model
        self._model: Any = None
        self._package: Package | None = None
        self._state, self._error = "idle", ""
        self._load_task: asyncio.Task | None = None
        self._closed = False

    @classmethod
    def new(cls, config: ServiceConfig, dependencies: Mapping[Any, ResourceBase]) -> Self:
        svc = cls(config.name)
        svc.reconfigure(config, dependencies)
        return svc

    @classmethod
    def validate_config(cls, config: ServiceConfig) -> tuple[Sequence[str], Sequence[str]]:
        PolicyConfig.parse(struct_to_dict(config.attributes))
        return [], []

    def reconfigure(self, config: ServiceConfig, dependencies: Mapping[Any, ResourceBase]) -> None:
        """Start loading in the background: compiling for an NPU or GPU can take a while."""
        cfg = PolicyConfig.parse(struct_to_dict(config.attributes))
        if self._load_task and not self._load_task.done():
            self._load_task.cancel()
        self._model, self._package = None, None
        self._state, self._error = "loading", ""
        self._load_task = asyncio.create_task(self._load(cfg))

    async def _load(self, cfg: PolicyConfig) -> None:
        try:
            package, model = await asyncio.wait_for(asyncio.to_thread(self._load_sync, cfg), cfg.load_timeout_s)
        except asyncio.TimeoutError:
            self._state, self._error = "failed", f"load timed out after {cfg.load_timeout_s}s (load_timeout_s)"
        except LoadError as exc:
            self._state, self._error = "failed", str(exc)
        except Exception as exc:  # noqa: BLE001 - must land in status, not die in a background task
            LOGGER.exception("policy load failed")
            self._state, self._error = "failed", f"{type(exc).__name__}: {exc}"
        else:
            self._package, self._model, self._state = package, model, "ready"
            LOGGER.info("policy ready on %s: %s", cfg.device, package.specs)
            return
        LOGGER.error("policy load failed: %s", self._error)

    def _load_sync(self, cfg: PolicyConfig) -> tuple[Package, Any]:
        package = read_package(cfg.model_path, device=cfg.device)
        model = self._model_factory(cfg.model_path, cfg.device)
        # Throwaway inferences: the first calls on a freshly compiled model are slow.
        specs = package.specs
        images = {key: np.zeros((*specs["input_features"][key][1:], 3), np.uint8) for key in specs["image_feature_keys"]}
        state = np.zeros(specs["state_dim"], np.float32)
        for _ in range(cfg.warmup_inferences):
            model.predict_action_chunk(self._batch(package, images, state, package.default_task))
        return package, model

    @staticmethod
    def _batch(package: Package, images: dict[str, np.ndarray], state: np.ndarray, task: str) -> dict[str, Any]:
        """HWC uint8 frames -> InferenceModel's batched CHW float32 in [0, 1]."""
        batch: dict[str, Any] = {
            key: (img.transpose(2, 0, 1).astype(np.float32) / 255.0)[None] for key, img in images.items()
        }
        batch[package.state_input] = state.astype(np.float32)[None]
        batch["task"] = [task]
        return batch

    async def close(self) -> None:
        self._closed = True
        if self._load_task and not self._load_task.done():
            LOGGER.warning("close() while loading %s: the load thread keeps running until it finishes", self.name)
            # The worker thread cannot be interrupted; this only stops waiting on it.
            self._load_task.cancel()
        self._model = None

    async def do_command(self, command: Mapping[str, Any], *, timeout: float | None = None, **kwargs) -> Mapping[str, Any]:
        if self._closed:
            raise RuntimeError(f"policy {self.name!r} is closed")
        name = command.get("command")
        if name == "status":
            return {"state": self._state, "error": self._error}
        if name == "specs":
            self._require_ready()
            return self._package.specs
        if name == "reset":
            self._require_ready()
            self._model.reset()
            return {"ok": True}
        if name == "infer":
            return await self._infer(command)
        raise ValueError(f"unknown command {name!r}")

    def _require_ready(self) -> None:
        if self._state != "ready":
            raise RuntimeError(f"policy not ready (state={self._state}, error={self._error})")

    async def _infer(self, command: Mapping[str, Any]) -> Mapping[str, Any]:
        self._require_ready()
        model, package = self._model, self._package  # a reconfigure mid-request must not swap them
        specs = package.specs
        if command.get("rtc"):
            raise WireError("RTC is not supported by ACT policies (specs.supports_rtc is false)")
        started = time.perf_counter()
        images = {}
        for key, payload in _require_dict(command.get("images") or {}, "images").items():
            try:
                images[key] = decode_image(_require_dict(payload, "image payload"))
            except WireError as exc:
                raise WireError(f"{key}: {exc}") from exc
        if set(images) != set(specs["image_feature_keys"]):
            raise WireError(
                f"'images' keys {sorted(images)} do not match the policy's image_feature_keys "
                f"{specs['image_feature_keys']}"
            )
        if "state" not in command:
            raise WireError("infer requires a 'state' vector")
        state = decode_vector(_require_dict(command["state"], "state"))
        if state.shape[0] != specs["state_dim"]:
            raise WireError(f"'state' has length {state.shape[0]}, expected {specs['state_dim']} (specs.state_dim)")
        task = command.get("task")
        if task is not None and not isinstance(task, str):
            raise WireError(f"'task' must be a string, got {type(task).__name__}")
        task = task or package.default_task  # the vla controller sends "" when no task is configured
        chunk = await asyncio.to_thread(model.predict_action_chunk, self._batch(package, images, state, task))
        actions = encode_matrix(np.asarray(chunk, np.float32))
        # ACT has no separate policy space, so raw_actions repeats actions for protocol parity.
        return {"actions": actions, "raw_actions": actions, "latency_s": time.perf_counter() - started}
