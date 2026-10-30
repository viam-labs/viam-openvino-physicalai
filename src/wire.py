"""DoCommand wire codec, mirroring viam-labs/vla-inference-service src/vla/wire.py @ 4a69106 (decode side). Keep in step with it: the vla controller is the client."""

from __future__ import annotations

import base64
import binascii
import io
from typing import Any

import numpy as np
from PIL import Image

_IMAGE_ENCODINGS = ("raw", "jpeg", "png")


class WireError(ValueError):
    """Raised when a wire payload is malformed."""


def decode_image(payload: dict[str, Any]) -> np.ndarray:
    # Check the encoding before touching 'data' at all: an unknown encoding
    # is a different fault than missing/corrupt data, and shouldn't be
    # masked by a base64 decode of a blob we're about to discard anyway.
    encoding = payload.get("encoding")
    if encoding not in _IMAGE_ENCODINGS:
        raise WireError(f"unknown encoding {encoding!r}")

    try:
        data = payload["data"]
    except KeyError as exc:
        raise WireError("image payload missing 'data'") from exc
    try:
        raw = base64.b64decode(data)
    except (binascii.Error, TypeError) as exc:
        raise WireError(f"invalid base64 'data' for {encoding!r} image") from exc

    if encoding == "raw":
        # Shape is carried explicitly so the decoder never has to infer it.
        try:
            height = int(payload["height"])
            width = int(payload["width"])
            channels = int(payload["channels"])
        except KeyError as exc:
            raise WireError("raw image payload missing shape fields") from exc
        except (TypeError, ValueError) as exc:
            raise WireError("raw image payload has non-numeric shape fields") from exc
        if height <= 0 or width <= 0:
            raise WireError(f"raw image payload has non-positive dimensions {(height, width)}")
        if channels != 3:
            raise WireError(f"raw image payload must have 3 channels, got {channels}")
        shape = (height, width, channels)
        expected = height * width * channels
        if len(raw) != expected:
            raise WireError(f"raw image payload is {len(raw)} bytes, expected {expected}")
        # Every path below guarantees a writable array: a bare frombuffer
        # view (or a PIL-backed np.asarray) is read-only, and torch.from_numpy
        # warns on non-writable input downstream.
        return np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()

    # jpeg / png
    try:
        # np.array (not np.asarray) over a PIL image copies, for the same
        # reason the raw branch does: torch.from_numpy warns on non-writable
        # input.
        out = np.array(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)
    except OSError as exc:
        raise WireError(f"could not decode {encoding!r} image data") from exc

    # The encoder writes height/width on every path; honor them on every
    # path too. A silent mismatch here (e.g. a controller resize that didn't
    # happen) would otherwise surface many layers away as "the model
    # behaves oddly".
    declared_height = payload.get("height")
    declared_width = payload.get("width")
    if declared_height is not None and out.shape[0] != int(declared_height):
        raise WireError(
            f"decoded {encoding} image height {out.shape[0]} does not match declared height {declared_height}"
        )
    if declared_width is not None and out.shape[1] != int(declared_width):
        raise WireError(
            f"decoded {encoding} image width {out.shape[1]} does not match declared width {declared_width}"
        )
    return out

def encode_matrix(m: np.ndarray) -> dict[str, Any]:
    if m.ndim != 2:
        raise WireError(f"expected 2D matrix, got shape {m.shape}")
    return {"rows": m.astype(float, copy=False).tolist()}


def decode_vector(payload: dict[str, Any]) -> np.ndarray:
    values = payload.get("values")
    if values is None:
        raise WireError("vector payload missing 'values'")
    if not isinstance(values, list):
        raise WireError(f"vector payload 'values' must be a list, got {type(values).__name__}")
    for i, v in enumerate(values):
        if v is None:
            raise WireError(f"vector payload has a null value at index {i}")
    try:
        out = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise WireError(f"vector payload 'values' must be a flat list of numbers: {exc}") from exc
    if out.ndim != 1:
        raise WireError(f"vector payload 'values' must be a flat list, got shape {out.shape}")
    return out
