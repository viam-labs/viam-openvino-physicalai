"""vla's encode_image, copied verbatim from viam-labs/vla-inference-service
src/vla/wire.py @ 4a69106: the controller builds every infer image with it, so decoding
its output is the protocol-compatibility check."""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
from PIL import Image

WireError = ValueError
_IMAGE_ENCODINGS = ("raw", "jpeg", "png")


def encode_image(img: np.ndarray, *, encoding: str = "jpeg", quality: int = 90) -> dict[str, Any]:
    if img.ndim != 3 or img.shape[2] != 3:
        raise WireError(f"expected HWC RGB image, got shape {img.shape}")
    if img.dtype != np.uint8:
        raise WireError(f"expected uint8 image, got {img.dtype}")
    if encoding not in _IMAGE_ENCODINGS:
        raise WireError(f"unknown encoding {encoding!r}")

    height, width, channels = img.shape

    if encoding == "raw":
        data = base64.b64encode(img.tobytes()).decode("ascii")
    else:
        # jpeg / png. libjpeg silently clamps an out-of-range quality to its
        # nearest valid value instead of erroring, which at 10 Hz can burn
        # several times the intended bandwidth with nothing in the log.
        if encoding == "jpeg" and (not isinstance(quality, int) or isinstance(quality, bool) or not (0 <= quality <= 100)):
            raise WireError(f"quality must be an integer in [0, 100], got {quality!r}")
        buf = io.BytesIO()
        pil_format = "JPEG" if encoding == "jpeg" else "PNG"
        kwargs = {"quality": quality} if encoding == "jpeg" else {}
        Image.fromarray(img).save(buf, format=pil_format, **kwargs)
        data = base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "encoding": encoding,
        "data": data,
        "height": height,
        "width": width,
        "channels": channels,
    }
