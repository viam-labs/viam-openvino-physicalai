import base64

import numpy as np
import pytest

from src.wire import WireError, decode_image, decode_vector, encode_matrix
from tests.vla_encode import encode_image

RNG = np.random.default_rng(0)
IMG = RNG.integers(0, 255, (12, 20, 3), dtype=np.uint8)


@pytest.mark.parametrize("encoding", ["raw", "png"])
def test_lossless_round_trip(encoding):
    np.testing.assert_array_equal(decode_image(encode_image(IMG, encoding=encoding)), IMG)


def test_jpeg_round_trip_shape():
    out = decode_image(encode_image(IMG, encoding="jpeg"))
    assert out.shape == IMG.shape and out.dtype == np.uint8


@pytest.mark.parametrize(
    "payload, msg",
    [
        ({"encoding": "bmp", "data": ""}, "unknown encoding"),
        ({"encoding": "png"}, "missing 'data'"),
        ({"encoding": "png", "data": "abc"}, "invalid base64"),
        ({"encoding": "raw", "data": base64.b64encode(b"12").decode(), "height": 1, "width": 1, "channels": 3},
         "bytes, expected"),
    ],
)
def test_bad_images_raise(payload, msg):
    with pytest.raises(WireError, match=msg):
        decode_image(payload)


def test_declared_size_mismatch_raises():
    payload = encode_image(IMG, encoding="png") | {"height": 99}
    with pytest.raises(WireError, match="does not match declared height"):
        decode_image(payload)


def test_vectors_and_matrices():
    np.testing.assert_array_equal(decode_vector({"values": [1, 2.5]}), np.array([1, 2.5], np.float32))
    with pytest.raises(WireError, match="null value"):
        decode_vector({"values": [1, None]})
    assert encode_matrix(np.ones((2, 3), np.float32)) == {"rows": [[1.0] * 3] * 2}
