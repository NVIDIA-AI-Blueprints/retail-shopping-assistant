# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``catalog_retriever.src.utils``.

The module handles three kinds of image conversion paths (url, local file,
raw PIL image) plus URL/path detection and resize-base64 logic. We use
``Pillow`` to fabricate tiny images in-memory so no filesystem or network
access is required.
"""

from __future__ import annotations

import base64
import io
from typing import Any
from unittest.mock import patch

import pytest
from PIL import Image

from catalog_retriever.src import utils as utils_mod
from catalog_retriever.src.utils import (
    image_path_to_base64,
    image_to_base64,
    image_url_to_base64,
    is_path,
    is_url,
    resize_base64_image,
)


# --------------------------------------------------------------------------->
# Helpers
# --------------------------------------------------------------------------->


def _build_jpeg_bytes(size: tuple[int, int] = (64, 64), color: str = "red") -> bytes:
    """Produce a small valid JPEG byte string."""
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG")
    return buffer.getvalue()


def _build_png_bytes(size: tuple[int, int] = (64, 64), color: str = "blue") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


# --------------------------------------------------------------------------->
# is_url / is_path
# --------------------------------------------------------------------------->


class TestIsUrl:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("http://example.com/a.png", True),
            ("https://example.com/a.png", True),
            ("HTTP://EXAMPLE.COM", False),  # scheme check is case-sensitive
            ("ftp://example.com/a.png", False),
            ("/tmp/foo.jpg", False),
            ("foo.jpg", False),
            ("", False),
        ],
    )
    def test_scheme_detection(self, value: str, expected: bool) -> None:
        assert is_url(value) is expected


class TestIsPath:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("/tmp/foo.jpg", True),
            ("/a", True),
            ("./a", False),
            ("foo.jpg", False),
            ("http://a.com/a.png", False),
            ("", False),
        ],
    )
    def test_path_detection(self, value: str, expected: bool) -> None:
        assert is_path(value) is expected


# --------------------------------------------------------------------------->
# image_path_to_base64
# --------------------------------------------------------------------------->


class TestImagePathToBase64:
    def test_reads_file_and_emits_data_uri(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The function hardcodes the ``/app/shared/`` prefix, so patch
        # ``open`` to redirect the path lookup to our temp file.
        img_bytes = _build_jpeg_bytes()
        expected_path = "/app/shared/img.jpg"
        real_path = tmp_path / "img.jpg"
        real_path.write_bytes(img_bytes)

        original_open = open

        def _fake_open(path: str, mode: str = "r", *args: Any, **kwargs: Any):
            if path == expected_path:
                return original_open(real_path, mode, *args, **kwargs)
            return original_open(path, mode, *args, **kwargs)

        monkeypatch.setattr("builtins.open", _fake_open)

        result = image_path_to_base64("img.jpg")

        assert result is not None
        assert result.startswith("data:image/jpeg;base64,")

        _, b64 = result.split(",", 1)
        decoded = base64.b64decode(b64)
        img = Image.open(io.BytesIO(decoded))
        assert img.format == "JPEG"

    def test_returns_none_when_encoded_exceeds_limit(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        img_bytes = _build_jpeg_bytes(size=(32, 32))
        real_path = tmp_path / "small.jpg"
        real_path.write_bytes(img_bytes)

        original_open = open

        def _fake_open(path: str, mode: str = "r", *args: Any, **kwargs: Any):
            if path.startswith("/app/shared/"):
                return original_open(real_path, mode, *args, **kwargs)
            return original_open(path, mode, *args, **kwargs)

        monkeypatch.setattr("builtins.open", _fake_open)

        result = image_path_to_base64("small.jpg", max_b64_length=50)

        assert result is None


# --------------------------------------------------------------------------->
# image_url_to_base64
# --------------------------------------------------------------------------->


class _FakeResponse:
    def __init__(
        self,
        content: bytes,
        content_type: str = "image/jpeg",
        status: int = 200,
    ) -> None:
        self.content = content
        self.headers = {"Content-Type": content_type}
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise utils_mod.requests.exceptions.HTTPError(
                f"HTTP {self.status_code}"
            )


class TestImageUrlToBase64:
    def test_successful_fetch_returns_data_uri(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            utils_mod.requests,
            "get",
            lambda url, timeout=120: _FakeResponse(_build_jpeg_bytes()),
        )

        result = image_url_to_base64("http://example.com/a.jpg")

        assert result is not None
        assert result.startswith("data:image/jpeg;base64,")

    def test_http_error_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raise(*_: Any, **__: Any) -> None:
            raise utils_mod.requests.exceptions.ConnectionError("down")

        monkeypatch.setattr(utils_mod.requests, "get", _raise)

        assert image_url_to_base64("http://dead.example.com/a.jpg") is None

    def test_invalid_image_bytes_return_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            utils_mod.requests,
            "get",
            lambda url, timeout=120: _FakeResponse(b"not an image"),
        )

        assert image_url_to_base64("http://example.com/garbage") is None

    def test_too_large_encoded_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            utils_mod.requests,
            "get",
            lambda url, timeout=120: _FakeResponse(_build_jpeg_bytes()),
        )

        # Force max_b64_length below any plausible encoded size.
        assert (
            image_url_to_base64("http://example.com/a.jpg", max_b64_length=10)
            is None
        )


# --------------------------------------------------------------------------->
# image_to_base64 (raw PIL input)
# --------------------------------------------------------------------------->


class TestImageToBase64:
    def test_roundtrips_pil_image(self) -> None:
        img = Image.new("RGB", (16, 16), color="green")
        result = image_to_base64(img)

        assert result.startswith("data:image/jpeg;base64,")
        _, b64 = result.split(",", 1)
        decoded = base64.b64decode(b64)
        recovered = Image.open(io.BytesIO(decoded))
        assert recovered.format == "JPEG"


# --------------------------------------------------------------------------->
# resize_base64_image
# --------------------------------------------------------------------------->


class TestResizeBase64Image:
    def test_resize_keeps_data_uri_prefix(self) -> None:
        jpeg = _build_jpeg_bytes(size=(512, 512))
        data_uri = f"data:image/jpeg;base64,{base64.b64encode(jpeg).decode()}"

        out = resize_base64_image(data_uri, max_width=32, max_height=32)

        assert out is not None
        assert out.startswith("data:image/jpeg;base64,")

        _, b64 = out.split(",", 1)
        resized_img = Image.open(io.BytesIO(base64.b64decode(b64)))
        width, height = resized_img.size
        assert width <= 32 and height <= 32

    def test_plain_base64_without_prefix_gets_default_header(self) -> None:
        jpeg = _build_jpeg_bytes(size=(128, 128))
        plain_b64 = base64.b64encode(jpeg).decode()

        out = resize_base64_image(plain_b64, max_width=32, max_height=32)

        assert out is not None
        assert out.startswith("data:image/jpeg;base64,")

    def test_invalid_base64_returns_none(self) -> None:
        assert resize_base64_image("data:image/jpeg;base64,!!not base64!!") is None
