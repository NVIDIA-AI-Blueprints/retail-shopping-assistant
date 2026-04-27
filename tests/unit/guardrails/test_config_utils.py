# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``guardrails.src.config_utils``.

The module is a small helper that mutates a ``RailsConfig``-shaped object
with base URL overrides from a YAML file. We test it using lightweight
stand-in objects so no ``nemoguardrails`` installation is required at
import time.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
import yaml

from guardrails.src.config_utils import apply_endpoint_overrides


def _make_config(model_entries: List[Dict[str, Any]]) -> SimpleNamespace:
    """Build a RailsConfig-like object with ``models`` attribute.

    Each model exposes ``type`` and a mutable ``parameters`` dict so we can
    observe updates done by ``apply_endpoint_overrides``.
    """
    models = [
        SimpleNamespace(
            type=entry["type"],
            parameters=dict(entry.get("parameters", {})),
        )
        for entry in model_entries
    ]
    return SimpleNamespace(models=models)


class TestApplyEndpointOverrides:
    def test_no_override_env_leaves_config_untouched(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.delenv("CONFIG_OVERRIDE", raising=False)
        config = _make_config(
            [{"type": "main", "parameters": {"base_url": "http://default"}}]
        )

        apply_endpoint_overrides(config, config_dir=str(tmp_path))

        assert config.models[0].parameters["base_url"] == "http://default"

    def test_missing_override_file_is_tolerated(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("CONFIG_OVERRIDE", "missing.yaml")
        config = _make_config(
            [{"type": "main", "parameters": {"base_url": "http://default"}}]
        )

        apply_endpoint_overrides(config, config_dir=str(tmp_path))

        # No exception; config untouched.
        assert config.models[0].parameters["base_url"] == "http://default"

    def test_override_updates_matching_model_base_url(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        override_path = tmp_path / "override.yaml"
        override_path.write_text(
            yaml.safe_dump(
                {
                    "models": [
                        {
                            "type": "main",
                            "parameters": {
                                "base_url": "https://integrate.api.nvidia.com/v1",
                            },
                        }
                    ]
                }
            )
        )
        monkeypatch.setenv("CONFIG_OVERRIDE", "override.yaml")
        config = _make_config(
            [{"type": "main", "parameters": {"base_url": "http://default"}}]
        )

        apply_endpoint_overrides(config, config_dir=str(tmp_path))

        assert (
            config.models[0].parameters["base_url"]
            == "https://integrate.api.nvidia.com/v1"
        )

    def test_override_of_non_matching_type_is_ignored(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        override_path = tmp_path / "override.yaml"
        override_path.write_text(
            yaml.safe_dump(
                {
                    "models": [
                        {
                            "type": "content-safety",
                            "parameters": {"base_url": "https://safety"},
                        }
                    ]
                }
            )
        )
        monkeypatch.setenv("CONFIG_OVERRIDE", "override.yaml")
        config = _make_config(
            [{"type": "main", "parameters": {"base_url": "http://default"}}]
        )

        apply_endpoint_overrides(config, config_dir=str(tmp_path))

        # No matching type → nothing to update.
        assert config.models[0].parameters["base_url"] == "http://default"

    def test_override_without_base_url_is_skipped(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        override_path = tmp_path / "override.yaml"
        override_path.write_text(
            yaml.safe_dump(
                {
                    "models": [
                        {
                            "type": "main",
                            "parameters": {"other_param": "x"},
                        }
                    ]
                }
            )
        )
        monkeypatch.setenv("CONFIG_OVERRIDE", "override.yaml")
        config = _make_config(
            [{"type": "main", "parameters": {"base_url": "http://default"}}]
        )

        apply_endpoint_overrides(config, config_dir=str(tmp_path))

        assert config.models[0].parameters["base_url"] == "http://default"

    def test_override_without_models_key_is_noop(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        override_path = tmp_path / "override.yaml"
        override_path.write_text(yaml.safe_dump({"other_key": "value"}))
        monkeypatch.setenv("CONFIG_OVERRIDE", "override.yaml")
        config = _make_config(
            [{"type": "main", "parameters": {"base_url": "http://default"}}]
        )

        apply_endpoint_overrides(config, config_dir=str(tmp_path))

        assert config.models[0].parameters["base_url"] == "http://default"

    def test_multiple_models_update_first_match_only(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        override_path = tmp_path / "override.yaml"
        override_path.write_text(
            yaml.safe_dump(
                {
                    "models": [
                        {
                            "type": "main",
                            "parameters": {"base_url": "https://new"},
                        }
                    ]
                }
            )
        )
        monkeypatch.setenv("CONFIG_OVERRIDE", "override.yaml")
        config = _make_config(
            [
                {"type": "main", "parameters": {"base_url": "http://default1"}},
                {"type": "main", "parameters": {"base_url": "http://default2"}},
            ]
        )

        apply_endpoint_overrides(config, config_dir=str(tmp_path))

        # The function breaks after the first match; the second stays put.
        assert config.models[0].parameters["base_url"] == "https://new"
        assert config.models[1].parameters["base_url"] == "http://default2"
