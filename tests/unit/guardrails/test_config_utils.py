# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``guardrails.src.config_utils``."""

from __future__ import annotations

from pathlib import Path
import os
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
import yaml

from guardrails.src.config_utils import apply_model_config


def _make_config(model_entries: List[Dict[str, Any]]) -> SimpleNamespace:
    models = [
        SimpleNamespace(
            type=entry["type"],
            model=entry.get("model", "old-model"),
            parameters=dict(entry.get("parameters", {})),
        )
        for entry in model_entries
    ]
    return SimpleNamespace(models=models)


def _write_model_config(root: Path) -> Path:
    config_root = root / "configs"
    rails_dir = config_root / "rails"
    rails_dir.mkdir(parents=True)
    (config_root / "models.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "local_nims": {"required_env": [], "services": {}},
                "models": {
                    "app_llm": {
                        "source": "endpoint",
                        "base_url": "https://llm.example/v1",
                        "model": "llm-model",
                        "api_key_env": None,
                    },
                    "content_safety": {
                        "source": "endpoint",
                        "base_url": "https://content.example/v1",
                        "model": "content-model",
                        "api_key_env": "RAIL_API_KEY",
                    },
                    "topic_control": {
                        "source": "endpoint",
                        "base_url": "https://topic.example/v1",
                        "model": "topic-model",
                        "api_key_env": "RAIL_API_KEY",
                    },
                },
            }
        )
    )
    return rails_dir


class TestApplyModelConfig:
    def test_model_config_updates_guardrails_models(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        rails_dir = _write_model_config(tmp_path)
        monkeypatch.setenv("SHARED_CONFIG_ROOT", str(rails_dir.parent))
        monkeypatch.setenv("RAIL_API_KEY", "secret-value")
        config = _make_config(
            [
                {"type": "main", "parameters": {"base_url": "http://old-main"}},
                {"type": "content_safety", "parameters": {"base_url": "http://old-content"}},
                {"type": "topic_control", "parameters": {"base_url": "http://old-topic"}},
            ]
        )

        apply_model_config(config, config_dir=str(rails_dir))

        assert config.models[0].model == "llm-model"
        assert config.models[0].parameters["base_url"] == "https://llm.example/v1"
        assert config.models[1].model == "content-model"
        assert config.models[1].parameters["base_url"] == "https://content.example/v1"
        assert config.models[2].model == "topic-model"
        assert config.models[2].parameters["base_url"] == "https://topic.example/v1"
        assert os.environ["NVIDIA_API_KEY"] == "secret-value"

    def test_unrelated_model_type_is_ignored(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        rails_dir = _write_model_config(tmp_path)
        monkeypatch.setenv("SHARED_CONFIG_ROOT", str(rails_dir.parent))
        config = _make_config(
            [{"type": "unrelated", "model": "old", "parameters": {"base_url": "http://old"}}]
        )

        apply_model_config(config, config_dir=str(rails_dir))

        assert config.models[0].model == "old"
        assert config.models[0].parameters["base_url"] == "http://old"
