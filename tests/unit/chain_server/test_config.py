# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``chain_server.src.config``.

The config module drives every downstream agent's construction. These tests
exercise both the on-disk YAML override flow and the pydantic validation
contract directly, without touching the real container layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
import yaml
from pydantic import ValidationError

from chain_server.src.config import (
    ChainServerConfig,
    load_config,
    load_config_with_override,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def write_yaml(tmp_path: Path):
    """Helper to drop a YAML config into a temporary directory."""

    def _write(name: str, data: Dict[str, Any]) -> Path:
        path = tmp_path / name
        path.write_text(yaml.safe_dump(data))
        return path

    return _write


class TestLoadConfigWithOverride:
    def test_returns_base_config_when_no_override_set(
        self, monkeypatch: pytest.MonkeyPatch, write_yaml, valid_config_dict: dict
    ) -> None:
        monkeypatch.delenv("CONFIG_OVERRIDE", raising=False)
        base_path = write_yaml("config.yaml", valid_config_dict)

        result = load_config_with_override(str(base_path))

        assert result == valid_config_dict

    def test_raises_file_not_found_for_missing_base_config(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist.yaml"
        with pytest.raises(FileNotFoundError):
            load_config_with_override(str(missing))

    def test_applies_top_level_override(
        self, monkeypatch: pytest.MonkeyPatch, write_yaml, valid_config_dict: dict
    ) -> None:
        base_path = write_yaml("config.yaml", valid_config_dict)
        write_yaml(
            "config-build.yaml",
            {
                "llm_port": "https://integrate.api.nvidia.com/v1",
                "llm_name": "overridden-model",
            },
        )
        monkeypatch.setenv("CONFIG_OVERRIDE", "config-build.yaml")

        merged = load_config_with_override(str(base_path))

        assert merged["llm_port"] == "https://integrate.api.nvidia.com/v1"
        assert merged["llm_name"] == "overridden-model"
        # Base-only fields survive the shallow merge.
        assert merged["memory_port"] == valid_config_dict["memory_port"]
        assert merged["categories"] == valid_config_dict["categories"]

    def test_override_is_shallow_not_deep_merge(
        self, monkeypatch: pytest.MonkeyPatch, write_yaml, valid_config_dict: dict
    ) -> None:
        # Document that nested keys under a shared top-level key are replaced
        # wholesale (not merged). A regression to deep-merge would change
        # deployment behaviour and should surface in CI.
        base = {**valid_config_dict, "categories": ["bag", "shoes", "dress"]}
        base_path = write_yaml("config.yaml", base)
        write_yaml(
            "config-build.yaml",
            {"categories": ["sunglasses"]},
        )
        monkeypatch.setenv("CONFIG_OVERRIDE", "config-build.yaml")

        merged = load_config_with_override(str(base_path))

        assert merged["categories"] == ["sunglasses"]

    def test_missing_override_file_is_tolerated(
        self, monkeypatch: pytest.MonkeyPatch, write_yaml, valid_config_dict: dict
    ) -> None:
        base_path = write_yaml("config.yaml", valid_config_dict)
        monkeypatch.setenv("CONFIG_OVERRIDE", "missing-override.yaml")

        merged = load_config_with_override(str(base_path))

        assert merged == valid_config_dict


class TestChainServerConfigValidation:
    def test_valid_dict_constructs_successfully(self, valid_config_dict: dict) -> None:
        config = ChainServerConfig(**valid_config_dict)

        assert config.llm_port == valid_config_dict["llm_port"]
        assert config.categories == valid_config_dict["categories"]
        assert config.multimodal is True

    @pytest.mark.parametrize(
        "missing_field",
        [
            "llm_port",
            "llm_name",
            "retriever_port",
            "memory_port",
            "rails_port",
            "routing_prompt",
            "chatter_prompt",
            "categories",
            "agent_choices",
            "memory_length",
            "top_k_retrieve",
            "multimodal",
            "unsafe_message",
        ],
    )
    def test_missing_required_field_fails(
        self, valid_config_dict: dict, missing_field: str
    ) -> None:
        bad = dict(valid_config_dict)
        del bad[missing_field]
        with pytest.raises(ValidationError):
            ChainServerConfig(**bad)

    @pytest.mark.parametrize(
        "url_field",
        ["llm_port", "retriever_port", "memory_port", "rails_port"],
    )
    def test_url_validator_rejects_non_http_schemes(
        self, valid_config_dict: dict, url_field: str
    ) -> None:
        bad = {**valid_config_dict, url_field: "not-a-url"}
        with pytest.raises(ValidationError):
            ChainServerConfig(**bad)

    @pytest.mark.parametrize(
        "url_field,value",
        [
            ("llm_port", "http://localhost:8000"),
            ("retriever_port", "https://example.com"),
        ],
    )
    def test_url_validator_accepts_http_and_https(
        self, valid_config_dict: dict, url_field: str, value: str
    ) -> None:
        cfg = ChainServerConfig(**{**valid_config_dict, url_field: value})
        assert getattr(cfg, url_field) == value

    @pytest.mark.parametrize("value", [0, -1, -100])
    def test_memory_length_must_be_positive(
        self, valid_config_dict: dict, value: int
    ) -> None:
        with pytest.raises(ValidationError):
            ChainServerConfig(**{**valid_config_dict, "memory_length": value})

    @pytest.mark.parametrize("value", [0, -4])
    def test_top_k_retrieve_must_be_positive(
        self, valid_config_dict: dict, value: int
    ) -> None:
        with pytest.raises(ValidationError):
            ChainServerConfig(**{**valid_config_dict, "top_k_retrieve": value})

    @pytest.mark.parametrize("field", ["categories", "agent_choices"])
    def test_empty_list_fields_are_rejected(
        self, valid_config_dict: dict, field: str
    ) -> None:
        with pytest.raises(ValidationError):
            ChainServerConfig(**{**valid_config_dict, field: []})

    def test_extra_fields_are_forbidden(self, valid_config_dict: dict) -> None:
        with pytest.raises(ValidationError):
            ChainServerConfig(**valid_config_dict, unexpected_field="oops")


class TestLoadConfig:
    def test_returns_typed_chain_server_config(
        self, write_yaml, valid_config_dict: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CONFIG_OVERRIDE", raising=False)
        path = write_yaml("config.yaml", valid_config_dict)

        config = load_config(str(path))

        assert isinstance(config, ChainServerConfig)
        assert config.memory_length == valid_config_dict["memory_length"]

    def test_invalid_yaml_surface_as_value_error(
        self, write_yaml, valid_config_dict: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CONFIG_OVERRIDE", raising=False)
        bad = dict(valid_config_dict)
        bad["llm_port"] = "not-a-url"
        path = write_yaml("config.yaml", bad)

        with pytest.raises(ValueError):
            load_config(str(path))


class TestRepoPromptContracts:
    def test_budget_only_browse_routes_to_chatter_for_clarification(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CONFIG_OVERRIDE", raising=False)
        config = load_config_with_override(
            str(REPO_ROOT / "shared/configs/chain_server/config.yaml")
        )

        routing_prompt = config["routing_prompt"]

        assert "UNDERSPECIFIED SHOPPING CONSTRAINTS -> chatter" in routing_prompt
        assert "show me anything under $100" in routing_prompt
        assert "show me dresses under $100" in routing_prompt
        assert "IMAGE ATTACHED is yes" in routing_prompt

    def test_chatter_asks_clarification_before_no_results(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CONFIG_OVERRIDE", raising=False)
        config = load_config_with_override(
            str(REPO_ROOT / "shared/configs/chain_server/config.yaml")
        )

        chatter_prompt = config["chatter_prompt"]

        assert "AMBIGUITY BEFORE RESULTS" in chatter_prompt
        assert "NO RESULTS AFTER RETRIEVAL" in chatter_prompt
        assert "ask one concise clarifying question" in chatter_prompt
