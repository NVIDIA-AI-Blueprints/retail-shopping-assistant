# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``chain_server.src.config``.

The config module drives every downstream agent's construction. These tests
exercise service YAML loading, model endpoint resolution, and the pydantic
validation contract directly, without touching the real container layout.
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
    load_config_data,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _clear_model_and_service_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "CATALOG_RETRIEVER_URL",
        "MEMORY_RETRIEVER_URL",
        "RAILS_URL",
        "CATALOG_SEARCH_TIMEOUT_SECONDS",
        "DEEPAGENTS_RECURSION_LIMIT",
        "MAX_CATALOG_SEARCHES_PER_TURN",
        "MAX_PRODUCT_DETAIL_READS_PER_TURN",
        "GROUNDING_REWRITE_ENABLED",
        "GROUNDING_REWRITE_MAX_EVIDENCE_CHARS",
        "GUARDRAILS_ENABLED",
        "LLM_BASE_URL",
        "LLM_MODEL",
        "VLM_BASE_URL",
        "VLM_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def write_yaml(tmp_path: Path):
    """Helper to drop a YAML config into a temporary directory."""

    def _write(name: str, data: Dict[str, Any]) -> Path:
        path = tmp_path / name
        path.write_text(yaml.safe_dump(data))
        return path

    return _write


class TestLoadConfigData:
    def test_returns_service_config(self, write_yaml, valid_config_dict: dict) -> None:
        base_path = write_yaml("config.yaml", valid_config_dict)

        result = load_config_data(str(base_path))

        assert result == valid_config_dict

    def test_raises_file_not_found_for_missing_base_config(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist.yaml"
        with pytest.raises(FileNotFoundError):
            load_config_data(str(missing))


class TestChainServerConfigValidation:
    def test_valid_dict_constructs_successfully(self, valid_config_dict: dict) -> None:
        config = ChainServerConfig(**valid_config_dict)

        assert config.llm_port == valid_config_dict["llm_port"]
        assert config.categories == valid_config_dict["categories"]
        assert config.multimodal is True
        assert config.vlm_enabled is False
        assert config.guardrails_enabled is True
        assert config.grounding_rewrite_enabled is True
        assert config.max_product_detail_reads_per_turn == 2
        assert config.grounding_rewrite_max_evidence_chars == 12000
        assert config.media_input.max_images_per_turn == 1
        assert config.media_input.max_videos_per_turn == 1

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

    @pytest.mark.parametrize("value", [0, -4])
    def test_deepagents_recursion_limit_must_be_positive(
        self, valid_config_dict: dict, value: int
    ) -> None:
        with pytest.raises(ValidationError):
            ChainServerConfig(
                **{**valid_config_dict, "deepagents_recursion_limit": value}
            )

    @pytest.mark.parametrize("value", [0, -4])
    def test_max_catalog_searches_per_turn_must_be_positive(
        self, valid_config_dict: dict, value: int
    ) -> None:
        with pytest.raises(ValidationError):
            ChainServerConfig(
                **{**valid_config_dict, "max_catalog_searches_per_turn": value}
            )

    @pytest.mark.parametrize("value", [0, -4])
    def test_max_product_detail_reads_per_turn_must_be_positive(
        self, valid_config_dict: dict, value: int
    ) -> None:
        with pytest.raises(ValidationError):
            ChainServerConfig(
                **{**valid_config_dict, "max_product_detail_reads_per_turn": value}
            )

    @pytest.mark.parametrize("value", [None, 30, 120.5])
    def test_catalog_search_timeout_accepts_none_or_positive_values(
        self, valid_config_dict: dict, value: float | None
    ) -> None:
        cfg = ChainServerConfig(
            **{**valid_config_dict, "catalog_search_timeout_seconds": value}
        )
        assert cfg.catalog_search_timeout_seconds == value

    @pytest.mark.parametrize("value", [0, -1])
    def test_catalog_search_timeout_rejects_non_positive_values(
        self, valid_config_dict: dict, value: float
    ) -> None:
        with pytest.raises(ValidationError):
            ChainServerConfig(
                **{**valid_config_dict, "catalog_search_timeout_seconds": value}
            )

    @pytest.mark.parametrize("value", [0, -1])
    def test_grounding_rewrite_evidence_window_must_be_positive(
        self, valid_config_dict: dict, value: int
    ) -> None:
        with pytest.raises(ValidationError):
            ChainServerConfig(
                **{**valid_config_dict, "grounding_rewrite_max_evidence_chars": value}
            )

    @pytest.mark.parametrize("field", ["agent_choices"])
    def test_empty_list_fields_are_rejected(
        self, valid_config_dict: dict, field: str
    ) -> None:
        with pytest.raises(ValidationError):
            ChainServerConfig(**{**valid_config_dict, field: []})

    def test_categories_are_optional_legacy_config(
        self, valid_config_dict: dict
    ) -> None:
        config_data = dict(valid_config_dict)
        del config_data["categories"]

        config = ChainServerConfig(**config_data)

        assert config.categories == []

    def test_extra_fields_are_forbidden(self, valid_config_dict: dict) -> None:
        with pytest.raises(ValidationError):
            ChainServerConfig(**valid_config_dict, unexpected_field="oops")


class TestLoadConfig:
    def test_returns_typed_chain_server_config(
        self, write_yaml, valid_config_dict: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _clear_model_and_service_env(monkeypatch)
        monkeypatch.setenv("SHARED_CONFIG_ROOT", str(REPO_ROOT / "shared/configs"))
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        path = write_yaml("config.yaml", valid_config_dict)

        config = load_config(str(path))

        assert isinstance(config, ChainServerConfig)
        assert config.memory_length == valid_config_dict["memory_length"]
        assert config.deepagents_recursion_limit == valid_config_dict["deepagents_recursion_limit"]
        assert config.max_catalog_searches_per_turn == valid_config_dict["max_catalog_searches_per_turn"]
        assert config.max_product_detail_reads_per_turn == valid_config_dict["max_product_detail_reads_per_turn"]
        assert config.guardrails_enabled is True
        assert config.llm_name == "nvidia/nemotron-3-super-120b-a12b"
        assert config.vlm_enabled is True

    @pytest.mark.parametrize(
        "raw_value,expected",
        [
            ("true", True),
            ("yes", True),
            ("on", True),
            ("1", True),
            ("false", False),
            ("no", False),
            ("off", False),
            ("0", False),
        ],
    )
    def test_guardrails_enabled_env_override_accepts_explicit_bools(
        self,
        write_yaml,
        valid_config_dict: dict,
        monkeypatch: pytest.MonkeyPatch,
        raw_value: str,
        expected: bool,
    ) -> None:
        _clear_model_and_service_env(monkeypatch)
        monkeypatch.setenv("SHARED_CONFIG_ROOT", str(REPO_ROOT / "shared/configs"))
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("GUARDRAILS_ENABLED", raw_value)
        path = write_yaml("config.yaml", valid_config_dict)

        config = load_config(str(path))

        assert config.guardrails_enabled is expected

    def test_guardrails_enabled_env_override_rejects_invalid_bool(
        self, write_yaml, valid_config_dict: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _clear_model_and_service_env(monkeypatch)
        monkeypatch.setenv("SHARED_CONFIG_ROOT", str(REPO_ROOT / "shared/configs"))
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("GUARDRAILS_ENABLED", "ture")
        path = write_yaml("config.yaml", valid_config_dict)

        with pytest.raises(ValueError, match="GUARDRAILS_ENABLED must be one of"):
            load_config(str(path))

    @pytest.mark.parametrize(
        "raw_value,expected",
        [
            ("true", True),
            ("false", False),
        ],
    )
    def test_grounding_rewrite_enabled_env_override_accepts_explicit_bools(
        self,
        write_yaml,
        valid_config_dict: dict,
        monkeypatch: pytest.MonkeyPatch,
        raw_value: str,
        expected: bool,
    ) -> None:
        _clear_model_and_service_env(monkeypatch)
        monkeypatch.setenv("SHARED_CONFIG_ROOT", str(REPO_ROOT / "shared/configs"))
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("GROUNDING_REWRITE_ENABLED", raw_value)
        path = write_yaml("config.yaml", valid_config_dict)

        config = load_config(str(path))

        assert config.grounding_rewrite_enabled is expected

    def test_grounding_rewrite_max_evidence_env_override(
        self, write_yaml, valid_config_dict: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _clear_model_and_service_env(monkeypatch)
        monkeypatch.setenv("SHARED_CONFIG_ROOT", str(REPO_ROOT / "shared/configs"))
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("GROUNDING_REWRITE_MAX_EVIDENCE_CHARS", "6400")
        path = write_yaml("config.yaml", valid_config_dict)

        config = load_config(str(path))

        assert config.grounding_rewrite_max_evidence_chars == 6400

    def test_max_product_detail_reads_env_override(
        self, write_yaml, valid_config_dict: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _clear_model_and_service_env(monkeypatch)
        monkeypatch.setenv("SHARED_CONFIG_ROOT", str(REPO_ROOT / "shared/configs"))
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("MAX_PRODUCT_DETAIL_READS_PER_TURN", "2")
        path = write_yaml("config.yaml", valid_config_dict)

        config = load_config(str(path))

        assert config.max_product_detail_reads_per_turn == 2

    def test_invalid_yaml_surface_as_value_error(
        self, write_yaml, valid_config_dict: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _clear_model_and_service_env(monkeypatch)
        monkeypatch.setenv("SHARED_CONFIG_ROOT", str(REPO_ROOT / "shared/configs"))
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        bad = dict(valid_config_dict)
        bad["memory_port"] = "not-a-url"
        path = write_yaml("config.yaml", bad)

        with pytest.raises(ValueError):
            load_config(str(path))


class TestRepoPromptContracts:
    def test_budget_only_browse_routes_to_chatter_for_clarification(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = load_config_data(
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
        config = load_config_data(
            str(REPO_ROOT / "shared/configs/chain_server/config.yaml")
        )

        chatter_prompt = config["chatter_prompt"]

        assert "AMBIGUITY BEFORE RESULTS" in chatter_prompt
        assert "NO RESULTS AFTER RETRIEVAL" in chatter_prompt
        assert "ask one concise clarifying question" in chatter_prompt
