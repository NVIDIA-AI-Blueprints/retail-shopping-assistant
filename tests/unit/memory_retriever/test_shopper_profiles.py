# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused coverage for the immutable representative-shopper registry."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from memory_retriever.src import main as memory_main
from memory_retriever.src.shopper_profiles import (
    ShopperProfileBootstrapError,
    load_shopper_profile_seed,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = REPO_ROOT / "shared" / "configs"
SEED_PATH = CONFIG_ROOT / "memory_retriever" / "shopper_profiles.json"

EXPECTED_PROFILES = [
    {
        "shopper_profile_id": "shopper_alex",
        "display_name": "Alex",
        "shopper_type": "occasion_driven_explorer",
        "behavior": (
            "Gives occasion and vibe first, answers concise clarification, "
            "then asks for a complete look."
        ),
        "zipcode": "98101",
    },
    {
        "shopper_profile_id": "shopper_casey",
        "display_name": "Casey",
        "shopper_type": "strict_budget_style_mixer",
        "behavior": (
            "Treats budget and style as equally important, asks for swaps, "
            "and rejects over-budget bundles."
        ),
        "zipcode": "85004",
    },
    {
        "shopper_profile_id": "shopper_jordan",
        "display_name": "Jordan",
        "shopper_type": "impatient_decisive",
        "behavior": (
            "Uses shorthand and pronouns after product comparisons, then "
            "changes their mind about cart contents."
        ),
        "zipcode": "10001",
    },
    {
        "shopper_profile_id": "shopper_morgan",
        "display_name": "Morgan",
        "shopper_type": "skeptical_researcher",
        "behavior": (
            "Probes for material, care burden, and repeated-wear practicality "
            "before choosing."
        ),
        "zipcode": "60601",
    },
    {
        "shopper_profile_id": "shopper_riley",
        "display_name": "Riley",
        "shopper_type": "iterative_refiner",
        "behavior": (
            "Accepts part of the outfit, rejects one constraint, and expects "
            "the assistant to revise intelligently."
        ),
        "zipcode": "33130",
    },
]


@pytest.fixture
def profile_service(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    test_engine = memory_main.build_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
    )
    session_factory = sessionmaker(bind=test_engine, expire_on_commit=False)
    monkeypatch.setattr(memory_main, "engine", test_engine)
    monkeypatch.setattr(memory_main, "SessionLocal", session_factory)
    monkeypatch.setenv("SHARED_CONFIG_ROOT", str(CONFIG_ROOT))

    with TestClient(memory_main.app) as client:
        yield client

    memory_main.Base.metadata.drop_all(bind=test_engine)
    test_engine.dispose()


def test_startup_migrates_bootstraps_and_lists_exact_profiles(
    profile_service: TestClient,
) -> None:
    response = profile_service.get("/shopper-profiles")

    assert response.status_code == 200
    assert response.json() == EXPECTED_PROFILES
    assert all(isinstance(row["zipcode"], str) for row in response.json())
    with memory_main.engine.connect() as connection:
        assert connection.exec_driver_sql(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).scalars().all() == [1, 2, 3, 4, 5, 6, 7, 8]


def test_get_profile_is_typed_and_unknown_or_malformed_ids_fail(
    profile_service: TestClient,
) -> None:
    found = profile_service.get("/shopper-profiles/shopper_morgan")
    missing = profile_service.get("/shopper-profiles/shopper_unknown")
    malformed = profile_service.get("/shopper-profiles/not.valid")

    assert found.status_code == 200
    assert found.json() == EXPECTED_PROFILES[3]
    assert missing.status_code == 404
    assert missing.json()["detail"] == "Shopper profile not found"
    assert malformed.status_code == 422


@pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
def test_profile_registry_has_no_write_endpoints(
    profile_service: TestClient,
    method: str,
) -> None:
    response = profile_service.request(
        method.upper(),
        "/shopper-profiles/shopper_alex",
        json=EXPECTED_PROFILES[0],
    )

    assert response.status_code == 405


def test_rebootstrap_is_an_exact_no_op(profile_service: TestClient) -> None:
    with memory_main.SessionLocal() as db:
        before = [
            (
                row.shopper_profile_id,
                row.display_name,
                row.shopper_type,
                row.behavior,
                row.zipcode,
            )
            for row in db.query(memory_main.ShopperProfile)
            .order_by(memory_main.ShopperProfile.shopper_profile_id)
            .all()
        ]

    memory_main._bootstrap_shopper_profiles(seed_path=str(SEED_PATH))

    with memory_main.SessionLocal() as db:
        after = [
            (
                row.shopper_profile_id,
                row.display_name,
                row.shopper_type,
                row.behavior,
                row.zipcode,
            )
            for row in db.query(memory_main.ShopperProfile)
            .order_by(memory_main.ShopperProfile.shopper_profile_id)
            .all()
        ]
    assert after == before


def test_rebootstrap_restores_a_missing_managed_row(
    profile_service: TestClient,
) -> None:
    with memory_main.SessionLocal() as db:
        db.query(memory_main.ShopperProfile).filter_by(
            shopper_profile_id="shopper_alex"
        ).delete()
        db.commit()

    memory_main._bootstrap_shopper_profiles(seed_path=str(SEED_PATH))

    assert profile_service.get("/shopper-profiles").json() == EXPECTED_PROFILES


def test_conflict_rolls_back_missing_row_insertion(
    profile_service: TestClient,
) -> None:
    with memory_main.SessionLocal() as db:
        db.query(memory_main.ShopperProfile).filter_by(
            shopper_profile_id="shopper_alex"
        ).delete()
        casey = db.query(memory_main.ShopperProfile).filter_by(
            shopper_profile_id="shopper_casey"
        ).one()
        casey.display_name = "Changed Casey"
        db.commit()

    with pytest.raises(
        ShopperProfileBootstrapError,
        match="conflicts with its seed",
    ):
        memory_main._bootstrap_shopper_profiles(seed_path=str(SEED_PATH))

    with memory_main.SessionLocal() as db:
        assert db.query(memory_main.ShopperProfile).count() == 4
        assert (
            db.query(memory_main.ShopperProfile)
            .filter_by(shopper_profile_id="shopper_alex")
            .first()
            is None
        )


def test_extra_registry_row_fails_bootstrap(
    profile_service: TestClient,
) -> None:
    with memory_main.SessionLocal() as db:
        db.add(
            memory_main.ShopperProfile(
                shopper_profile_id="shopper_extra",
                display_name="Extra",
                shopper_type="extra_type",
                behavior="Unexpected managed profile.",
                zipcode="02108",
            )
        )
        db.commit()

    with pytest.raises(
        ShopperProfileBootstrapError,
        match="unmanaged rows",
    ):
        memory_main._bootstrap_shopper_profiles(seed_path=str(SEED_PATH))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda rows: rows[:-1],
        lambda rows: [
            {**rows[0], "shopper_profile_id": "shopper_wrong"},
            *rows[1:],
        ],
        lambda rows: [*rows[:-1], {**rows[-1], "shopper_type": "wrong_type"}],
        lambda rows: [*rows, dict(rows[0])],
        lambda rows: [{**rows[0], "zipcode": "1234"}, *rows[1:]],
        lambda rows: [{**rows[0], "display_name": " Alex"}, *rows[1:]],
        lambda rows: [
            {**rows[0], "behavior": "Two-line\nbehavior"},
            *rows[1:],
        ],
        lambda rows: [{**rows[0], "unexpected": True}, *rows[1:]],
    ],
)
def test_invalid_seed_manifests_fail_closed(
    tmp_path: Path,
    mutate,
) -> None:
    rows = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(mutate(rows)), encoding="utf-8")

    with pytest.raises(ShopperProfileBootstrapError):
        load_shopper_profile_seed(path)


def test_database_constraints_reject_non_ascii_zipcode(
    profile_service: TestClient,
) -> None:
    with memory_main.SessionLocal() as db:
        db.add(
            memory_main.ShopperProfile(
                shopper_profile_id="shopper_invalid",
                display_name="Invalid",
                shopper_type="invalid_type",
                behavior="Invalid ZIP should not persist.",
                zipcode="１２３４５",
            )
        )
        with pytest.raises(IntegrityError):
            db.commit()


def test_seed_behavior_matches_committed_evaluation_profiles() -> None:
    seed = load_shopper_profile_seed(SEED_PATH)
    expected_pairs = {
        profile.shopper_type: profile.behavior for profile in seed
    }

    actual_pairs: dict[str, str] = {}
    for relative_path in (
        "tests/evaluation/datasets/style_guide/scenarios.yaml",
        "tests/evaluation/datasets/text_shopping/scenarios.yaml",
    ):
        dataset = yaml.safe_load(
            (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        )
        for scenario in dataset["scenarios"]:
            behavior = scenario.get("shopper_behavior") or {}
            shopper_type = behavior.get("type")
            if shopper_type in expected_pairs:
                actual_pairs[shopper_type] = behavior["notes"]

    assert actual_pairs == expected_pairs
