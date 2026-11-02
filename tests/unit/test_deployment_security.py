# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILES = (
    REPO_ROOT / "docker-compose.yaml",
    REPO_ROOT / "docker-compose-nim-local.yaml",
)
INTERNAL_APPLICATION_SERVICES = (
    "chain-server",
    "catalog-retriever",
    "memory-retriever",
    "rails",
)


def _compose(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _host_ip(port: object) -> str | None:
    if isinstance(port, str):
        parts = port.split(":")
        return parts[0] if len(parts) == 3 else None
    if isinstance(port, dict):
        value = port.get("host_ip")
        return str(value) if value is not None else None
    return None


def test_internal_application_services_are_not_published_to_the_host() -> None:
    services = _compose(REPO_ROOT / "docker-compose.yaml")["services"]

    for service_name in INTERNAL_APPLICATION_SERVICES:
        assert "ports" not in services[service_name], (
            f"{service_name} must remain reachable only on the Compose network; "
            "route browser traffic through nginx"
        )


def test_all_published_compose_ports_are_bound_to_loopback() -> None:
    for compose_file in COMPOSE_FILES:
        services = _compose(compose_file)["services"]
        for service_name, service in services.items():
            assert service.get("network_mode") != "host", (
                f"{compose_file.name}:{service_name} must not bypass network "
                "isolation with host networking"
            )
            for port in service.get("ports", []):
                assert _host_ip(port) == "127.0.0.1", (
                    f"{compose_file.name}:{service_name} publishes {port!r} without "
                    "an explicit loopback binding"
                )


def test_compose_services_do_not_claim_global_container_names() -> None:
    for compose_file in COMPOSE_FILES:
        services = _compose(compose_file)["services"]
        for service_name, service in services.items():
            assert "container_name" not in service, (
                f"{compose_file.name}:{service_name} must use a project-scoped "
                "Compose name"
            )


def test_chain_server_cors_is_limited_to_local_ui_origins() -> None:
    main_path = REPO_ROOT / "chain_server" / "src" / "main.py"
    tree = ast.parse(main_path.read_text(encoding="utf-8"))
    middleware_calls = (
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_middleware"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "CORSMiddleware"
    )
    middleware_call = next(middleware_calls)
    origins_keyword = next(
        keyword
        for keyword in middleware_call.keywords
        if keyword.arg == "allow_origins"
    )
    origins = ast.literal_eval(origins_keyword.value)

    assert origins == [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    assert "*" not in origins
