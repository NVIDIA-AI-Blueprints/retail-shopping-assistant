#!/usr/bin/env python3
"""Inspect model routing and deploy required services without exposing secrets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("SHARED_CONFIG_ROOT", str(REPO_ROOT / "shared" / "configs"))

from shared.model_config import (  # noqa: E402
    ModelConfigError,
    model_config_snapshot,
    resolve_model_config,
    validate_local_nim_env,
    validate_model_config,
)


def _run(command: Sequence[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    env = os.environ.copy()
    env["COMPOSE_DISABLE_ENV_FILE"] = "1"
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def _print_human(snapshot: dict) -> None:
    for role, model in snapshot["models"].items():
        if model["source"] == "disabled":
            print(f"{role}: source=disabled")
            continue

        key_state = "present" if model["api_key_present"] else "missing"
        key_detail = "none"
        if model["api_key_env"]:
            key_detail = f"{model['api_key_env']} (required, {key_state})"
        parts = [
            f"{role}: source={model['source']}",
            f"provider={model['provider']}",
            f"base_url={model['base_url']}",
            f"model={model['model']}",
            f"api_key_env={key_detail}",
        ]
        if model.get("local_service"):
            parts.append(f"local_service={model['local_service']}")
        print(", ".join(parts))

    services = snapshot["required_local_nim_services"]
    print("required_local_nim_services: " + (", ".join(services) if services else "[]"))
    required_env = snapshot["required_local_nim_env"] if services else []
    print("required_local_nim_env: " + (", ".join(required_env) if required_env else "[]"))


def show(args: argparse.Namespace) -> int:
    config = resolve_model_config()
    snapshot = model_config_snapshot(config)
    if args.json:
        print(json.dumps(snapshot, indent=2, sort_keys=True))
    else:
        _print_human(snapshot)
    if args.validate:
        validate_model_config(config)
        validate_local_nim_env(config)
    return 0


def deploy(args: argparse.Namespace) -> int:
    config = resolve_model_config()
    validate_model_config(config)
    validate_local_nim_env(config)
    snapshot = model_config_snapshot(config)
    _print_human(snapshot)

    services = list(config.required_local_nim_services)
    if services:
        _run(["docker", "compose", "-f", "docker-compose-nim-local.yaml", "up", "-d", *services])
    else:
        print("No local NIM services required by models.yaml.")

    command = ["docker", "compose", "-f", "docker-compose.yaml", "up", "-d"]
    if args.build:
        command.append("--build")
    _run(command)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    show_parser = subparsers.add_parser("show", help="Print resolved non-secret model config.")
    show_parser.add_argument("--json", action="store_true", help="Print JSON output.")
    show_parser.add_argument("--validate", action="store_true", help="Fail if required values are missing.")
    show_parser.set_defaults(func=show)

    deploy_parser = subparsers.add_parser("deploy", help="Deploy using shared/configs/models.yaml.")
    deploy_parser.add_argument("--build", action="store_true", help="Build app images before starting.")
    deploy_parser.set_defaults(func=deploy)

    args = parser.parse_args()
    try:
        return args.func(args)
    except (ModelConfigError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
