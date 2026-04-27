#!/usr/bin/env python3
"""Local runner for Retail Shopping Assistant.

This script keeps the app services outside containers while using Docker only for
the Milvus infrastructure services from docker-compose.yaml.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from http.client import HTTPException
from pathlib import Path
from typing import Iterable
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = REPO_ROOT / ".local-run"
PID_DIR = RUN_DIR / "pids"
LOG_DIR = RUN_DIR / "logs"
STAMP_DIR = RUN_DIR / "install-stamps"
DEV_VENV_DIR = RUN_DIR / "dev-venv"
CONFIG_NAME = "config-local.yaml"
DEV_REQUIREMENTS = REPO_ROOT / "tests" / "requirements-dev.txt"
INTEGRATION_REQUIREMENTS = REPO_ROOT / "tests" / "requirements.txt"
UI_PUBLIC_IMAGES = REPO_ROOT / "ui" / "public" / "images"
SHARED_IMAGES = REPO_ROOT / "shared" / "images"

INFRA_SERVICES = ("etcd", "minio", "milvus")
MILVUS_GRPC_PORT = 19530
MILVUS_HEALTH_URL = "http://localhost:9091/healthz"

PYTHON_SERVICES = {
    "memory-retriever": {
        "service_dir": REPO_ROOT / "memory_retriever",
        "requirements": REPO_ROOT / "memory_retriever" / "requirements.txt",
        "module": "src.main:app",
        "port": 8011,
        "health_url": "http://localhost:8011/health",
        "pythonpath": [REPO_ROOT / "memory_retriever"],
    },
    "guardrails": {
        "service_dir": REPO_ROOT / "guardrails",
        "requirements": REPO_ROOT / "guardrails" / "src" / "requirements.txt",
        "module": "src.main:app",
        "port": 8012,
        "health_url": None,
        "pythonpath": [REPO_ROOT / "guardrails" / "src", REPO_ROOT / "guardrails"],
    },
    "catalog-retriever": {
        "service_dir": REPO_ROOT / "catalog_retriever",
        "requirements": REPO_ROOT / "catalog_retriever" / "requirements.txt",
        "module": "src.main:app",
        "port": 8010,
        "health_url": "http://localhost:8010/health",
        "pythonpath": [REPO_ROOT / "catalog_retriever"],
    },
    "chain-server": {
        "service_dir": REPO_ROOT / "chain_server",
        "requirements": REPO_ROOT / "chain_server" / "requirements.txt",
        "module": "src.main:app",
        "port": 8009,
        "health_url": "http://localhost:8009/health",
        "pythonpath": [REPO_ROOT / "chain_server"],
    },
}

UI_SERVICE = "ui"
ALL_LOCAL_SERVICES = (*PYTHON_SERVICES.keys(), UI_SERVICE)


def ensure_dirs() -> None:
    for path in (RUN_DIR, PID_DIR, LOG_DIR, STAMP_DIR):
        path.mkdir(parents=True, exist_ok=True)


def shell_join(command: Iterable[str]) -> str:
    return shlex.join(str(part) for part in command)


def run(command: list[str], *, cwd: Path = REPO_ROOT, env: dict[str, str] | None = None) -> None:
    print(f"$ {shell_join(command)}")
    subprocess.run(command, cwd=cwd, env=env, check=True)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def venv_python(service_dir: Path) -> Path:
    return service_dir / "venv" / "bin" / "python"


def ensure_virtualenv(venv_dir: Path) -> Path:
    python_bin = venv_dir / "bin" / "python"
    if not python_bin.exists():
        run([sys.executable, "-m", "venv", str(venv_dir)])
    return python_bin


def ensure_python_deps(name: str, spec: dict[str, object]) -> None:
    service_dir = spec["service_dir"]
    requirements = spec["requirements"]
    assert isinstance(service_dir, Path)
    assert isinstance(requirements, Path)

    python_bin = ensure_virtualenv(service_dir / "venv")

    req_hash = file_sha256(requirements)
    stamp = STAMP_DIR / f"{name}.requirements.sha256"
    if not stamp.exists() or stamp.read_text().strip() != req_hash:
        run([str(python_bin), "-m", "pip", "install", "-r", str(requirements)], cwd=service_dir)
        stamp.write_text(f"{req_hash}\n")


def ensure_python_dev_deps(*, include_integration: bool) -> None:
    python_bin = ensure_virtualenv(DEV_VENV_DIR)
    requirement_files = [DEV_REQUIREMENTS]
    if include_integration:
        requirement_files.append(INTEGRATION_REQUIREMENTS)

    for requirements in requirement_files:
        req_hash = file_sha256(requirements)
        stamp = STAMP_DIR / f"dev.{requirements.name}.sha256"
        if not stamp.exists() or stamp.read_text().strip() != req_hash:
            run([str(python_bin), "-m", "pip", "install", "-r", str(requirements)], cwd=REPO_ROOT)
            stamp.write_text(f"{req_hash}\n")
        else:
            print(f"{requirements.relative_to(REPO_ROOT)} already installed in {DEV_VENV_DIR.relative_to(REPO_ROOT)}")

    print(f"Python dev environment ready: {python_bin.relative_to(REPO_ROOT)}")


def ensure_ui_deps() -> None:
    ui_dir = REPO_ROOT / "ui"
    if not (ui_dir / "node_modules").exists():
        run(["npm", "install"], cwd=ui_dir)


def ensure_ui_image_assets() -> None:
    """Expose shared product images from the React dev server public root."""
    if not SHARED_IMAGES.is_dir():
        raise SystemExit(f"Shared image directory is missing: {SHARED_IMAGES}")

    if UI_PUBLIC_IMAGES.is_symlink():
        if UI_PUBLIC_IMAGES.resolve() == SHARED_IMAGES.resolve():
            return
        UI_PUBLIC_IMAGES.unlink()
    elif UI_PUBLIC_IMAGES.exists():
        raise SystemExit(
            f"{UI_PUBLIC_IMAGES.relative_to(REPO_ROOT)} exists and is not the expected symlink. "
            f"Move it aside before starting the local UI."
        )

    UI_PUBLIC_IMAGES.symlink_to(SHARED_IMAGES, target_is_directory=True)
    print(f"linked {UI_PUBLIC_IMAGES.relative_to(REPO_ROOT)} -> {SHARED_IMAGES.relative_to(REPO_ROOT)}")


def parse_nim_host(raw_host: str) -> tuple[str, str]:
    host = raw_host.strip().rstrip("/")
    if not host:
        raise SystemExit("NIM host cannot be empty.")
    if "://" not in host:
        host = f"http://{host}"

    parsed = urlparse(host)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise SystemExit(f"Invalid NIM host URL: {raw_host}")

    hostname = parsed.hostname
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    return parsed.scheme, hostname


def nim_endpoint(raw_host: str, port: int) -> str:
    scheme, host = parse_nim_host(raw_host)
    return f"{scheme}://{host}:{port}/v1"


def get_nim_host(args: argparse.Namespace, *, required: bool) -> str | None:
    if getattr(args, "nim_host", None):
        return args.nim_host
    if sys.stdin.isatty():
        prompt = "Remote NIM host, without per-model port (example: http://NIM_HOST): "
        value = input(prompt).strip()
        if value:
            return value
    if required:
        raise SystemExit(
            "Missing NIM host. Ask the user: What is the remote NIM host URL? "
            "Use the base host without per-model ports, for example http://NIM_HOST. "
            "Then re-run with --nim-host http://HOST."
        )
    return None


def config_paths() -> list[Path]:
    return [
        REPO_ROOT / "shared" / "configs" / "chain_server" / CONFIG_NAME,
        REPO_ROOT / "shared" / "configs" / "catalog_retriever" / CONFIG_NAME,
        REPO_ROOT / "shared" / "configs" / "rails" / CONFIG_NAME,
    ]


def local_configs_exist() -> bool:
    return all(path.exists() for path in config_paths())


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"wrote {path.relative_to(REPO_ROOT)}")


def configure(args: argparse.Namespace) -> None:
    nim_host = get_nim_host(args, required=True)
    assert nim_host is not None

    llm_url = nim_endpoint(nim_host, 8000)
    text_embed_url = nim_endpoint(nim_host, 8001)
    image_embed_url = nim_endpoint(nim_host, 8002)
    content_url = nim_endpoint(nim_host, 8003)
    topic_url = nim_endpoint(nim_host, 8004)
    data_source = REPO_ROOT / "shared" / "data" / "products_extended.csv"

    write_text(
        REPO_ROOT / "shared" / "configs" / "chain_server" / CONFIG_NAME,
        "\n".join(
            [
                "# Generated by skills/retail-local-runner/scripts/local_runner.py",
                f'llm_port: "{llm_url}"',
                'llm_name: "nvidia/nemotron-3-super-120b-a12b"',
                'retriever_port: "http://localhost:8010"',
                'memory_port: "http://localhost:8011"',
                'rails_port: "http://localhost:8012"',
                "",
            ]
        ),
    )
    write_text(
        REPO_ROOT / "shared" / "configs" / "catalog_retriever" / CONFIG_NAME,
        "\n".join(
            [
                "# Generated by skills/retail-local-runner/scripts/local_runner.py",
                f'text_embed_port: "{text_embed_url}"',
                'text_model_name: "nvidia/nv-embedqa-e5-v5"',
                f'image_embed_port: "{image_embed_url}"',
                'image_model_name: "nvidia/nvclip"',
                'db_port: "http://localhost:19530"',
                f'data_source: "{data_source}"',
                "",
            ]
        ),
    )
    write_text(
        REPO_ROOT / "shared" / "configs" / "rails" / CONFIG_NAME,
        "\n".join(
            [
                "# Generated by skills/retail-local-runner/scripts/local_runner.py",
                "models:",
                "  - type: content_safety",
                "    parameters:",
                f'      base_url: "{content_url}"',
                "  - type: topic_control",
                "    parameters:",
                f'      base_url: "{topic_url}"',
                "",
            ]
        ),
    )


def base_env() -> dict[str, str]:
    env = os.environ.copy()
    local_key = env.get("NGC_API_KEY", "local-nim")
    env.setdefault("LLM_API_KEY", local_key)
    env.setdefault("EMBED_API_KEY", local_key)
    env.setdefault("RAIL_API_KEY", local_key)
    env.setdefault("NVIDIA_API_KEY", env.get("RAIL_API_KEY", local_key))
    env["CONFIG_OVERRIDE"] = CONFIG_NAME
    env["SHARED_ROOT"] = str(REPO_ROOT / "shared")
    env["SHARED_CONFIG_ROOT"] = str(REPO_ROOT / "shared" / "configs")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def with_pythonpath(env: dict[str, str], paths: list[Path]) -> dict[str, str]:
    new_env = env.copy()
    path_parts = [str(path) for path in paths]
    if new_env.get("PYTHONPATH"):
        path_parts.append(new_env["PYTHONPATH"])
    new_env["PYTHONPATH"] = os.pathsep.join(path_parts)
    return new_env


def pid_file(service: str) -> Path:
    return PID_DIR / f"{service}.pid"


def log_file(service: str) -> Path:
    return LOG_DIR / f"{service}.log"


def read_pid(service: str) -> int | None:
    path = pid_file(service)
    if not path.exists():
        return None
    try:
        return int(path.read_text().strip())
    except ValueError:
        return None


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def tracked_running(service: str) -> bool:
    pid = read_pid(service)
    return bool(pid and pid_alive(pid))


def port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) == 0


def wait_port(service: str, port: int, timeout: int) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if port_open(port):
            return True
        if not tracked_running(service):
            return False
        time.sleep(1)
    return False


def wait_http(service: str, url: str, timeout: int) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=2) as response:
                if response.status < 500:
                    return True
        except (HTTPException, OSError, TimeoutError, URLError):
            pass
        if service in ALL_LOCAL_SERVICES and not tracked_running(service):
            return False
        time.sleep(2)
    return False


def local_milvus_available(timeout: int = 3) -> bool:
    return port_open(MILVUS_GRPC_PORT) and wait_http("milvus", MILVUS_HEALTH_URL, timeout)


def last_log_lines(service: str, lines: int = 30) -> str:
    path = log_file(service)
    if not path.exists():
        return f"No log file found for {service}."
    content = path.read_text(errors="replace").splitlines()
    return "\n".join(content[-lines:])


def assert_port_available(service: str, port: int) -> None:
    if tracked_running(service):
        return
    if port_open(port):
        raise SystemExit(f"Port {port} is already in use, but {service} is not tracked in {PID_DIR}.")


def preflight_ports_available() -> None:
    for service, spec in PYTHON_SERVICES.items():
        port = spec["port"]
        assert isinstance(port, int)
        assert_port_available(service, port)
    assert_port_available(UI_SERVICE, 3000)


def start_process(
    service: str,
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    port: int | None = None,
) -> None:
    if tracked_running(service):
        print(f"{service} already running with pid {read_pid(service)}")
        return

    if port is not None:
        assert_port_available(service, port)

    ensure_dirs()
    pid_file(service).unlink(missing_ok=True)
    log_path = log_file(service)
    log_handle = log_path.open("ab")
    log_handle.write(f"\n\n=== starting {service}: {shell_join(command)} ===\n".encode())
    log_handle.flush()

    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    pid_file(service).write_text(f"{process.pid}\n")
    time.sleep(1)
    if not pid_alive(process.pid):
        raise SystemExit(f"{service} exited during startup.\n{last_log_lines(service)}")
    print(f"started {service} pid={process.pid} log={log_path.relative_to(REPO_ROOT)}")


def stop_process(service: str, timeout: int = 15) -> None:
    pid = read_pid(service)
    if not pid:
        print(f"{service}: no tracked pid")
        return
    if not pid_alive(pid):
        print(f"{service}: stale pid {pid}")
        pid_file(service).unlink(missing_ok=True)
        return

    print(f"stopping {service} pid={pid}")
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pid_file(service).unlink(missing_ok=True)
        return

    deadline = time.time() + timeout
    while time.time() < deadline:
        if not pid_alive(pid):
            pid_file(service).unlink(missing_ok=True)
            return
        time.sleep(0.5)

    print(f"{service}: SIGTERM timed out; sending SIGKILL")
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    pid_file(service).unlink(missing_ok=True)


def start_infra() -> None:
    if local_milvus_available():
        print(f"Reusing existing local Milvus at localhost:{MILVUS_GRPC_PORT}.")
        return

    run(["docker", "compose", "-f", "docker-compose.yaml", "up", "-d", *INFRA_SERVICES])
    if not wait_http("milvus", MILVUS_HEALTH_URL, 120):
        raise SystemExit(f"Milvus did not become healthy at {MILVUS_HEALTH_URL}")


def stop_infra() -> None:
    run(["docker", "compose", "-f", "docker-compose.yaml", "stop", "milvus", "minio", "etcd"])


def install_dev(args: argparse.Namespace) -> None:
    ensure_python_dev_deps(include_integration=args.include_integration)


def start_python_service(name: str, spec: dict[str, object], *, skip_install: bool) -> None:
    service_dir = spec["service_dir"]
    module = spec["module"]
    port = spec["port"]
    pythonpath = spec["pythonpath"]
    assert isinstance(service_dir, Path)
    assert isinstance(module, str)
    assert isinstance(port, int)
    assert isinstance(pythonpath, list)

    if not skip_install:
        ensure_python_deps(name, spec)

    env = with_pythonpath(base_env(), pythonpath)
    command = [
        str(venv_python(service_dir)),
        "-m",
        "uvicorn",
        module,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
    ]
    start_process(name, command, cwd=service_dir, env=env, port=port)


def start_ui(*, skip_install: bool) -> None:
    ensure_ui_image_assets()
    if not skip_install:
        ensure_ui_deps()
    env = os.environ.copy()
    env["PORT"] = "3000"
    env["BROWSER"] = "none"
    env["REACT_APP_API_BASE_URL"] = "http://localhost:8009"
    env.setdefault("CI", "true")
    start_process(UI_SERVICE, ["npm", "start"], cwd=REPO_ROOT / "ui", env=env, port=3000)


def start(args: argparse.Namespace) -> None:
    ensure_dirs()
    if args.nim_host or not local_configs_exist():
        configure(args)

    if not local_configs_exist():
        raise SystemExit("Local config files are missing. Run configure first.")

    preflight_ports_available()
    start_infra()

    start_python_service("memory-retriever", PYTHON_SERVICES["memory-retriever"], skip_install=args.skip_install)
    if not wait_http("memory-retriever", "http://localhost:8011/health", 30):
        raise SystemExit(f"memory-retriever did not become healthy.\n{last_log_lines('memory-retriever')}")

    start_python_service("guardrails", PYTHON_SERVICES["guardrails"], skip_install=args.skip_install)
    if not wait_port("guardrails", 8012, 120):
        raise SystemExit(f"guardrails did not open port 8012.\n{last_log_lines('guardrails')}")

    start_python_service("catalog-retriever", PYTHON_SERVICES["catalog-retriever"], skip_install=args.skip_install)
    if not wait_http("catalog-retriever", "http://localhost:8010/health", args.catalog_timeout):
        raise SystemExit(f"catalog-retriever did not become healthy.\n{last_log_lines('catalog-retriever')}")

    start_python_service("chain-server", PYTHON_SERVICES["chain-server"], skip_install=args.skip_install)
    if not wait_http("chain-server", "http://localhost:8009/health", 60):
        raise SystemExit(f"chain-server did not become healthy.\n{last_log_lines('chain-server')}")

    start_ui(skip_install=args.skip_install)
    if not wait_http(UI_SERVICE, "http://localhost:3000", 90):
        raise SystemExit(f"ui did not become reachable.\n{last_log_lines(UI_SERVICE)}")

    print("Local Retail Shopping Assistant is running at http://localhost:3000")


def stop(_args: argparse.Namespace) -> None:
    for service in reversed(ALL_LOCAL_SERVICES):
        stop_process(service)
    stop_infra()


def status(_args: argparse.Namespace) -> None:
    ensure_dirs()
    print("Local processes:")
    for service in ALL_LOCAL_SERVICES:
        pid = read_pid(service)
        alive = bool(pid and pid_alive(pid))
        port = PYTHON_SERVICES.get(service, {}).get("port") if service != UI_SERVICE else 3000
        health_url = PYTHON_SERVICES.get(service, {}).get("health_url") if service != UI_SERVICE else "http://localhost:3000"
        port_state = f"port {port} open={port_open(port)}" if isinstance(port, int) else "port n/a"
        health_state = ""
        if isinstance(health_url, str):
            health_state = f" health={wait_http(service, health_url, 1)}"
        print(f"  {service}: pid={pid or '-'} alive={alive} {port_state}{health_state}")

    print("\nMilvus infra:")
    print(
        f"  endpoint localhost:{MILVUS_GRPC_PORT} "
        f"open={port_open(MILVUS_GRPC_PORT)} health={wait_http('milvus', MILVUS_HEALTH_URL, 1)}"
    )
    subprocess.run(["docker", "compose", "-f", "docker-compose.yaml", "ps", *INFRA_SERVICES], cwd=REPO_ROOT)


def logs(args: argparse.Namespace) -> None:
    ensure_dirs()
    services = [args.service] if args.service else list(ALL_LOCAL_SERVICES)
    for service in services:
        print(f"\n=== {service} ({log_file(service).relative_to(REPO_ROOT)}) ===")
        print(last_log_lines(service, args.lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Retail Shopping Assistant locally.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    configure_parser = subparsers.add_parser("configure", help="Write local config overrides for remote NIMs.")
    configure_parser.add_argument("--nim-host", help="Remote NIM host URL, for example http://NIM_HOST")
    configure_parser.set_defaults(func=configure)

    install_dev_parser = subparsers.add_parser("install-dev", help="Create a repo-local Python dev venv and install test/dev packages.")
    install_dev_parser.add_argument(
        "--include-integration",
        action="store_true",
        help="Also install tests/requirements.txt for integration scripts.",
    )
    install_dev_parser.set_defaults(func=install_dev)

    start_parser = subparsers.add_parser("start", help="Start local app services plus local Milvus infra.")
    start_parser.add_argument("--nim-host", help="Remote NIM host URL. Rewrites local config before starting.")
    start_parser.add_argument("--skip-install", action="store_true", help="Do not install Python or UI dependencies.")
    start_parser.add_argument("--catalog-timeout", type=int, default=600, help="Seconds to wait for catalog startup.")
    start_parser.set_defaults(func=start)

    stop_parser = subparsers.add_parser("stop", help="Stop tracked local services and local Milvus infra.")
    stop_parser.set_defaults(func=stop)

    status_parser = subparsers.add_parser("status", help="Show process, port, health, and infra status.")
    status_parser.set_defaults(func=status)

    logs_parser = subparsers.add_parser("logs", help="Show recent local service logs.")
    logs_parser.add_argument("--service", choices=ALL_LOCAL_SERVICES, help="Limit logs to one service.")
    logs_parser.add_argument("--lines", type=int, default=80, help="Number of log lines to show.")
    logs_parser.set_defaults(func=logs)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    ensure_dirs()
    args.func(args)


if __name__ == "__main__":
    main()
