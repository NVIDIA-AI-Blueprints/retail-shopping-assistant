#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-container CPU and memory for Prometheus, read from the Docker API.

cAdvisor is the usual answer and does not work here. Docker on this host uses
the newer `overlayfs` storage driver, and cAdvisor cannot find a container's
read-write layer under it. That lookup happens while it registers the container,
so the failure takes CPU and memory down with it: every container falls back to
a raw cgroup handler keyed by id, with no compose labels, which is unreadable.
Disabling its disk metrics does not help, because registration still needs them.

So this asks Docker directly, which is where `docker stats` gets its numbers.
Roughly forty lines against a stable API, versus a tool that has to infer the
same facts from the filesystem.

Why it is needed at all: the engine's own metrics describe the engine. They
cannot distinguish a run limited by the GPUs from one limited by the chain
server, the retriever or Milvus -- and those call for opposite responses. A
ceiling with the application near idle means the model is the limit; the same
ceiling with a service pinned at 100% of a core means it is not.

Serves Prometheus text format on :9105/metrics.
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

SOCKET = os.environ.get("DOCKER_SOCKET", "/var/run/docker.sock")
PORT = int(os.environ.get("EXPORTER_PORT", "9105"))
INTERVAL = float(os.environ.get("SAMPLE_INTERVAL", "5"))


class _UnixConnection(http.client.HTTPConnection):
    """http.client over a unix socket, so no third-party dependency is needed."""

    def __init__(self, path: str) -> None:
        super().__init__("localhost")
        self._path = path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(30)
        self.sock.connect(self._path)


def _get(path: str):
    conn = _UnixConnection(SOCKET)
    try:
        conn.request("GET", path)
        return json.loads(conn.getresponse().read())
    finally:
        conn.close()


def _cpu_percent(stats: dict) -> float:
    """Percent of one core, matching what `docker stats` prints.

    Expressed against a single core rather than the whole machine, so 250%
    means two and a half cores. Absolute cores are what capacity planning
    needs; a percentage of a 128-core host would round everything to zero.
    """

    cpu, pre = stats.get("cpu_stats", {}), stats.get("precpu_stats", {})
    used = cpu.get("cpu_usage", {}).get("total_usage", 0) - \
        pre.get("cpu_usage", {}).get("total_usage", 0)
    elapsed = cpu.get("system_cpu_usage", 0) - pre.get("system_cpu_usage", 0)
    cores = cpu.get("online_cpus") or 1
    return (used / elapsed) * cores * 100 if elapsed > 0 and used > 0 else 0.0


def _memory_mib(stats: dict) -> float:
    """Working set, not raw usage.

    Raw usage counts page cache, which makes an idle container that once read a
    large file look busy. Subtracting inactive_file is what `docker stats` does
    and what makes the figure mean "memory this container actually needs".
    """

    mem = stats.get("memory_stats", {})
    usage = mem.get("usage", 0)
    inactive = (mem.get("stats") or {}).get("inactive_file", 0)
    return max(usage - inactive, 0) / 2**20


class Collector(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.lines: list[str] = []

    def sample(self) -> list[str]:
        out = [
            "# HELP container_cpu_percent CPU as a percentage of one core.",
            "# TYPE container_cpu_percent gauge",
            "# HELP container_memory_mib Working set memory in MiB.",
            "# TYPE container_memory_mib gauge",
        ]
        try:
            containers = _get("/containers/json")
        except Exception:
            return out

        for entry in containers:
            labels = entry.get("Labels") or {}
            service = labels.get("com.docker.compose.service")
            name = (entry.get("Names") or ["?"])[0].lstrip("/")
            if not service:
                # Anything not started by compose -- including this host's
                # Kubernetes pods -- is not part of the system under test and
                # would only add noise.
                continue
            project = labels.get("com.docker.compose.project", "")
            try:
                stats = _get(f"/containers/{entry['Id']}/stats?stream=false")
            except Exception:
                continue
            tags = f'service="{service}",project="{project}",name="{name}"'
            out.append(f"container_cpu_percent{{{tags}}} {_cpu_percent(stats):.2f}")
            out.append(f"container_memory_mib{{{tags}}} {_memory_mib(stats):.1f}")
        return out

    def run(self) -> None:
        while True:
            started = time.monotonic()
            self.lines = self.sample()
            # Each stats call blocks about a second server-side, so a busy host
            # can take longer to walk than the interval. Sleeping on what is
            # left, rather than a fixed interval, keeps it from falling behind.
            time.sleep(max(1.0, INTERVAL - (time.monotonic() - started)))


class Handler(BaseHTTPRequestHandler):
    collector: Collector

    def do_GET(self) -> None:
        body = ("\n".join(self.collector.lines) + "\n").encode()
        self.send_response(200 if self.path == "/metrics" else 404)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        return


def main() -> None:
    collector = Collector()
    collector.start()
    Handler.collector = collector
    print(f"docker stats exporter on :{PORT}/metrics, every {INTERVAL}s",
          flush=True)
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
