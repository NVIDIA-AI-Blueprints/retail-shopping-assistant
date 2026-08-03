# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Convert tests/evaluation datasets (scenarios.yaml) into NeMo Gym JSONL.

One Gym task = one scenario. The Challenger (agent harness) generates the turns at
run time, so the row's `input` is only a seed; the full scenario travels in
`verifier_metadata` for the Challenger loop and the Judge.

    python tests/evaluation/gym/build_dataset.py                 # default: text_shopping
    python tests/evaluation/gym/build_dataset.py --dataset image_shopping
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_ROOT = os.path.normpath(os.path.join(HERE, ".."))          # tests/evaluation
DATASETS_DIR = os.path.join(EVAL_ROOT, "datasets")
OUT_DIR = os.path.join(HERE, "build")


_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
         ".webp": "image/webp", ".mp4": "video/mp4", ".mov": "video/quicktime"}
_VIDEO_EXT = {".mp4", ".mov"}


def _load_media(dataset: str, image_id: str) -> dict | None:
    """Look up assets/<image_id> (image or video) -> {type, mime_type, data(b64)}."""
    assets = os.path.join(DATASETS_DIR, dataset, "assets")
    stems = {image_id, image_id.replace("_", "-"), image_id.replace("-", "_")}
    for stem in stems:
        for ext, mime in _MIME.items():
            for path in glob.glob(os.path.join(assets, stem + ext)):
                data = base64.b64encode(open(path, "rb").read()).decode("ascii")
                kind = "video" if ext in _VIDEO_EXT else "image"
                return {"type": kind, "mime_type": mime, "data": data}
    return None


def build(dataset: str) -> str:
    src = os.path.join(DATASETS_DIR, dataset, "scenarios.yaml")
    doc = yaml.safe_load(open(src)) or {}
    scenarios = doc.get("scenarios", []) or []
    if not scenarios:
        raise SystemExit(f"No scenarios in {src}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{dataset}.jsonl")
    with open(out, "w") as f:
        for i, sc in enumerate(scenarios):
            meta = {"dataset": dataset, "scenario": sc}
            image_id = sc.get("image_id")
            if image_id:
                media = _load_media(dataset, image_id)
                if media:
                    meta["media"] = [media]      # Challenger sends on turn 1 (image or video)
                else:
                    print(f"  WARN {sc.get('id')}: asset for image_id '{image_id}' not found")
            row = {
                "id": i,
                # seed only — the Challenger generates real turns from `scenario`
                "responses_create_params": {
                    "input": [{"role": "user", "content": sc.get("shopper_goal", sc.get("brief", ""))}]
                },
                "verifier_metadata": meta,
            }
            f.write(json.dumps(row) + "\n")
            tag = f"  [{meta['media'][0]['type']}]" if meta.get("media") else ""
            print(f"  + id={i} {sc.get('id')}{tag}")
    print(f"wrote {len(scenarios)} scenario(s) -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="text_shopping")
    args = ap.parse_args()
    build(args.dataset)


if __name__ == "__main__":
    main()
