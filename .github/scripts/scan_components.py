#!/usr/bin/env python3
"""scan_components.py — Scan repo for NVIDIA and third-party components.
Generates COMPONENTS.md with links and license information.
"""

import json
import os
import re
import urllib.request
from pathlib import Path

# ── NVIDIA identifiers ────────────────────────────────────────────────────────

NVIDIA_PYPI = {
    'torch', 'torchvision', 'torchaudio', 'torchtext', 'torchdata',
    'tensorrt', 'tensorrt-cu11', 'tensorrt-cu12',
    'pynvml', 'nvidia-ml-py', 'nvidia-ml-py3',
    'cuda-python',
    'cudf', 'cuml', 'cugraph', 'cusignal',
    'cupy', 'cupy-cuda11x', 'cupy-cuda12x', 'cupy-cuda13x',
    'nemo-toolkit', 'nemo_toolkit',
    'tritonclient',
    'onnxruntime-gpu',
    'xformers',
    'apex',
    'megatron-core',
    'nvtx',
    'nvidia-riva-client',
    'langchain-nvidia-ai-endpoints',
}

NVIDIA_NPM = {
    '@nvidia/cuda-runtime',
    'onnxruntime-node',
}

NVIDIA_PATTERNS = [
    r'^nvidia[_-]',
    r'^cuda[_-]',
    r'^cudnn',
    r'^tensorrt',
    r'^triton[_-]',
    r'^nemo[_-]',
    r'^rapids[_-]',
    r'^cugraph',
    r'^cudf',
    r'^cuml',
    r'^cuspatial',
    r'^cusignal',
    r'^cupy',
]

# ── Known NVIDIA container catalog entries ────────────────────────────────────
# Maps nvcr.io image prefix → display info
# Containers: NGC base images used in Dockerfiles
NVIDIA_CONTAINER_CATALOG = {
    'nvcr.io/nvidia/base/ubuntu': {
        'name': 'NVIDIA Base Ubuntu',
        'url': 'https://catalog.ngc.nvidia.com/orgs/nvidia/containers/base',
        'license': 'NVIDIA Deep Learning Container License',
        'license_url': 'https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license',
    },
}

_APACHE = ('Apache-2.0', 'https://spdx.org/licenses/Apache-2.0.html')
THIRD_PARTY_CONTAINER_CATALOG = {
    'python': {'name': 'Python', 'url': 'https://hub.docker.com/_/python', 'license': 'PSF-2.0', 'license_url': 'https://spdx.org/licenses/Python-2.0.html'},
    'postgres': {'name': 'PostgreSQL', 'url': 'https://hub.docker.com/_/postgres', 'license': 'PostgreSQL License', 'license_url': 'https://www.postgresql.org/about/licence/'},
    'nginx': {'name': 'nginx', 'url': 'https://hub.docker.com/_/nginx', 'license': 'BSD-2-Clause', 'license_url': 'https://spdx.org/licenses/BSD-2-Clause.html'},
    'quay.io/coreos/etcd': {'name': 'etcd', 'url': 'https://github.com/etcd-io/etcd', 'license': _APACHE[0], 'license_url': _APACHE[1]},
    'chrislusf/seaweedfs': {'name': 'SeaweedFS', 'url': 'https://github.com/seaweedfs/seaweedfs', 'license': _APACHE[0], 'license_url': _APACHE[1]},
    'milvusdb/milvus': {'name': 'Milvus', 'url': 'https://github.com/milvus-io/milvus', 'license': _APACHE[0], 'license_url': _APACHE[1]},
    'otel/opentelemetry-collector-contrib': {'name': 'OpenTelemetry Collector Contrib', 'url': 'https://github.com/open-telemetry/opentelemetry-collector-contrib', 'license': _APACHE[0], 'license_url': _APACHE[1]},
    'arizephoenix/phoenix': {'name': 'Arize Phoenix', 'url': 'https://github.com/Arize-ai/phoenix', 'license': 'Elastic License 2.0', 'license_url': 'https://www.elastic.co/licensing/elastic-license'},
    'vllm/vllm-openai': {'name': 'vLLM', 'url': 'https://github.com/vllm-project/vllm', 'license': _APACHE[0], 'license_url': _APACHE[1]},
}


# Hosted third-party services, listed when a marker string appears in the source.
THIRD_PARTY_SERVICE_CATALOG = [
    {
        'marker': 'weather.visualcrossing.com',
        'name': 'Visual Crossing Weather API',
        'url': 'https://www.visualcrossing.com/',
        'license': 'Visual Crossing Terms of Service',
        'license_url': 'https://www.visualcrossing.com/weather-services-terms/',
    },
]

# Model containers used in compose files (keyed by nvcr.io image prefix)
NVIDIA_NIM_CATALOG = {}

# Models referenced by ID in config files (keyed by "org/model-name")
MODEL_ID_CATALOG = {
    'nvidia/nemotron-3.5-super-vl-preview': {'name': 'Nemotron 3.5 Super VL', 'url': 'https://build.nvidia.com/nvidia/nemotron-3.5-super-vl-preview', 'license': 'NVIDIA Community Models License', 'license_url': 'https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-community-models-license/'},
    'nvidia/nemotron-3-embed-1b': {'name': 'Nemotron 3 Embed 1B', 'url': 'https://build.nvidia.com/nvidia/nemotron-3-embed-1b', 'license': 'NVIDIA Community Models License', 'license_url': 'https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-community-models-license/'},
    'nvidia/nemotron-3-embed-1b-bf16': {'name': 'Nemotron 3 Embed 1B', 'url': 'https://build.nvidia.com/nvidia/nemotron-3-embed-1b', 'license': 'NVIDIA Community Models License', 'license_url': 'https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-community-models-license/'},
    'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning': {'name': 'Nemotron 3 Nano Omni 30B A3B Reasoning', 'url': 'https://build.nvidia.com/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning', 'license': 'NVIDIA Community Models License', 'license_url': 'https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-community-models-license/'},
    'nvidia/nemotron-3.5-content-safety': {'name': 'Nemotron 3.5 Content Safety', 'url': 'https://huggingface.co/nvidia/Nemotron-3.5-Content-Safety', 'license': 'NVIDIA Community Models License', 'license_url': 'https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-community-models-license/'},
    'nvidia/llama-3.1-nemoguard-8b-topic-control': {
        'name': 'Llama 3.1 NemoGuard 8B Topic Control',
        'url': 'https://build.nvidia.com/nvidia/llama-3_1-nemoguard-8b-topic-control',
        'license': 'Llama 3.1 Community License',
        'license_url': 'https://www.llama.com/llama3_1/license/',
    },
}

# ── Manual overrides for packages/repos without registry license metadata ─────
LICENSE_OVERRIDES = {}

# ── Internal service display names (docker-compose service key → display name) ─
INTERNAL_SERVICE_NAMES = {
    'chain-server': 'Chain Server',
    'catalog-indexer': 'Catalog Indexer',
    'catalog-retriever': 'Catalog Retriever',
    'memory-retriever': 'Memory Retriever',
    'rails': 'Guardrails Service',
    'frontend': 'Frontend UI',
}


def is_nvidia(name: str) -> bool:
    n = name.lower().replace('_', '-')
    if n in {p.lower() for p in NVIDIA_PYPI | NVIDIA_NPM}:
        return True
    return any(re.match(pat, n) for pat in NVIDIA_PATTERNS)


# ── HTTP helper ───────────────────────────────────────────────────────────────

def http_get(url: str):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'bp-legal-scanner/1.0'})
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())
    except Exception:
        return None


# ── Registry fetchers ─────────────────────────────────────────────────────────

CLASSIFIER_TO_SPDX = {
    'MIT License': 'MIT',
    'MIT': 'MIT',
    'Apache Software License': 'Apache-2.0',
    'Apache 2.0': 'Apache-2.0',
    'Apache-2.0': 'Apache-2.0',
    'BSD License': 'BSD-3-Clause',
    'BSD 2-Clause': 'BSD-2-Clause',
    'BSD 3-Clause': 'BSD-3-Clause',
    'BSD 3-Clause License': 'BSD-3-Clause',
    'GNU General Public License v2 (GPLv2)': 'GPL-2.0-only',
    'GNU General Public License v3 (GPLv3)': 'GPL-3.0-only',
    'GNU Lesser General Public License v2 (LGPLv2)': 'LGPL-2.0-only',
    'GNU Lesser General Public License v3 (LGPLv3)': 'LGPL-3.0-only',
    'ISC License (ISCL)': 'ISC',
    'Mozilla Public License 2.0 (MPL 2.0)': 'MPL-2.0',
    'Python Software Foundation License': 'PSF-2.0',
    'The Unlicense (Unlicense)': 'Unlicense',
    'Boost Software License 1.0 (BSL-1.0)': 'BSL-1.0',
}


def license_from_classifiers(classifiers):
    for c in classifiers:
        if c.startswith('License :: OSI Approved ::'):
            label = c.split('::')[-1].strip()
            return CLASSIFIER_TO_SPDX.get(label, label)
        if c.startswith('License ::'):
            label = c.split('::')[-1].strip()
            if label and label != 'OSI Approved':
                return CLASSIFIER_TO_SPDX.get(label, label)
    return None


def fetch_pypi(pkg: str):
    data = http_get(f'https://pypi.org/pypi/{pkg}/json')
    if data:
        info = data.get('info', {})
        lic = (info.get('license') or '').strip()
        if not lic or len(lic) > 80:
            lic = (
                (info.get('license_expression') or '').strip()
                or license_from_classifiers(info.get('classifiers', []))
                or ('See project page' if len(lic) > 80 else 'Unknown')
            )
        name = info.get('name', pkg)
        return {'name': name, 'license': lic or 'Unknown', 'url': f'https://pypi.org/project/{name}/'}
    return {'name': pkg, 'license': 'Unknown', 'url': f'https://pypi.org/project/{pkg}/'}


def fetch_npm(pkg: str):
    encoded = pkg.replace('/', '%2F')
    data = http_get(f'https://registry.npmjs.org/{encoded}/latest')
    lic = None
    name = pkg
    if data:
        name = data.get('name', pkg)
        lic = data.get('license')
        if isinstance(lic, dict):
            lic = lic.get('type')
    if not lic:
        full = http_get(f'https://registry.npmjs.org/{encoded}')
        if full:
            lic = full.get('license')
            if isinstance(lic, dict):
                lic = lic.get('type')
            if not lic:
                versions = full.get('versions', {})
                latest_ver = full.get('dist-tags', {}).get('latest')
                ver_data = versions.get(latest_ver, {})
                lic = ver_data.get('license')
                if isinstance(lic, dict):
                    lic = lic.get('type')
    return {
        'name': name,
        'license': (lic or 'Unknown') if not isinstance(lic, dict) else lic.get('type', 'Unknown'),
        'url': f'https://www.npmjs.com/package/{pkg}',
    }


def spdx_url(spdx: str):
    if not spdx or spdx.lower() in ('unknown', 'see project page', ''):
        return None
    first = re.split(r'\s+(?:OR|AND)\s+', spdx)[0].strip()
    first = re.sub(r'[^A-Za-z0-9._-]', '', first)
    return f'https://spdx.org/licenses/{first}.html' if first else None


# ── Manifest parsers ──────────────────────────────────────────────────────────

def parse_requirements(path: Path):
    pkgs = []
    for line in path.read_text(errors='ignore').splitlines():
        line = line.strip()
        if not line or line.startswith(('#', '-', '/', 'http')):
            continue
        m = re.match(r'^([A-Za-z0-9_.-]+)', line)
        if m:
            pkgs.append(m.group(1))
    return pkgs


def parse_setup_cfg(path: Path):
    pkgs, in_install = [], False
    for line in path.read_text(errors='ignore').splitlines():
        if re.match(r'install_requires\s*=', line):
            in_install = True
            continue
        if in_install:
            if line.startswith('[') or (line and not line[0].isspace()):
                break
            m = re.match(r'\s+([A-Za-z0-9_.-]+)', line)
            if m:
                pkgs.append(m.group(1))
    return pkgs


def parse_setup_py(path: Path):
    """Extract packages from install_requires in setup.py."""
    pkgs = []
    content = path.read_text(errors='ignore')
    m = re.search(r'install_requires\s*=\s*\[(.*?)\]', content, re.DOTALL)
    if m:
        pkgs += re.findall(r'["\']([A-Za-z0-9_.-]+)', m.group(1))
    return pkgs


def parse_pyproject_toml(path: Path):
    pkgs = []
    content = path.read_text(errors='ignore')
    m = re.search(r'\[project\].*?dependencies\s*=\s*\[(.*?)\]', content, re.DOTALL)
    if m:
        pkgs += re.findall(r'"([A-Za-z0-9_.-]+)', m.group(1))
    m = re.search(r'\[tool\.poetry\.dependencies\](.*?)(?=\n\[|\Z)', content, re.DOTALL)
    if m:
        for pkg in re.findall(r'^([A-Za-z0-9_.-]+)\s*=', m.group(1), re.MULTILINE):
            if pkg.lower() != 'python':
                pkgs.append(pkg)
    return pkgs


def parse_conda_env(path: Path):
    pkgs, in_deps = [], False
    for line in path.read_text(errors='ignore').splitlines():
        s = line.strip()
        if s == 'dependencies:':
            in_deps = True
            continue
        if in_deps:
            if s and not s.startswith('-') and not line.startswith(' '):
                break
            if '::' in s:
                pkg = s.split('::')[-1].split('=')[0].strip('- ')
                pkgs.append(pkg)
            else:
                m = re.match(r'\s*-\s*([A-Za-z0-9_.-]+)', s)
                if m:
                    pkgs.append(m.group(1))
    return pkgs


def parse_package_json(path: Path):
    try:
        data = json.loads(path.read_text())
        return list(data.get('dependencies', {}).keys())
    except Exception:
        return []


def parse_models_json(path: Path):
    """Extract model IDs from a models.json config file."""
    model_ids = []
    try:
        data = json.loads(path.read_text())
        items = data if isinstance(data, list) else data.values() if isinstance(data, dict) else []
        for item in items:
            if isinstance(item, dict):
                for key in ('model', 'model_id', 'name', 'id'):
                    val = item.get(key, '')
                    if val and re.match(r'^[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+$', str(val)):
                        model_ids.append(str(val))
                        break
    except Exception:
        pass
    return model_ids


def parse_dockerfile(path: Path):
    """Return list of nvcr.io base image strings from FROM lines, resolving ARG defaults."""
    args = {}
    images = []
    for line in path.read_text(errors='ignore').splitlines():
        line = line.strip()
        m = re.match(r'^ARG\s+(\w+)[=\s]["\']?([^\s"\']+)', line, re.IGNORECASE)
        if m:
            args[m.group(1)] = m.group(2)
            continue
        m = re.match(r'^FROM\s+(\S+)', line, re.IGNORECASE)
        if m:
            img = m.group(1)
            def replace_arg(match):
                return args.get(match.group(1) or match.group(2), match.group(0))
            img = re.sub(r'\$\{(\w+)\}|\$(\w+)', replace_arg, img)
            if img.startswith('nvcr.io/'):
                images.append(img)
    return images


def parse_dockerfile_pip(path: Path):
    """Extract Python packages from RUN pip install lines in Dockerfiles."""
    pkgs = []
    # Join continuation lines (lines ending with \)
    lines = path.read_text(errors='ignore').splitlines()
    joined, buf = [], ''
    for line in lines:
        if line.rstrip().endswith('\\'):
            buf += line.rstrip()[:-1] + ' '
        else:
            buf += line
            joined.append(buf)
            buf = ''
    if buf:
        joined.append(buf)
    for line in joined:
        if not re.search(r'\bpip[23]?\s+install\b', line, re.IGNORECASE):
            continue
        tokens = re.split(r'\s+', line.strip())
        in_install = False
        skip_next = False
        for token in tokens:
            if skip_next:
                skip_next = False
                continue
            if token in ('&&', '||', ';', 'fi', 'then', 'else', 'done'):
                in_install = False
                continue
            if token in ('RUN', 'pip', 'pip3', 'pip2', ''):
                continue
            if token == 'install':
                in_install = True
                continue
            if not in_install:
                continue
            # Flags that consume the next token
            if token in ('--index-url', '--extra-index-url', '--trusted-host',
                         '-f', '--find-links', '-r', '--requirement', '-c', '-i'):
                skip_next = True
                continue
            if token.startswith('-'):
                continue
            # Skip local paths and URLs
            if token.startswith(('/', './', '../', 'http')):
                continue
            m = re.match(r'^([A-Za-z0-9_.-]+)', token)
            if m:
                pkgs.append(m.group(1))
    return pkgs


def parse_compose(path: Path):
    """Return list of nvcr.io/nim/ image strings from docker-compose files."""
    images = []
    for line in path.read_text(errors='ignore').splitlines():
        m = re.match(r'\s+image:\s*["\']?(nvcr\.io/nim/[^\s"\']+)', line)
        if m:
            images.append(m.group(1))
    return images


def parse_compose_all_images(path: Path):
    """Return every image: value from a compose file, with ${VAR:-default} resolved to the default."""
    images = []
    for line in path.read_text(errors='ignore').splitlines():
        m = re.match(r'\s+image:\s*["\']?([^\s"\']+)', line)
        if m:
            images.append(re.sub(r'\$\{[^:}]+:-([^}]*)\}', r'\1', m.group(1)))
    return images


def parse_compose_internal_services(path: Path, repo_root: Path):
    """Return services built from the repo's own source directories."""
    services = []
    content = path.read_text(errors='ignore')
    current_service = {}
    for line in content.splitlines():
        m = re.match(r'^  ([a-zA-Z][a-zA-Z0-9_-]+):\s*$', line)
        if m:
            if current_service.get('build'):
                services.append(current_service)
            current_service = {'service_key': m.group(1)}
        elif re.match(r'\s+container_name:\s*(.+)', line):
            current_service['container_name'] = re.match(r'\s+container_name:\s*(.+)', line).group(1).strip()
        elif re.match(r'\s+build:\s+\S', line):  # short-form: build: ./path
            m = re.match(r'\s+build:\s+["\']?([^\s"\']+)', line)
            if m:
                current_service['build'] = m.group(1).strip()
        elif re.match(r'\s+context:\s*(.+)', line):
            current_service['build'] = re.match(r'\s+context:\s*(.+)', line).group(1).strip()
        elif re.match(r'\s+dockerfile:\s*(.+)', line):
            # Any service with a dockerfile key is an internal build
            if 'build' not in current_service:
                current_service['build'] = '.'
    if current_service.get('build'):
        services.append(current_service)
    return services


def parse_config_for_models(path: Path):
    """Extract model IDs (org/name format) from YAML config files."""
    model_ids = []
    for line in path.read_text(errors='ignore').splitlines():
        m = re.match(r'\s+model:\s*["\']?([a-zA-Z0-9_-]+(?:/[a-zA-Z0-9_.-]+)+)["\']?', line)
        if m:
            model_ids.append('/'.join(m.group(1).split('/')[-2:]).lower())
    return model_ids


def parse_env_example_models(path: Path):
    """Extract default model IDs from lines like export X_MODEL="${X_MODEL:-org/name}"."""
    model_ids = []
    for line in path.read_text(errors='ignore').splitlines():
        m = re.match(r'\s*export\s+\w*MODEL\w*="\$\{\w+:-([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+)\}"', line)
        if m:
            model_ids.append('/'.join(m.group(1).split('/')[-2:]).lower())
    return model_ids


def parse_notebook(path: Path):
    """Extract model IDs and nvcr.io images from Jupyter notebooks."""
    model_ids = []
    nim_images = []
    try:
        data = json.loads(path.read_text(errors='ignore'))
        for cell in data.get('cells', []):
            source = ''.join(cell.get('source', []))
            for m in re.finditer(r'model["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)', source):
                model_ids.append(m.group(1))
            for m in re.finditer(r'(nvcr\.io/nim/[^\s"\']+)', source):
                nim_images.append(m.group(1))
    except Exception:
        pass
    return model_ids, nim_images


def resolve_image(image: str, catalog: dict):
    """Match an image string to a catalog entry by prefix."""
    base = image.split(':')[0]
    for prefix in sorted(catalog.keys(), key=len, reverse=True):
        if base == prefix or base.startswith(prefix):
            return catalog[prefix]
    return None


# ── Submodule parser ──────────────────────────────────────────────────────────

def parse_gitmodules(root: Path):
    p = root / '.gitmodules'
    if not p.exists():
        return []
    submodules = []
    current = {}
    for line in p.read_text(errors='ignore').splitlines():
        line = line.strip()
        m = re.match(r'\[submodule "(.+)"\]', line)
        if m:
            if current:
                submodules.append(current)
            current = {'name': m.group(1).split('/')[-1]}
        elif line.startswith('url ='):
            url = line.split('=', 1)[1].strip()
            current['url'] = url[:-4] if url.endswith('.git') else url
    if current:
        submodules.append(current)
    return submodules


def fetch_github_license(github_url: str):
    m = re.search(r'github\.com/([^/]+/[^/]+)', github_url)
    if not m:
        return 'Unknown'
    slug = m.group(1).rstrip('/')
    data = http_get(f'https://api.github.com/repos/{slug}')
    if data:
        lic = (data.get('license') or {})
        spdx = lic.get('spdx_id', '')
        if spdx and spdx != 'NOASSERTION':
            return spdx
        branch = data.get('default_branch') or 'main'
        for fname in ('LICENSE', 'LICENSE.md', 'LICENSE.txt'):
            raw = None
            try:
                req = urllib.request.Request(
                    f'https://raw.githubusercontent.com/{slug}/{branch}/{fname}',
                    headers={'User-Agent': 'bp-legal-scanner/1.0'}
                )
                with urllib.request.urlopen(req, timeout=10) as r:
                    raw = r.read(500).decode('utf-8', errors='ignore')
            except Exception:
                pass
            if raw:
                if 'MIT License' in raw or 'MIT\n' in raw:
                    return 'MIT'
                if 'Apache License' in raw and '2.0' in raw:
                    return 'Apache-2.0'
                if 'BSD 3-Clause' in raw:
                    return 'BSD-3-Clause'
                if 'BSD 2-Clause' in raw:
                    return 'BSD-2-Clause'
    return 'Unknown'


# ── Repo scanner ──────────────────────────────────────────────────────────────

def scan_repo(root: Path):
    found = {'pypi': set(), 'npm': set(), 'docker_images': [], 'nim_images': [], 'model_ids': set(), 'internal_services': [], 'compose_images': [], 'services': []}

    # Directories to skip for package manifests (not for model config)
    pkg_skip = {'.git', 'node_modules', '.tox', '__pycache__', '.venv', 'venv', 'external'}
    # Directories to always skip
    always_skip = {'.git', 'node_modules', '__pycache__', 'tests'}

    for p in root.rglob('requirements*.txt'):
        if not any(s in p.parts for s in pkg_skip):
            found['pypi'].update(parse_requirements(p))

    for p in root.rglob('setup.cfg'):
        if not any(s in p.parts for s in pkg_skip):
            found['pypi'].update(parse_setup_cfg(p))

    for p in root.rglob('setup.py'):
        if not any(s in p.parts for s in pkg_skip):
            found['pypi'].update(parse_setup_py(p))

    for p in root.rglob('pyproject.toml'):
        if not any(s in p.parts for s in pkg_skip):
            found['pypi'].update(parse_pyproject_toml(p))

    for name in ('environment.yml', 'environment.yaml', 'conda.yml', 'conda.yaml'):
        p = root / name
        if p.exists():
            found['pypi'].update(parse_conda_env(p))

    for p in root.rglob('package.json'):
        if not any(s in p.parts for s in pkg_skip):
            found['npm'].update(parse_package_json(p))

    for p in root.rglob('Dockerfile*'):
        if not any(s in p.parts for s in pkg_skip):
            found['docker_images'].extend(parse_dockerfile(p))
            found['pypi'].update(parse_dockerfile_pip(p))

    # Compose, config, README, and notebooks scanned everywhere (including external/)
    for p in root.rglob('docker-compose*.y*ml'):
        if not any(s in p.parts for s in always_skip):
            found['nim_images'].extend(parse_compose(p))
    for p in root.rglob('compose*.y*ml'):
        if not any(s in p.parts for s in always_skip):
            found['nim_images'].extend(parse_compose(p))

    for p in root.rglob('docker-compose*.y*ml'):
        if not any(s in p.parts for s in always_skip):
            found['compose_images'].extend(parse_compose_all_images(p))

    for p in root.rglob('config.y*ml'):
        if not any(s in p.parts for s in always_skip):
            found['model_ids'].update(parse_config_for_models(p))

    for p in root.rglob('README.md'):
        if not any(s in p.parts for s in always_skip):
            found['model_ids'].update(parse_config_for_models(p))

    for p in root.rglob('models.y*ml'):
        if not any(s in p.parts for s in always_skip):
            found['model_ids'].update(parse_config_for_models(p))

    for p in root.rglob('.env*example*'):
        if not any(s in p.parts for s in always_skip):
            found['model_ids'].update(parse_env_example_models(p))

    for p in root.rglob('models.json'):
        if not any(s in p.parts for s in always_skip):
            found['model_ids'].update(parse_models_json(p))

    for p in root.rglob('*.ipynb'):
        if not any(s in p.parts for s in always_skip):
            nb_models, nb_nims = parse_notebook(p)
            found['model_ids'].update(nb_models)
            found['nim_images'].extend(nb_nims)

    for p in root.rglob('docker-compose*.y*ml'):
        if not any(s in p.parts for s in always_skip):
            found['internal_services'].extend(parse_compose_internal_services(p, root))

    source_files = [p for ext in ('*.py', '*.ts', '*.tsx', '*.yaml', '*.yml') for p in root.rglob(ext)
                    if not any(x in p.parts for x in always_skip | {'.github'})]
    for svc in THIRD_PARTY_SERVICE_CATALOG:
        if any(svc['marker'] in p.read_text(errors='ignore') for p in source_files):
            found['services'].append(svc)

    return found


# ── Enrich with registry data ─────────────────────────────────────────────────

def enrich(packages: dict, submodules: list):
    nvidia, third_party, internal = [], [], []

    for pkg in sorted(packages.get('pypi', [])):
        info = {**fetch_pypi(pkg), 'ecosystem': 'Python'}
        info['license_url'] = spdx_url(info['license'])
        (nvidia if is_nvidia(pkg) else third_party).append(info)

    for pkg in sorted(packages.get('npm', [])):
        info = {**fetch_npm(pkg), 'ecosystem': 'Node.js'}
        if info['license'] == 'Unknown' and pkg in LICENSE_OVERRIDES:
            info['license'] = LICENSE_OVERRIDES[pkg]
        info['license_url'] = spdx_url(info['license'])
        (nvidia if is_nvidia(pkg) else third_party).append(info)

    for sub in submodules:
        lic = LICENSE_OVERRIDES.get(sub['name']) or fetch_github_license(sub['url'])
        info = {
            'name': sub['name'],
            'license': lic or 'Unknown',
            'url': sub['url'],
            'license_url': spdx_url(lic),
            'ecosystem': 'GitHub',
        }
        if re.search(r'github\.com/NVIDIA/', sub['url'], re.IGNORECASE):
            nvidia.append(info)
        else:
            third_party.append(info)

    seen_containers = set()
    for img in packages.get('docker_images', []):
        entry = resolve_image(img, NVIDIA_CONTAINER_CATALOG)
        if entry and entry['name'] not in seen_containers:
            seen_containers.add(entry['name'])
            nvidia.append({**dict(entry), 'ecosystem': 'Containers'})

    for img in packages.get('docker_images', []) + packages.get('compose_images', []):
        base = img.split(':')[0]
        entry = THIRD_PARTY_CONTAINER_CATALOG.get(base)
        if entry and entry['name'] not in seen_containers:
            seen_containers.add(entry['name'])
            third_party.append({**dict(entry), 'ecosystem': 'Containers'})

    for svc in packages.get('services', []):
        third_party.append({k: v for k, v in svc.items() if k != 'marker'} | {'ecosystem': 'Services'})

    seen_nims = set()
    for img in packages.get('nim_images', []):
        entry = resolve_image(img, NVIDIA_NIM_CATALOG)
        if entry and entry['name'] not in seen_nims:
            seen_nims.add(entry['name'])
            nvidia.append({**dict(entry), 'ecosystem': 'Models'})

    for model_id in sorted(packages.get('model_ids', [])):
        entry = MODEL_ID_CATALOG.get(model_id.lower())
        if entry and entry['name'] not in seen_nims:
            seen_nims.add(entry['name'])
            item = {**dict(entry), 'ecosystem': 'Models'}
            org = model_id.split('/')[0].lower()
            if org in ('meta', 'mistral', 'google', 'microsoft', 'amazon') or 'llama' in model_id.lower():
                third_party.append(item)
            else:
                nvidia.append(item)

    repo_url = 'https://github.com/' + os.environ.get('GITHUB_REPOSITORY', 'NVIDIA-AI-Blueprints/this-repo')
    seen_internal = set()
    for svc in packages.get('internal_services', []):
        container = svc.get('container_name', svc.get('service_key', ''))
        if container in seen_internal:
            continue
        seen_internal.add(container)
        display = INTERNAL_SERVICE_NAMES.get(container, container)
        source = svc.get('build', '').replace('../', '').lstrip('./')
        internal.append({'name': display, 'source': source, 'repo_url': repo_url})

    return nvidia, third_party, internal


# ── Markdown renderer ─────────────────────────────────────────────────────────

def render_item(comp: dict) -> str:
    name_link = f"[{comp['name']}]({comp['url']})"
    lic = comp.get('license', 'Unknown')
    lic_url = comp.get('license_url')
    lic_part = f"[{lic}]({lic_url})" if lic_url else lic
    return f"- {name_link} ({lic_part})"


def generate_markdown(nvidia, third_party, internal):
    lines = [
        '# Component List',
        '',
        '> Auto-generated by the `component-list` GitHub Action. Do not edit manually.',
        '',
    ]

    # Merge internal services into nvidia under 'Source' ecosystem
    for comp in internal:
        source = comp.get('source', '')
        url = f"{comp['repo_url']}/tree/main/{source}" if source else comp['repo_url']
        nvidia.append({
            'name': comp['name'],
            'url': url,
            'license': 'NVIDIA Proprietary',
            'license_url': None,
            'ecosystem': 'Source',
        })

    def section(title, components):
        if not components:
            return
        lines.extend([f'## {title}', ''])
        for eco in ('Source', 'GitHub', 'Models', 'Services', 'Containers', 'Python', 'Node.js'):
            eco_pkgs = [c for c in components if c.get('ecosystem') == eco]
            if eco_pkgs:
                lines.extend([f'### {eco}', ''])
                for comp in sorted(eco_pkgs, key=lambda x: x['name'].lower()):
                    lines.append(render_item(comp))
                lines.append('')

    section('NVIDIA Components', nvidia)
    section('Third-Party Components', third_party)

    if not nvidia and not third_party:
        lines += ['*No components detected.*', '']

    return '\n'.join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    root = Path(os.environ.get('GITHUB_WORKSPACE', '.'))
    print(f'Scanning {root} ...')

    packages = scan_repo(root)
    submodules = parse_gitmodules(root)
    print(f'Found packages, {len(submodules)} submodules, '
          f'{len(packages["docker_images"])} container images, '
          f'{len(packages["nim_images"])} model containers — fetching registry info ...')

    nvidia, third_party, internal = enrich(packages, submodules)
    print(f'NVIDIA: {len(nvidia) + len(internal)}  |  Third-party: {len(third_party)}')

    md = generate_markdown(nvidia, third_party, internal)
    out = root / 'COMPONENTS.md'
    out.write_text(md)
    print(f'Written: {out}')
