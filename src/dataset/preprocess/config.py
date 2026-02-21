"""Configuration for the preprocessing pipeline."""

import os
import re
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path("/home/ubuntu/EVO")
ENV_PATH = BASE_DIR / ".env"
DATASET_DIR = BASE_DIR / "dataset"
PDF_DIR = DATASET_DIR / "pdf"
ROOT_PAPERS_FILE = DATASET_DIR / "root_papers.txt"
PARSED_DIR = DATASET_DIR / "parsed_references"
S2_CACHE_DIR = DATASET_DIR / "s2_cache"
GRAPH_DIR = DATASET_DIR / "graphs"
MINERU_OUTPUT_DIR = DATASET_DIR / "mineru_output"

# ── Load .env manually (handles 'KEY = "value"' with spaces) ────────────────
_env_vars: dict[str, str] = {}


def _load_env(path: Path = ENV_PATH) -> dict[str, str]:
    if _env_vars:
        return _env_vars
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"?([^"]*)"?\s*$', line)
            if m:
                _env_vars[m.group(1)] = m.group(2)
    return _env_vars


def get_env(key: str, default: str = "") -> str:
    env = _load_env()
    return env.get(key, os.environ.get(key, default))


# ── API Keys ─────────────────────────────────────────────────────────────────
def semantic_scholar_api_key() -> str:
    return get_env("SEMANTIC_SCHOLAR_API_KEY")


def azure_gpt41mini_endpoint(idx: int = 1) -> str:
    return get_env(f"AZURE_GPT41MINI_ENDPOINT_{idx}")


def azure_gpt41mini_key(idx: int = 1) -> str:
    return get_env(f"AZURE_GPT41MINI_KEY_{idx}")


def azure_gpt52_endpoint(idx: int = 1) -> str:
    return get_env(f"AZURE_GPT52_ENDPOINT_{idx}")


def azure_gpt52_key(idx: int = 1) -> str:
    return get_env(f"AZURE_GPT52_KEY_{idx}")


# ── Semantic Scholar settings ────────────────────────────────────────────────
S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
S2_RATE_LIMIT = 3  # requests per second (conservative to avoid 429s)
S2_MAX_RETRIES = 4
S2_TIMEOUT = 30


# ── Helpers ──────────────────────────────────────────────────────────────────
def load_root_papers() -> list[tuple[str, str]]:
    """Return list of (arxiv_id, title) from root_papers.txt."""
    papers = []
    with open(ROOT_PAPERS_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            arxiv_id, title = line.split(",", 1)
            papers.append((arxiv_id.strip(), title.strip()))
    return papers


def ensure_dirs():
    """Create all output directories."""
    for d in [PARSED_DIR, S2_CACHE_DIR, GRAPH_DIR, MINERU_OUTPUT_DIR]:
        d.mkdir(parents=True, exist_ok=True)
