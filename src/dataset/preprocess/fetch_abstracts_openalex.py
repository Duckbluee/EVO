"""Fetch missing abstracts from OpenAlex API (free, no key needed).

Fills in abstracts for nodes where Semantic Scholar had no abstract.
Uses title-based search with 10 req/s rate limit.
"""

import json
import time
import torch
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .config import GRAPH_DIR, ensure_dirs

_OPENALEX_BASE = "https://api.openalex.org"
_USER_AGENT = "EVO-CitationGraph/1.0 (mailto:research@evo-project.org)"
_RATE_LIMIT = 10  # requests per second
_MIN_INTERVAL = 1.0 / _RATE_LIMIT
_last_request = 0.0


def _reconstruct_abstract(inverted_index: dict) -> str:
    """Reconstruct abstract text from OpenAlex inverted index format."""
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort()
    return " ".join(w for _, w in word_positions)


def _search_abstract(title: str) -> str | None:
    """Search OpenAlex for a paper by title and return its abstract."""
    global _last_request

    elapsed = time.time() - _last_request
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)

    encoded = urllib.parse.quote(title)
    url = f"{_OPENALEX_BASE}/works?search={encoded}&per_page=1"

    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", _USER_AGENT)
        _last_request = time.time()

        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())

        if not data.get("results"):
            return None

        work = data["results"][0]
        inv_idx = work.get("abstract_inverted_index")
        if not inv_idx:
            return None

        # Verify title roughly matches (avoid wrong paper)
        oa_title = (work.get("title") or "").lower().strip()
        query_title = title.lower().strip()
        # Simple check: first 40 chars should overlap
        if oa_title[:40] != query_title[:40]:
            return None

        return _reconstruct_abstract(inv_idx)

    except Exception:
        return None


def update_missing_abstracts(
    graph_dir: Path = GRAPH_DIR,
    verbose: bool = True,
) -> dict[str, dict]:
    """Fetch missing abstracts from OpenAlex for all graphs."""
    ensure_dirs()
    graph_files = sorted(graph_dir.glob("*.pt"))

    total_missing = 0
    total_found = 0
    stats = {}

    for i, gf in enumerate(graph_files):
        arxiv_id = gf.stem
        graph = torch.load(gf, weights_only=False)
        n = graph.x.size(0)

        # Find nodes missing abstracts
        missing_indices = [
            j for j in range(n)
            if graph.abstract[j] == "N/A" and graph.title[j] and graph.title[j] != "Unknown"
        ]

        if not missing_indices:
            stats[arxiv_id] = {"missing": 0, "found": 0}
            continue

        found = 0
        for j in missing_indices:
            abstract = _search_abstract(graph.title[j])
            if abstract:
                graph.abstract[j] = abstract
                found += 1

        total_missing += len(missing_indices)
        total_found += found

        if found > 0:
            torch.save(graph, gf)

        if verbose:
            print(
                f"[{i+1}/{len(graph_files)}] {arxiv_id}: "
                f"{found}/{len(missing_indices)} abstracts found via OpenAlex",
                flush=True,
            )

        stats[arxiv_id] = {"missing": len(missing_indices), "found": found}

    print(f"\nDone: found {total_found}/{total_missing} missing abstracts via OpenAlex")
    remaining = total_missing - total_found
    print(f"Still missing: {remaining}")

    return stats


if __name__ == "__main__":
    update_missing_abstracts()
