"""Fetch missing abstracts from Crossref API (free, no daily limit).

Fills in abstracts for nodes where both S2 and OpenAlex had no abstract.
Uses title-based search with polite rate limit (~50 req/s with email).
"""

import json
import re
import time
import torch
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .config import GRAPH_DIR, ensure_dirs

_CROSSREF_BASE = "https://api.crossref.org"
_USER_AGENT = "EVO-CitationGraph/1.0 (mailto:research@evo-project.org)"


def _normalize(title: str) -> str:
    return " ".join(title.lower().strip().rstrip(".").split())


def _search_abstract_crossref(title: str) -> str | None:
    """Search Crossref for a paper by title and return its abstract."""
    encoded = urllib.parse.quote(title)
    url = f"{_CROSSREF_BASE}/works?query.title={encoded}&rows=1&select=title,abstract"

    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", _USER_AGENT)

        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())

        items = data.get("message", {}).get("items", [])
        if not items:
            return None

        item = items[0]
        # Verify title match
        cr_title = (item.get("title") or [""])[0]
        if _normalize(cr_title)[:40] != _normalize(title)[:40]:
            return None

        abstract = item.get("abstract", "")
        if not abstract:
            return None

        # Strip JATS XML tags
        clean = re.sub(r"<[^>]+>", "", abstract).strip()
        return clean if len(clean) > 20 else None

    except Exception:
        return None


def update_missing_abstracts(
    graph_dir: Path = GRAPH_DIR,
    max_workers: int = 5,
    verbose: bool = True,
) -> dict[str, dict]:
    """Fetch missing abstracts from Crossref for all graphs."""
    ensure_dirs()
    graph_files = sorted(graph_dir.glob("*.pt"))

    total_missing = 0
    total_found = 0

    for gi, gf in enumerate(graph_files):
        arxiv_id = gf.stem
        graph = torch.load(gf, weights_only=False)
        n = graph.x.size(0)

        missing = [
            j for j in range(n)
            if graph.abstract[j] == "N/A"
            and graph.title[j]
            and graph.title[j] != "Unknown"
        ]

        if not missing:
            continue

        # Parallel fetch with ThreadPoolExecutor
        found = 0
        results = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_search_abstract_crossref, graph.title[j]): j
                for j in missing
            }
            for fut in as_completed(futures):
                j = futures[fut]
                try:
                    abstract = fut.result()
                    if abstract:
                        results[j] = abstract
                except Exception:
                    pass

        for j, abstract in results.items():
            graph.abstract[j] = abstract
            found += 1

        total_missing += len(missing)
        total_found += found

        if found > 0:
            torch.save(graph, gf)

        if verbose:
            print(
                f"[{gi+1}/{len(graph_files)}] {arxiv_id}: "
                f"{found}/{len(missing)} abstracts found via Crossref",
                flush=True,
            )

    print(f"\nDone: found {total_found}/{total_missing} missing abstracts via Crossref")
    remaining = total_missing - total_found
    print(f"Still missing: {remaining}")


if __name__ == "__main__":
    update_missing_abstracts()
