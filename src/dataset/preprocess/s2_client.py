"""Semantic Scholar API client with rate limiting, disk caching, and batch support."""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .config import (
    S2_API_BASE,
    S2_CACHE_DIR,
    S2_MAX_RETRIES,
    S2_RATE_LIMIT,
    S2_TIMEOUT,
    ensure_dirs,
    semantic_scholar_api_key,
)

# Batch endpoint accepts up to 500 IDs per call
_BATCH_MAX = 500


class S2Client:
    """Rate-limited, cached Semantic Scholar Graph API client."""

    def __init__(self, cache_dir: Path | None = None, rate_limit: int | None = None):
        ensure_dirs()
        self.cache_dir = cache_dir or S2_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = semantic_scholar_api_key()
        self._min_interval = 1.0 / (rate_limit or S2_RATE_LIMIT)
        self._last_request_time = 0.0
        self._request_count = 0

    # ── Low-level requests ───────────────────────────────────────────────

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

    def _get(self, url: str) -> dict | None:
        """Send a GET request with retries and rate limiting."""
        for attempt in range(S2_MAX_RETRIES):
            self._rate_limit()
            try:
                req = urllib.request.Request(url)
                req.add_header("User-Agent", "EVO-CitationGraph/1.0 (Academic Research)")
                if self.api_key:
                    req.add_header("x-api-key", self.api_key)

                self._last_request_time = time.time()
                self._request_count += 1
                with urllib.request.urlopen(req, timeout=S2_TIMEOUT) as resp:
                    return json.loads(resp.read().decode("utf-8"))

            except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait = min(2 ** (attempt + 2), 60)
                    print(f"  S2 rate limited, waiting {wait}s...")
                    time.sleep(wait)
                elif e.code == 404:
                    return None
                else:
                    if attempt < S2_MAX_RETRIES - 1:
                        time.sleep(2 * (attempt + 1))
                    else:
                        print(f"  S2 HTTP {e.code} for {url}")
                        return None
            except Exception as e:
                if attempt < S2_MAX_RETRIES - 1:
                    time.sleep(2 * (attempt + 1))
                else:
                    print(f"  S2 error: {e}")
                    return None
        return None

    def _post(self, url: str, body: dict) -> list | dict | None:
        """Send a POST request with retries and rate limiting."""
        payload = json.dumps(body).encode("utf-8")
        for attempt in range(S2_MAX_RETRIES):
            self._rate_limit()
            try:
                req = urllib.request.Request(url, data=payload, method="POST")
                req.add_header("Content-Type", "application/json")
                req.add_header("User-Agent", "EVO-CitationGraph/1.0 (Academic Research)")
                if self.api_key:
                    req.add_header("x-api-key", self.api_key)

                self._last_request_time = time.time()
                self._request_count += 1
                with urllib.request.urlopen(req, timeout=S2_TIMEOUT) as resp:
                    return json.loads(resp.read().decode("utf-8"))

            except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait = min(2 ** (attempt + 2), 60)
                    print(f"  S2 rate limited (batch), waiting {wait}s...")
                    time.sleep(wait)
                else:
                    if attempt < S2_MAX_RETRIES - 1:
                        time.sleep(2 * (attempt + 1))
                    else:
                        print(f"  S2 batch HTTP {e.code}")
                        return None
            except Exception as e:
                if attempt < S2_MAX_RETRIES - 1:
                    time.sleep(2 * (attempt + 1))
                else:
                    print(f"  S2 batch error: {e}")
                    return None
        return None

    # ── Cache helpers ────────────────────────────────────────────────────

    def _cache_path(self, key: str) -> Path:
        safe = key.replace("/", "_").replace(":", "_")
        return self.cache_dir / f"{safe}.json"

    def _read_cache(self, key: str) -> dict | None:
        p = self._cache_path(key)
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                return None
        return None

    def _write_cache(self, key: str, data: dict):
        self._cache_path(key).write_text(
            json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        )

    # ── Single-paper API ─────────────────────────────────────────────────

    def get_paper(self, paper_id: str,
                  fields: str = "paperId,title,externalIds,abstract") -> dict | None:
        """Look up a paper by S2 paper ID, DOI, or ARXIV:{id}."""
        cache_key = f"paper_{paper_id}"
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached

        encoded = urllib.parse.quote(paper_id, safe=":")
        url = f"{S2_API_BASE}/paper/{encoded}?fields={fields}"
        data = self._get(url)
        if data:
            self._write_cache(cache_key, data)
        return data

    def get_paper_by_arxiv(self, arxiv_id: str) -> dict | None:
        """Look up a paper by its arXiv ID."""
        return self.get_paper(f"ARXIV:{arxiv_id}")

    def get_references(self, paper_id: str,
                       fields: str = "paperId,title,externalIds"
                       ) -> list[dict]:
        """Get all references (cited papers) for a given paper. Handles pagination."""
        cache_key = f"refs_{paper_id}"
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached.get("references", [])

        all_refs = []
        offset = 0
        limit = 500

        while True:
            url = (f"{S2_API_BASE}/paper/{paper_id}/references"
                   f"?fields={fields}&offset={offset}&limit={limit}")
            data = self._get(url)
            if not data or not data.get("data"):
                break

            for entry in data["data"]:
                cited = entry.get("citedPaper")
                if cited and cited.get("paperId"):
                    all_refs.append(cited)

            if data.get("next"):
                offset = data["next"]
            else:
                break

        self._write_cache(cache_key, {"references": all_refs})
        return all_refs

    # ── Batch API ────────────────────────────────────────────────────────

    def batch_get_references(
        self,
        paper_ids: list[str],
        fields: str = "paperId,references.paperId",
    ) -> dict[str, list[str]]:
        """Batch-fetch references for many papers in one call.

        Uses POST /paper/batch with references.paperId in fields.
        Returns {paper_id: [list of cited paper IDs]}.

        Papers are chunked into groups of 500 (S2 batch limit).
        Results are cached per-batch key.
        """
        # Check cache first — keyed per individual paper
        result: dict[str, list[str]] = {}
        uncached_ids: list[str] = []

        for pid in paper_ids:
            cache_key = f"batch_refs_{pid}"
            cached = self._read_cache(cache_key)
            if cached is not None:
                result[pid] = cached.get("ref_ids", [])
            else:
                uncached_ids.append(pid)

        if not uncached_ids:
            return result

        # Process in chunks of _BATCH_MAX
        for chunk_start in range(0, len(uncached_ids), _BATCH_MAX):
            chunk = uncached_ids[chunk_start:chunk_start + _BATCH_MAX]
            url = f"{S2_API_BASE}/paper/batch?fields={fields}"
            body = {"ids": chunk}

            data = self._post(url, body)
            if not data or not isinstance(data, list):
                # Mark all as empty on failure
                for pid in chunk:
                    result[pid] = []
                    self._write_cache(f"batch_refs_{pid}", {"ref_ids": []})
                continue

            for i, paper_data in enumerate(data):
                pid = chunk[i]
                if paper_data is None:
                    ref_ids = []
                else:
                    refs = paper_data.get("references") or []
                    ref_ids = [
                        r["paperId"] for r in refs
                        if r and r.get("paperId")
                    ]
                result[pid] = ref_ids
                self._write_cache(f"batch_refs_{pid}", {"ref_ids": ref_ids})

        return result

    def batch_get_abstracts(
        self,
        paper_ids: list[str],
    ) -> dict[str, str]:
        """Batch-fetch abstracts for many papers in one call.

        Uses POST /paper/batch with fields=paperId,abstract.
        Returns {paper_id: abstract_text}.
        """
        result: dict[str, str] = {}
        uncached_ids: list[str] = []

        for pid in paper_ids:
            cache_key = f"abstract_{pid}"
            cached = self._read_cache(cache_key)
            if cached is not None:
                result[pid] = cached.get("abstract", "")
            else:
                uncached_ids.append(pid)

        if not uncached_ids:
            return result

        for chunk_start in range(0, len(uncached_ids), _BATCH_MAX):
            chunk = uncached_ids[chunk_start:chunk_start + _BATCH_MAX]
            url = f"{S2_API_BASE}/paper/batch?fields=paperId,abstract"
            body = {"ids": chunk}

            data = self._post(url, body)
            if not data or not isinstance(data, list):
                for pid in chunk:
                    result[pid] = ""
                continue

            for i, paper_data in enumerate(data):
                pid = chunk[i]
                abstract = ""
                if paper_data and paper_data.get("abstract"):
                    abstract = paper_data["abstract"]
                result[pid] = abstract
                self._write_cache(f"abstract_{pid}", {"abstract": abstract})

        return result

    @property
    def request_count(self) -> int:
        return self._request_count
