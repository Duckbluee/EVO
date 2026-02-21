"""Extract methodology taxonomy from survey PDFs using LLM.

Workflow:
  1. Use pypdf to extract full text from each survey PDF.
  2. Send the text to Azure GPT-5.2 with the taxonomy extraction prompt.
  3. Parse the structured JSON output (taxonomy_tree + reference_index).
  4. Cache results as JSON per paper.
"""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import AzureOpenAI
from pypdf import PdfReader

from .config import (
    PDF_DIR,
    PARSED_DIR,
    azure_gpt52_endpoint,
    azure_gpt52_key,
    ensure_dirs,
)

# ── Azure OpenAI client ─────────────────────────────────────────────────────

def _get_llm_client() -> AzureOpenAI:
    endpoint = azure_gpt52_endpoint(1).removesuffix("/openai")
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=azure_gpt52_key(1),
        api_version="2024-12-01-preview",
    )


# ── Taxonomy extraction prompt (from English_prompt.txt) ─────────────────────

_SYSTEM_PROMPT = r"""Task:
Extract Methodology Taxonomy from a Survey/Review Paper and output a Section-organized hierarchical JSON.

Input:
You will receive a survey/review paper (PDF or extracted text).

Objective:

Extract ONLY methodology/technical classification sections.

Construct full section hierarchy.

Attach references appearing within each section.

Organize output by Section-first structure.

Do NOT organize by paper globally.

Required Output Structure:

{
"survey_title": "...",
"taxonomy_tree": [...],
"reference_index": {...}
}

Each Section node must be:

{
"label": "3.2 Graph-based Methods",
"path": ["3 Methods", "3.2 Graph-based Methods"],
"children": [...],
"papers": [...]
}

Rules:

label must preserve original numbering and title.

path must contain full hierarchy from methodology root.

children must contain nested subsections.

papers must contain references appearing within that section.

Citation Handling:

Case 1: Numeric citations
Examples: [12], [3,5,7], [8-10]

Rules:

Use original number as ref_id.

Expand ranges.

Record each reference separately.

Assign to the most fine-grained section node.

Case 2: Author-Year citations
Examples: (Smith, 2020), Smith et al., 2019

Rules:

Assign ref_id sequentially based on first appearance.

Reuse ref_id for repeated citations.

Reference Parsing Requirements:

For each ref_id extract:

title_raw

title (cleaned and repaired)

year (null if unavailable)

Title Repair Rules:

Fix broken words.

Fix concatenated words.

Do not hallucinate content.

Always preserve title_raw.

If uncertain, keep original.

Methodology Section Filtering:

Include only sections that describe technical taxonomy.

Exclude:

Introduction

Conclusion

Future Work

Acknowledgement

General Background (unless taxonomy included)

Pure Related Work (unless structured taxonomy)

Decision signals:

High citation density.

Contains keywords: method, model, framework, algorithm, taxonomy, category.

Explicit technical classification.

Paper storage format:

{
"ref_id": 12,
"title": "...",
"title_raw": "...",
"year": 2019,
"count": 3
}

Rules:

count = appearances within this section.

Same paper may appear in multiple sections.

Do not merge across sections.

Mandatory reference_index:

"reference_index": {
"12": {
"title": "...",
"title_raw": "...",
"year": 2019
}
}

Forbidden:

Do not organize by paper.

Do not flatten hierarchy.

Do not modify section labels.

Do not hallucinate titles.

Do not output explanations.

Error Handling:

If citation appears but reference missing -> year = null.

If mapping fails -> keep ref_id.

If title uncertain -> preserve raw.

Output:
JSON only.
No explanations.
No markdown.
No extra text."""


def _repair_truncated_json(text: str) -> str:
    """Attempt to repair truncated JSON by closing open brackets/braces."""
    # Track nesting
    stack = []
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ('{', '['):
            stack.append(ch)
        elif ch == '}' and stack and stack[-1] == '{':
            stack.pop()
        elif ch == ']' and stack and stack[-1] == '[':
            stack.pop()
    # Close open structures
    closing = []
    for opener in reversed(stack):
        if opener == '{':
            closing.append('}')
        elif opener == '[':
            closing.append(']')
    if closing:
        # Trim any trailing partial value (incomplete string or number)
        stripped = text.rstrip()
        # If ends mid-string, close the string
        if in_string:
            stripped += '"'
        # Remove trailing comma if present
        stripped = stripped.rstrip(',')
        return stripped + ''.join(closing)
    return text


def _trim_paper_text(text: str, max_chars: int = 120000) -> str:
    """For long papers, keep section structure + references section.

    Strategy: find the References/Bibliography section at the end,
    keep the main body (truncated if needed) + full references.
    """
    if len(text) <= max_chars:
        return text

    # Find the References / Bibliography section
    ref_pattern = re.compile(
        r'\n\s*(?:References|Bibliography|REFERENCES|BIBLIOGRAPHY)\s*\n',
        re.IGNORECASE,
    )
    matches = list(ref_pattern.finditer(text))
    if matches:
        # Use the last match (most likely the actual references section)
        ref_start = matches[-1].start()
        ref_section = text[ref_start:]
        body = text[:ref_start]

        # Budget: reserve space for references, give rest to body
        ref_budget = min(len(ref_section), 40000)
        body_budget = max_chars - ref_budget
        trimmed = body[:body_budget] + "\n...[body truncated]...\n" + ref_section[:ref_budget]
        return trimmed
    else:
        # No clear references section found, just truncate
        return text[:max_chars]


def _extract_taxonomy_with_llm(text: str, client: AzureOpenAI) -> dict | None:
    """Send full paper text to GPT-5.2 and get taxonomy JSON."""
    # Smart trim: keep section structure + references for long papers
    text = _trim_paper_text(text, max_chars=120000)

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-5.2",
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
                max_completion_tokens=65000,
            )
            finish_reason = response.choices[0].finish_reason
            content = response.choices[0].message.content
            if not content or not content.strip():
                if attempt < 2:
                    print(f"    Empty response (finish_reason={finish_reason}), retrying...")
                    time.sleep(3)
                    continue
                print(f"    LLM returned empty content on all attempts (finish_reason={finish_reason})")
                return None
            content = content.strip()
            # Strip markdown fences if present
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*", "", content)
                content = re.sub(r"\s*```$", "", content)
            # If truncated (finish_reason=length), try to repair JSON
            if finish_reason == "length":
                content = _repair_truncated_json(content)
            result = json.loads(content)
            if isinstance(result, dict) and "taxonomy_tree" in result:
                return result
            # Sometimes the model wraps it differently
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            if attempt < 2:
                time.sleep(2)
                continue
            print(f"    LLM returned invalid JSON on all attempts")
            return None
        except Exception as e:
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
                continue
            print(f"    LLM error: {e}")
            return None
    return None


# ── PDF text extraction ──────────────────────────────────────────────────────

def _extract_full_text(pdf_path: Path) -> str:
    """Extract all text from a PDF using pypdf."""
    reader = PdfReader(str(pdf_path))
    parts = []
    for page in reader.pages:
        text = page.extract_text() or ""
        parts.append(text)
    return "\n".join(parts)


# ── Helper: extract flat reference list from taxonomy ────────────────────────

def extract_paper_section_labels(taxonomy: dict) -> dict[str, dict]:
    """From a taxonomy JSON, build a mapping: ref_id -> {title, sections, paths, level}.

    Returns a dict keyed by ref_id (as string).
    """
    ref_index = taxonomy.get("reference_index", {})
    paper_info: dict[str, dict] = {}

    def _walk(node: dict, depth: int = 0):
        label = node.get("label", "")
        path = node.get("path", [])
        for paper in node.get("papers", []):
            rid = str(paper.get("ref_id", ""))
            title = paper.get("title") or ref_index.get(rid, {}).get("title", "")
            if rid not in paper_info:
                paper_info[rid] = {
                    "ref_id": rid,
                    "title": title,
                    "year": paper.get("year") or ref_index.get(rid, {}).get("year"),
                    "sections": [],
                    "paths": [],
                    "levels": [],
                }
            paper_info[rid]["sections"].append(label)
            paper_info[rid]["paths"].append(path)
            paper_info[rid]["levels"].append(depth)
        for child in node.get("children", []):
            _walk(child, depth + 1)

    for tree_node in taxonomy.get("taxonomy_tree", []):
        _walk(tree_node, depth=0)

    return paper_info


# ── Public API ───────────────────────────────────────────────────────────────

def parse_pdf_taxonomy(pdf_path: str | Path, client: AzureOpenAI | None = None) -> dict:
    """Extract methodology taxonomy from a survey PDF using GPT-5.2.

    Returns the full taxonomy dict with taxonomy_tree and reference_index.
    """
    pdf_path = Path(pdf_path)
    arxiv_id = pdf_path.stem.split("_")[0]

    if client is None:
        client = _get_llm_client()

    text = _extract_full_text(pdf_path)

    if not text.strip():
        return {
            "arxiv_id": arxiv_id,
            "survey_title": "",
            "taxonomy_tree": [],
            "reference_index": {},
            "error": "No text extracted from PDF",
        }

    taxonomy = _extract_taxonomy_with_llm(text, client)

    if taxonomy is None:
        return {
            "arxiv_id": arxiv_id,
            "survey_title": "",
            "taxonomy_tree": [],
            "reference_index": {},
            "error": "LLM failed to extract taxonomy",
        }

    taxonomy["arxiv_id"] = arxiv_id
    return taxonomy


def _get_all_llm_clients(n: int = 3) -> list[AzureOpenAI]:
    """Create LLM clients for all available Azure GPT-5.2 endpoints."""
    clients = []
    for idx in range(1, n + 1):
        try:
            endpoint = azure_gpt52_endpoint(idx).removesuffix("/openai")
            key = azure_gpt52_key(idx)
            if endpoint and key:
                clients.append(AzureOpenAI(
                    azure_endpoint=endpoint,
                    api_key=key,
                    api_version="2024-12-01-preview",
                ))
        except Exception:
            pass
    return clients if clients else [_get_llm_client()]


def _parse_one_pdf(
    pdf_path: Path,
    output_dir: Path,
    client: AzureOpenAI,
    idx: int,
    total: int,
) -> tuple[str, int]:
    """Parse a single PDF - designed for parallel execution."""
    arxiv_id = pdf_path.stem.split("_")[0]
    out_file = output_dir / f"{arxiv_id}.json"

    print(f"[{idx}/{total}] Parsing taxonomy for {arxiv_id}...", flush=True)
    result = parse_pdf_taxonomy(pdf_path, client=client)
    out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    n_refs = len(result.get("reference_index", {}))
    n_tree = len(result.get("taxonomy_tree", []))
    status = "ok" if "error" not in result else "FAILED"
    print(f"  [{status}] {arxiv_id}: {n_refs} refs, {n_tree} sections", flush=True)
    return arxiv_id, n_refs


def parse_all_pdfs(
    pdf_dir: Path = PDF_DIR,
    output_dir: Path = PARSED_DIR,
    paper_ids: list[str] | None = None,
    max_workers: int = 3,
) -> dict[str, int]:
    """Parse all survey PDFs and save taxonomy JSONs.

    Uses multiple Azure endpoints in parallel for speed.
    Returns dict mapping arxiv_id -> number of references in taxonomy.
    """
    ensure_dirs()
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if paper_ids:
        id_set = set(paper_ids)
        pdf_files = [p for p in pdf_files if p.stem.split("_")[0] in id_set]

    # Filter out already-parsed files
    to_process: list[tuple[int, Path]] = []
    stats = {}
    for i, pdf_path in enumerate(pdf_files):
        arxiv_id = pdf_path.stem.split("_")[0]
        out_file = output_dir / f"{arxiv_id}.json"

        if out_file.exists():
            try:
                existing = json.loads(out_file.read_text())
                if "error" not in existing and existing.get("taxonomy_tree"):
                    n = len(existing.get("reference_index", {}))
                    stats[arxiv_id] = n
                    continue
            except Exception:
                pass

        to_process.append((i + 1, pdf_path))

    print(f"Already done: {len(stats)}, to process: {len(to_process)}", flush=True)

    if not to_process:
        return stats

    # Get all available LLM clients
    clients = _get_all_llm_clients(max_workers)
    n_workers = len(clients)
    print(f"Using {n_workers} parallel workers", flush=True)

    total = len(pdf_files)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for job_idx, (orig_idx, pdf_path) in enumerate(to_process):
            client = clients[job_idx % n_workers]
            fut = executor.submit(
                _parse_one_pdf, pdf_path, output_dir, client, orig_idx, total,
            )
            futures[fut] = pdf_path.stem.split("_")[0]

        for fut in as_completed(futures):
            arxiv_id = futures[fut]
            try:
                _, n_refs = fut.result()
                stats[arxiv_id] = n_refs
            except Exception as e:
                print(f"  [ERROR] {arxiv_id}: {e}", flush=True)
                stats[arxiv_id] = 0

    return stats


if __name__ == "__main__":
    stats = parse_all_pdfs()
    total = sum(stats.values())
    print(f"\nParsed {len(stats)} papers, {total} total references in taxonomies")
    zero = [k for k, v in stats.items() if v == 0]
    if zero:
        print(f"WARNING: {len(zero)} papers with 0 references: {zero[:10]}")
