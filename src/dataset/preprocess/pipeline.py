#!/usr/bin/env python3
"""Main pipeline for preprocessing survey papers into ego-graphs.

Usage (from /home/ubuntu/EVO, with conda env evo):

    # Run full pipeline (parse PDFs + build ego-graphs)
    python -m src.dataset.preprocess.pipeline

    # Only parse PDFs (extract references via LLM)
    python -m src.dataset.preprocess.pipeline --step parse

    # Only build ego-graphs (Semantic Scholar API)
    python -m src.dataset.preprocess.pipeline --step graph

    # Process specific papers
    python -m src.dataset.preprocess.pipeline --papers 2401.05459 2310.15654

    # Reprocess everything (ignore cached results)
    python -m src.dataset.preprocess.pipeline --no-resume
"""

import argparse
import sys
import time

from .config import GRAPH_DIR, PARSED_DIR, PDF_DIR, load_root_papers, ensure_dirs
from .parse_references import parse_all_pdfs
from .build_ego_graph import build_all_ego_graphs


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess survey PDFs into citation ego-graphs."
    )
    parser.add_argument(
        "--step",
        choices=["all", "parse", "graph"],
        default="all",
        help="Which pipeline step to run (default: all).",
    )
    parser.add_argument(
        "--papers",
        nargs="*",
        help="Specific arXiv IDs to process (default: all from root_papers.txt).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Reprocess papers even if output already exists.",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity.",
    )
    args = parser.parse_args()

    ensure_dirs()
    all_papers = load_root_papers()

    # Filter to specific papers if requested
    if args.papers:
        paper_set = set(args.papers)
        papers = [(aid, t) for aid, t in all_papers if aid in paper_set]
        if not papers:
            print(f"No matching papers found for: {args.papers}")
            sys.exit(1)
    else:
        papers = all_papers

    paper_ids = [aid for aid, _ in papers]
    print(f"Processing {len(papers)} papers")
    t0 = time.time()

    # ── Step 1: Parse PDFs → extract reference titles via LLM ────────────
    if args.step in ("all", "parse"):
        print("\n=== Step 1: Extracting references from PDFs (GPT-5.2) ===")
        stats = parse_all_pdfs(
            pdf_dir=PDF_DIR,
            output_dir=PARSED_DIR,
            paper_ids=paper_ids,
        )
        total_refs = sum(stats.values())
        print(f"Parsed {len(stats)} papers, {total_refs} total references")

    # ── Step 2: Build ego-graphs via Semantic Scholar ────────────────────
    if args.step in ("all", "graph"):
        print("\n=== Step 2: Building ego-graphs via Semantic Scholar ===")
        results = build_all_ego_graphs(
            papers=papers,
            output_dir=GRAPH_DIR,
            resume=not args.no_resume,
            verbose=not args.quiet,
        )

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
