"""Build 1-hop ego citation graphs with taxonomy labels for root survey papers.

Algorithm (batch-optimised):
  1. Load the taxonomy JSON (from parse_references) for section/level labels.
  2. Look up the root (ego) paper on Semantic Scholar by arXiv ID.
  3. Get all references of the ego paper -> 1-hop nodes.
  4. Batch-fetch references of ALL 1-hop nodes (POST /paper/batch).
  5. Match 2-hop references against the 1-hop set -> internal edges.
  6. Match each 1-hop node to the taxonomy to get section labels.
  7. Save the densified, labeled ego-graph as a PyG Data object.
"""

import json
import torch
from pathlib import Path
from torch_geometric.data import Data

from .config import GRAPH_DIR, PARSED_DIR, ensure_dirs
from .s2_client import S2Client
from .parse_references import extract_paper_section_labels


def _normalize(title: str) -> str:
    """Normalize a title for fuzzy matching."""
    return " ".join(title.lower().strip().rstrip(".").split())


def _build_title_index(paper_info: dict) -> dict[str, str]:
    """Build a normalized-title -> ref_id mapping from taxonomy paper info."""
    idx = {}
    for rid, info in paper_info.items():
        t = info.get("title", "")
        if t:
            idx[_normalize(t)] = rid
    return idx


def build_ego_graph(
    arxiv_id: str,
    title: str,
    client: S2Client,
    parsed_dir: Path = PARSED_DIR,
    verbose: bool = True,
) -> Data | None:
    """Build a densified 1-hop ego citation graph with taxonomy labels.

    Returns a PyG Data object with:
        x              - dummy node features (N, 768)
        edge_index     - [2, E] directed citation edges
        arxiv_id       - list[str]  per-node arXiv IDs
        title          - list[str]  per-node titles
        abstract       - list[str]  per-node abstracts
        paper_id       - list[str]  per-node S2 paper IDs
        is_ego         - list[bool] True for ego node (index 0)
        section_labels - list[list[str]]  taxonomy sections each paper appears in
        section_paths  - list[list[list[str]]] full hierarchy paths
        section_levels - list[list[int]]  depth levels in taxonomy
    """
    # ── Load taxonomy labels if available ────────────────────────────────
    taxonomy_file = parsed_dir / f"{arxiv_id}.json"
    paper_info: dict[str, dict] = {}
    title_to_rid: dict[str, str] = {}

    if taxonomy_file.exists():
        try:
            taxonomy = json.loads(taxonomy_file.read_text())
            if taxonomy.get("taxonomy_tree"):
                paper_info = extract_paper_section_labels(taxonomy)
                title_to_rid = _build_title_index(paper_info)
                if verbose:
                    print(f"  taxonomy: {len(paper_info)} papers with section labels")
        except Exception as e:
            if verbose:
                print(f"  warning: could not load taxonomy: {e}")

    # ── Step 1: resolve the ego paper ────────────────────────────────────
    ego_paper = client.get_paper_by_arxiv(arxiv_id)
    if ego_paper is None:
        if verbose:
            print(f"  [SKIP] Cannot find {arxiv_id} on Semantic Scholar")
        return None

    ego_s2id = ego_paper["paperId"]
    ego_title = ego_paper.get("title") or title
    ego_abstract = ego_paper.get("abstract") or "N/A"

    # ── Step 2: get 1-hop references ─────────────────────────────────────
    one_hop_refs = client.get_references(ego_s2id)
    if not one_hop_refs:
        if verbose:
            print(f"  [SKIP] No references found for {arxiv_id}")
        return None

    # Deduplicate by paperId
    seen_ids: set[str] = set()
    unique_refs: list[dict] = []
    for ref in one_hop_refs:
        pid = ref.get("paperId")
        if pid and pid not in seen_ids:
            seen_ids.add(pid)
            unique_refs.append(ref)
    one_hop_refs = unique_refs

    if verbose:
        print(f"  1-hop: {len(one_hop_refs)} unique references")

    # ── Build node lists ─────────────────────────────────────────────────
    s2id_to_idx: dict[str, int] = {ego_s2id: 0}
    node_titles = [ego_title]
    node_abstracts = [ego_abstract]
    node_arxiv_ids = [arxiv_id]
    node_paper_ids = [ego_s2id]
    node_is_ego = [True]
    node_section_labels: list[list[str]] = [[]]
    node_section_paths: list[list[list[str]]] = [[]]
    node_section_levels: list[list[int]] = [[]]

    one_hop_pids: list[str] = []
    for ref in one_hop_refs:
        pid = ref["paperId"]
        if pid not in s2id_to_idx:
            s2id_to_idx[pid] = len(node_titles)
            ref_title = ref.get("title") or "Unknown"
            node_titles.append(ref_title)
            node_abstracts.append("N/A")
            ext = ref.get("externalIds") or {}
            node_arxiv_ids.append(ext.get("ArXiv", "N/A"))
            node_paper_ids.append(pid)
            node_is_ego.append(False)
            one_hop_pids.append(pid)

            # Match to taxonomy by normalized title
            norm_t = _normalize(ref_title)
            rid = title_to_rid.get(norm_t)
            if rid and rid in paper_info:
                info = paper_info[rid]
                node_section_labels.append(info["sections"])
                node_section_paths.append(info["paths"])
                node_section_levels.append(info["levels"])
            else:
                node_section_labels.append([])
                node_section_paths.append([])
                node_section_levels.append([])

    one_hop_id_set = set(one_hop_pids)

    matched = sum(1 for sl in node_section_labels[1:] if sl)
    if verbose and paper_info:
        print(f"  taxonomy match: {matched}/{len(one_hop_pids)} nodes have section labels")

    # ── Step 3: ego -> 1-hop edges ───────────────────────────────────────
    src_list: list[int] = []
    dst_list: list[int] = []

    for pid in one_hop_pids:
        src_list.append(0)
        dst_list.append(s2id_to_idx[pid])

    # ── Step 4: batch-fetch 2-hop & find internal edges ──────────────────
    if verbose:
        print(f"  fetching 2-hop references via batch API...")

    two_hop_map = client.batch_get_references(one_hop_pids)

    internal_edge_count = 0
    for pid in one_hop_pids:
        ref_ids = two_hop_map.get(pid, [])
        src_idx = s2id_to_idx[pid]
        for target_pid in ref_ids:
            if target_pid in one_hop_id_set and target_pid != pid:
                dst_idx = s2id_to_idx[target_pid]
                src_list.append(src_idx)
                dst_list.append(dst_idx)
                internal_edge_count += 1

    if verbose:
        print(f"  internal edges: {internal_edge_count}, "
              f"total edges: {len(src_list)}")

    # ── Step 5: build PyG Data ───────────────────────────────────────────
    num_nodes = len(node_titles)
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    x = torch.ones(num_nodes, 768)  # dummy features, replaced by embedder

    graph = Data(
        x=x,
        edge_index=edge_index,
        arxiv_id=node_arxiv_ids,
        title=node_titles,
        abstract=node_abstracts,
        paper_id=node_paper_ids,
        is_ego=node_is_ego,
        section_labels=node_section_labels,
        section_paths=node_section_paths,
        section_levels=node_section_levels,
    )

    return graph


def build_all_ego_graphs(
    papers: list[tuple[str, str]] | None = None,
    output_dir: Path = GRAPH_DIR,
    parsed_dir: Path = PARSED_DIR,
    resume: bool = True,
    verbose: bool = True,
) -> dict[str, str]:
    """Build ego-graphs for all root papers.

    Returns dict mapping arxiv_id -> status ("ok", "skipped", "failed").
    """
    from .config import load_root_papers
    ensure_dirs()
    output_dir.mkdir(parents=True, exist_ok=True)

    if papers is None:
        papers = load_root_papers()

    client = S2Client()
    results = {}

    for i, (arxiv_id, title) in enumerate(papers):
        out_path = output_dir / f"{arxiv_id}.pt"

        if resume and out_path.exists():
            if verbose:
                print(f"[{i+1}/{len(papers)}] {arxiv_id}: already exists, skipping")
            results[arxiv_id] = "skipped"
            continue

        if verbose:
            print(f"[{i+1}/{len(papers)}] Building ego-graph for {arxiv_id}: {title}")

        graph = build_ego_graph(arxiv_id, title, client,
                                parsed_dir=parsed_dir, verbose=verbose)

        if graph is None:
            results[arxiv_id] = "failed"
            continue

        torch.save(graph, out_path)
        results[arxiv_id] = "ok"

        if verbose:
            n = graph.x.size(0)
            e = graph.edge_index.size(1)
            print(f"  -> saved {out_path.name} ({n} nodes, {e} edges)\n")

    ok = sum(1 for v in results.values() if v == "ok")
    skipped = sum(1 for v in results.values() if v == "skipped")
    failed = sum(1 for v in results.values() if v == "failed")
    print(f"\nDone: ok={ok}, skipped={skipped}, failed={failed}")
    print(f"Total S2 API requests: {client.request_count}")

    return results


if __name__ == "__main__":
    build_all_ego_graphs()
