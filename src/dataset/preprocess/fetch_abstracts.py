"""Batch-fetch abstracts from Semantic Scholar and update saved PyG graphs.

Uses POST /paper/batch (up to 500 IDs per call) so each graph needs
only 1 API call to get all abstracts.
"""

import torch
from pathlib import Path

from .config import GRAPH_DIR, ensure_dirs
from .s2_client import S2Client


def update_graph_abstracts(
    graph_dir: Path = GRAPH_DIR,
    verbose: bool = True,
) -> dict[str, dict]:
    """Fetch abstracts for all nodes in all saved graphs.

    Returns stats dict per graph.
    """
    ensure_dirs()
    client = S2Client()
    graph_files = sorted(graph_dir.glob("*.pt"))

    stats = {}
    for i, gf in enumerate(graph_files):
        arxiv_id = gf.stem
        graph = torch.load(gf, weights_only=False)
        n = graph.x.size(0)

        # Collect paper IDs that need abstracts
        pids_to_fetch = []
        for j in range(n):
            if graph.abstract[j] == "N/A" and graph.paper_id[j]:
                pids_to_fetch.append(graph.paper_id[j])

        if not pids_to_fetch:
            if verbose:
                print(f"[{i+1}/{len(graph_files)}] {arxiv_id}: all abstracts present, skip")
            stats[arxiv_id] = {"total": n, "fetched": 0, "found": 0}
            continue

        if verbose:
            print(f"[{i+1}/{len(graph_files)}] {arxiv_id}: fetching {len(pids_to_fetch)} abstracts...",
                  end="", flush=True)

        abstracts = client.batch_get_abstracts(pids_to_fetch)

        # Update graph
        found = 0
        for j in range(n):
            pid = graph.paper_id[j]
            if pid in abstracts and abstracts[pid]:
                graph.abstract[j] = abstracts[pid]
                found += 1

        # Save updated graph
        torch.save(graph, gf)

        if verbose:
            print(f" {found}/{len(pids_to_fetch)} found")

        stats[arxiv_id] = {"total": n, "fetched": len(pids_to_fetch), "found": found}

    total_fetched = sum(s["found"] for s in stats.values())
    total_nodes = sum(s["total"] for s in stats.values())
    print(f"\nDone: {total_fetched}/{total_nodes} nodes now have abstracts")
    print(f"S2 API requests: {client.request_count}")

    return stats


if __name__ == "__main__":
    update_graph_abstracts()
