# EVO Dataset: Ego Citation Graphs with Taxonomy Labels

## Overview

**387 densified 1-hop ego citation graphs** built from survey papers. Each graph captures a survey paper (ego node) and its references (1-hop nodes), with internal citation edges discovered via 2-hop Semantic Scholar lookups. Nodes are annotated with methodology taxonomy labels extracted from the survey PDFs using GPT-5.2.

## Directory Structure

```
dataset/
├── graphs/                    # PyG Data objects (.pt files)
│   ├── 1812.08434.pt
│   ├── 1901.00596.pt
│   └── ...                    # 387 files, named by arXiv ID
├── parsed_references/         # LLM-extracted taxonomy JSONs
│   ├── 1812.08434.json
│   └── ...                    # 418 files
├── pdf/                       # Source survey PDFs
│   └── ...                    # 418 files
├── root_papers.txt            # List of 418 survey papers (arxiv_id, title)
├── s2_cache/                  # Semantic Scholar API response cache
└── README.md
```

## Loading a Graph

```python
import torch
from torch_geometric.data import Data

graph = torch.load("dataset/graphs/1812.08434.pt", weights_only=False)

print(graph.x.shape)           # [N, 768] dummy features
print(graph.edge_index.shape)  # [2, E] directed citation edges
print(len(graph.title))        # N titles
print(graph.title[0])          # ego paper title
```

## Node Features

Each graph is a `torch_geometric.data.Data` object with the following attributes:

| Attribute | Type | Coverage | Description |
|---|---|---|---|
| `x` | `Tensor [N, 768]` | 100% | Placeholder features (ones), to be replaced by embeddings |
| `edge_index` | `Tensor [2, E]` | 100% | Directed citation edges (src cites dst) |
| `title` | `list[str]` | 100% | Paper titles |
| `abstract` | `list[str]` | 83% | Paper abstracts (`"N/A"` if unavailable) |
| `paper_id` | `list[str]` | 100% | Semantic Scholar paper IDs |
| `arxiv_id` | `list[str]` | 67% | ArXiv IDs (`"N/A"` if not on arXiv) |
| `is_ego` | `list[bool]` | 100% | `True` for node 0 (the survey paper) |
| `section_labels` | `list[list[str]]` | 39% | Taxonomy section names where the paper is cited |
| `section_paths` | `list[list[list[str]]]` | 39% | Full root-to-leaf hierarchy paths |
| `section_levels` | `list[list[int]]` | 39% | Depth in taxonomy (0=top, 1=sub, 2=sub-sub, ...) |

### Node 0 is always the ego (survey) paper. Nodes 1..N-1 are its references.

## Edge Structure

Edges represent directed citations (source → target means source cites target):

- **Ego → reference edges**: the survey paper citing each of its references (N-1 edges)
- **Internal edges**: citations between reference papers discovered via 2-hop S2 lookup

```python
n_ego_edges = graph.x.size(0) - 1
n_internal = graph.edge_index.size(1) - n_ego_edges
print(f"Ego edges: {n_ego_edges}, Internal edges: {n_internal}")
```

## Taxonomy Labels

Papers cited in methodology sections of the survey have taxonomy labels extracted by GPT-5.2. A paper can appear in multiple sections.

```python
# Example: node with taxonomy labels
idx = 5
print(graph.title[idx])
# "Scaling Graph Neural Networks with Approximate PageRank"

print(graph.section_labels[idx])
# ['4.5. Large graphs']

print(graph.section_paths[idx])
# [['4. Variants considering graph type and scale', '4.5. Large graphs']]

print(graph.section_levels[idx])
# [1]
```

A paper appearing in multiple sections:
```python
print(graph.section_labels[idx])
# ['3.1.2. Basic spatial approaches', '3.4.1. Node sampling']

print(graph.section_paths[idx])
# [['3. Instantiations...', '3.1. Propagation modules', '3.1.2. Basic spatial approaches'],
#  ['3. Instantiations...', '3.4. Sampling modules', '3.4.1. Node sampling']]

print(graph.section_levels[idx])
# [2, 2]
```

Papers with empty `section_labels` (`[]`) are cited only in non-methodology sections (introduction, background, conclusion, etc.) which are excluded from taxonomy extraction.

## Graph Statistics

| Metric | Mean | Median | Min | Max |
|---|---|---|---|---|
| Nodes per graph | 178.4 | 159 | 14 | 609 |
| Edges per graph | 1,129.9 | 913 | 14 | 6,196 |
| Internal edges | 952.5 | 753 | — | — |
| Edge density (E/N) | 6.33 | — | — | — |

## Pipeline

The graphs were built using the preprocessing pipeline in `src/dataset/preprocess/`:

1. **PDF taxonomy extraction** (`parse_references.py`): GPT-5.2 extracts methodology taxonomy from each survey PDF
2. **Ego-graph construction** (`build_ego_graph.py`): Semantic Scholar batch API fetches references and discovers internal edges
3. **Abstract enrichment** (`fetch_abstracts.py`, `fetch_abstracts_crossref.py`): Batch-fetches abstracts from S2, OpenAlex, and Crossref

```bash
# Run full pipeline
conda run -n evo python -m src.dataset.preprocess.pipeline

# Fetch abstracts separately
conda run -n evo python -m src.dataset.preprocess.fetch_abstracts
conda run -n evo python -m src.dataset.preprocess.fetch_abstracts_crossref
```

## Data Sources

- **Survey PDFs**: Downloaded from arXiv
- **Citation data**: Semantic Scholar Graph API (batch endpoint)
- **Abstracts**: Semantic Scholar (81.3%), OpenAlex (0.7%), Crossref (1.0%)
- **Taxonomy labels**: GPT-5.2 extraction from survey PDF full text
