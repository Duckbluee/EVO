#!/usr/bin/env python3
"""简单读取 PyTorch Geometric .pt 图文件并输出基础统计信息"""

import argparse
import os
import torch


def load_graph(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph file not found: {path}")
    return torch.load(path)


def infer_num_nodes(data) -> int:
    num_nodes = getattr(data, 'num_nodes', None)
    if isinstance(num_nodes, int) and num_nodes > 0:
        return num_nodes
    if hasattr(data, 'x') and data.x is not None:
        return data.x.size(0)
    if hasattr(data, 'edge_index') and data.edge_index.numel() > 0:
        return int(data.edge_index.max().item()) + 1
    return 0


def infer_feature_dim(data) -> int:
    if hasattr(data, 'x') and data.x is not None:
        if data.x.dim() == 1:
            return 1
        return data.x.size(1)
    # 如果没有 x，但有其他一维属性，可视作 1 维特征
    if hasattr(data, 'years') and data.years is not None:
        return 1
    return 0


def describe_graph(path: str):
    data = load_graph(path)
    if not hasattr(data, 'edge_index'):
        raise ValueError("Loaded object does not contain edge_index")

    num_nodes = infer_num_nodes(data)
    num_edges = data.edge_index.size(1)
    feature_dim = infer_feature_dim(data)
    avg_degree = (num_edges / num_nodes) if num_nodes else 0.0

    print(f"图文件: {path}")
    print(f"节点数: {num_nodes}")
    print(f"边数(有向计): {num_edges}")
    print(f"特征维度: {feature_dim}")
    print(f"平均度数: {avg_degree:.3f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect PyG graph stats from .pt file")
    parser.add_argument('pt_file', help='Path to the PyTorch Geometric .pt file')
    return parser.parse_args()


def main():
    args = parse_args()
    describe_graph(args.pt_file)


if __name__ == '__main__':
    main()
