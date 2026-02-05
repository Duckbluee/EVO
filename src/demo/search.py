#!/usr/bin/env python3
"""
Citation Graph Builder v2.0
从JSON taxonomy文件构建Citation Graph，使用Semantic Scholar API获取论文信息。
输出PyTorch Geometric格式的图数据。

功能特性:
- 断点续传：支持中断后从上次位置继续
- 进度保存：每处理10篇论文自动保存
- 智能重试：网络错误自动重试
- 标题清洗：处理OCR导致的标题问题
- 双输出格式：同时输出PyG .pt文件和JSON文件

使用方法:
    python build_citation_graph_v2.py input.json output.json
    
    # 从断点继续
    python build_citation_graph_v2.py input.json output.json --resume
"""

import json
import re
import time
import urllib.parse
import urllib.request
import argparse
import os
from typing import Optional, Dict, List, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime

# ============== 配置 ==============

CONFIG = {
    'api_delay': 0.3,           # API请求间隔（秒），Semantic Scholar免费限制约100次/5分钟
    'max_retries': 3,           # 最大重试次数
    'save_interval': 10,        # 每处理N篇论文保存一次进度
    'match_threshold': 0.4,     # 标题匹配阈值（Jaccard相似度）
    'timeout': 30,              # 请求超时时间（秒）
}

# ============== 数据结构 ==============

@dataclass
class Paper:
    """论文数据结构"""
    ref_num: int                # -1 表示 root paper
    title: str                  # 原始标题
    title_cleaned: str          # 清洗后的标题
    year: str                   # 发表年份
    abstract: Optional[str] = None
    semantic_scholar_id: Optional[str] = None
    references: List[str] = field(default_factory=list)  # 该论文引用的其他论文title
    found: bool = False
    search_attempted: bool = False  # 是否已尝试搜索（用于断点续传）

    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> 'Paper':
        return cls(**d)

# ============== 标题清洗 ==============

def clean_title(title: str) -> str:
    """
    清洗论文标题，处理OCR问题
    """
    if not title:
        return ""
    
    # 移除尾部标点
    title = title.rstrip('.,;:')
    
    # 在小写字母后跟大写字母的地方插入空格
    title = re.sub(r'([a-z])([A-Z])', r'\1 \2', title)
    
    # 在逗号/句号/冒号/分号后面如果紧跟字母，添加空格
    title = re.sub(r'([,.:;])([a-zA-Z])', r'\1 \2', title)
    
    # 标准化多个空格为单个空格
    title = re.sub(r'\s+', ' ', title)
    
    return title.strip()

def normalize_title_for_matching(title: str) -> str:
    """
    将标题规范化用于匹配比较
    """
    title = title.lower()
    title = re.sub(r'[^a-z0-9\s]', '', title)
    title = re.sub(r'\s+', ' ', title)
    return title.strip()

def calculate_title_similarity(title1: str, title2: str) -> float:
    """
    计算两个标题的相似度（改进的Jaccard + 子序列匹配）
    """
    norm1 = normalize_title_for_matching(title1)
    norm2 = normalize_title_for_matching(title2)
    
    words1 = set(norm1.split())
    words2 = set(norm2.split())
    
    if not words1 or not words2:
        return 0.0
    
    # Jaccard相似度
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    jaccard = intersection / union if union > 0 else 0
    
    # 检查是否一个是另一个的子集（处理缩写标题）
    subset_bonus = 0
    if words1.issubset(words2) or words2.issubset(words1):
        subset_bonus = 0.2
    
    return min(1.0, jaccard + subset_bonus)

# ============== Semantic Scholar API ==============

BASE_URL = "https://api.semanticscholar.org/graph/v1"

def make_api_request(url: str, max_retries: int = None) -> Optional[dict]:
    """
    发送API请求，带重试机制
    """
    if max_retries is None:
        max_retries = CONFIG['max_retries']
    
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'CitationGraphBuilder/2.0 (Academic Research)')
            
            with urllib.request.urlopen(req, timeout=CONFIG['timeout']) as response:
                return json.loads(response.read().decode('utf-8'))
                
        except urllib.error.HTTPError as e:
            if e.code == 429:  # Rate limit
                wait_time = min(2 ** (attempt + 1), 60)  # 最多等60秒
                print(f"      ⏳ Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            elif e.code == 404:
                return None
            else:
                print(f"      ⚠️  HTTP Error {e.code}: {e.reason}")
                if attempt < max_retries - 1:
                    time.sleep(2)
        except urllib.error.URLError as e:
            print(f"      ⚠️  Network error: {e.reason}")
            if attempt < max_retries - 1:
                time.sleep(2)
        except Exception as e:
            print(f"      ⚠️  Error: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
    
    return None

def search_paper_by_title(title: str) -> Optional[dict]:
    """
    使用Semantic Scholar API按标题搜索论文
    """
    search_query = clean_title(title)
    encoded_query = urllib.parse.quote(search_query)
    
    # 搜索时获取更多候选结果
    url = f"{BASE_URL}/paper/search?query={encoded_query}&limit=5&fields=title,abstract,year,paperId"
    
    data = make_api_request(url)
    
    if not data or not data.get('data'):
        return None
    
    # 找最佳匹配
    best_match = None
    best_score = 0
    
    for paper in data['data']:
        score = calculate_title_similarity(search_query, paper.get('title', ''))
        if score > best_score:
            best_score = score
            best_match = paper
    
    # 只有分数足够高才返回
    if best_match and best_score >= CONFIG['match_threshold']:
        best_match['_match_score'] = best_score
        return best_match
    
    # 如果没有好的匹配，返回第一个结果但标记为低置信度
    if data['data']:
        result = data['data'][0]
        result['_match_score'] = best_score
        result['_low_confidence'] = True
        return result
    
    return None

def get_paper_references(paper_id: str) -> List[dict]:
    """
    获取论文的references列表
    """
    url = f"{BASE_URL}/paper/{paper_id}?fields=references.title,references.paperId"
    
    data = make_api_request(url)
    
    if data and data.get('references'):
        return [ref for ref in data['references'] if ref and ref.get('title')]
    
    return []

# ============== 核心逻辑 ==============

def load_taxonomy(filepath: str) -> Tuple[Paper, List[Paper], Set[str]]:
    """
    加载taxonomy JSON文件
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Root paper
    root_info = data['root_paper']
    root_paper = Paper(
        ref_num=-1,
        title=root_info['title'],
        title_cleaned=clean_title(root_info['title']),
        year=str(root_info.get('year', ''))
    )
    
    # Reference papers (去重)
    papers_dict: Dict[int, Paper] = {}
    
    def extract_papers(node):
        if 'papers' in node:
            for p in node['papers']:
                ref_num = p['ref_num']
                if ref_num not in papers_dict:
                    papers_dict[ref_num] = Paper(
                        ref_num=ref_num,
                        title=p['title'],
                        title_cleaned=clean_title(p['title']),
                        year=str(p.get('year', ''))
                    )
        if 'children' in node:
            for child in node['children']:
                extract_papers(child)
    
    for section in data['taxonomy']:
        extract_papers(section)
    
    # 按ref_num排序
    reference_papers = [papers_dict[k] for k in sorted(papers_dict.keys())]
    
    # 创建normalized titles集合
    all_titles = {normalize_title_for_matching(root_paper.title_cleaned)}
    for p in reference_papers:
        all_titles.add(normalize_title_for_matching(p.title_cleaned))
    
    return root_paper, reference_papers, all_titles

def fetch_paper_info(paper: Paper, all_titles: Set[str]) -> bool:
    """
    获取单篇论文的信息
    """
    # 搜索论文
    result = search_paper_by_title(paper.title_cleaned)
    paper.search_attempted = True
    
    if result:
        paper.found = True
        paper.semantic_scholar_id = result.get('paperId')
        paper.abstract = result.get('abstract') or ''
        
        # 获取references
        if paper.semantic_scholar_id:
            refs = get_paper_references(paper.semantic_scholar_id)
            for ref in refs:
                ref_title = ref.get('title', '')
                ref_normalized = normalize_title_for_matching(ref_title)
                if ref_normalized in all_titles:
                    paper.references.append(ref_title)
        
        confidence = "✓" if not result.get('_low_confidence') else "~"
        print(f"    {confidence} Found (score={result.get('_match_score', 0):.2f}), "
              f"abstract={len(paper.abstract)}chars, refs_in_set={len(paper.references)}")
        return True
    else:
        print(f"    ✗ Not found")
        return False

def build_edges(root_paper: Paper, reference_papers: List[Paper]) -> List[Tuple[int, int]]:
    """
    构建边列表（无向边）
    节点索引: 0 = root_paper, 1~N = reference_papers
    """
    # title -> index 映射
    title_to_idx = {normalize_title_for_matching(root_paper.title_cleaned): 0}
    for i, p in enumerate(reference_papers):
        title_to_idx[normalize_title_for_matching(p.title_cleaned)] = i + 1
    
    edges = set()
    
    # Root -> all references
    for i in range(len(reference_papers)):
        edges.add((0, i + 1))
    
    # Reference之间的引用关系
    all_papers = [root_paper] + reference_papers
    for i, paper in enumerate(all_papers):
        for ref_title in paper.references:
            ref_normalized = normalize_title_for_matching(ref_title)
            if ref_normalized in title_to_idx:
                j = title_to_idx[ref_normalized]
                if i != j:
                    edge = (min(i, j), max(i, j))
                    edges.add(edge)
    
    return sorted(list(edges))

# ============== 保存/加载进度 ==============

def save_progress(root_paper: Paper, reference_papers: List[Paper], 
                  progress_file: str):
    """保存当前进度"""
    data = {
        'timestamp': datetime.now().isoformat(),
        'root_paper': root_paper.to_dict(),
        'reference_papers': [p.to_dict() for p in reference_papers]
    }
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  📁 Progress saved to {progress_file}")

def load_progress(progress_file: str) -> Tuple[Paper, List[Paper]]:
    """加载之前的进度"""
    with open(progress_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    root_paper = Paper.from_dict(data['root_paper'])
    reference_papers = [Paper.from_dict(p) for p in data['reference_papers']]
    
    return root_paper, reference_papers

def save_final_output(root_paper: Paper, reference_papers: List[Paper],
                      edges: List[Tuple[int, int]], output_path: str):
    """
    保存最终输出（PyG格式 + JSON格式）
    """
    all_papers = [root_paper] + reference_papers
    
    # 准备节点数据
    nodes_data = []
    for i, p in enumerate(all_papers):
        nodes_data.append({
            'idx': i,
            'ref_num': p.ref_num,
            'title': p.title,
            'title_cleaned': p.title_cleaned,
            'year': p.year,
            'abstract': p.abstract or '',
            'found': p.found,
            'semantic_scholar_id': p.semantic_scholar_id,
            'references_in_set': p.references
        })
    
    # 尝试保存PyG格式
    pt_saved = False
    try:
        import torch
        from torch_geometric.data import Data
        
        # 构建双向edge_index
        if edges:
            edge_list = []
            for (i, j) in edges:
                edge_list.append([i, j])
                edge_list.append([j, i])
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # 创建Data对象
        pyg_data = Data(
            edge_index=edge_index,
            num_nodes=len(all_papers)
        )
        
        # 添加节点属性
        pyg_data.years = torch.tensor([int(p.year) if p.year.isdigit() else 0 for p in all_papers])
        pyg_data.found_mask = torch.tensor([p.found for p in all_papers])
        
        pt_path = output_path.replace('.json', '.pt')
        torch.save(pyg_data, pt_path)
        print(f"  ✅ Saved PyTorch Geometric data: {pt_path}")
        pt_saved = True
        
    except ImportError:
        print("  ⚠️  PyTorch Geometric not installed, skipping .pt output")
        print("     Install with: pip install torch torch_geometric")
    
    # 保存JSON格式
    output_data = {
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'root_paper_title': root_paper.title,
            'pyg_file_saved': pt_saved
        },
        'nodes': nodes_data,
        'edges': edges,
        'statistics': {
            'num_nodes': len(all_papers),
            'num_edges': len(edges),
            'num_undirected_edges': len(edges) * 2,  # PyG中的实际边数
            'total_papers': len(all_papers),
            'found_papers': sum(1 for p in all_papers if p.found),
            'papers_with_abstract': sum(1 for p in all_papers if p.abstract),
            'root_reference_edges': len(reference_papers),
            'inter_reference_edges': len(edges) - len(reference_papers)
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"  ✅ Saved JSON data: {output_path}")

# ============== 主函数 ==============

def main():
    parser = argparse.ArgumentParser(
        description='Build citation graph from taxonomy JSON',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python build_citation_graph_v2.py gnn1_taxonomy.json citation_graph.json
    python build_citation_graph_v2.py input.json output.json --resume
    python build_citation_graph_v2.py input.json output.json --delay 0.5
        """
    )
    parser.add_argument('input', help='Input taxonomy JSON file')
    parser.add_argument('output', help='Output file path (will create .json and .pt)')
    parser.add_argument('--resume', action='store_true', help='Resume from previous progress')
    parser.add_argument('--delay', type=float, default=CONFIG['api_delay'],
                        help=f'Delay between API calls in seconds (default: {CONFIG["api_delay"]})')
    
    args = parser.parse_args()
    
    CONFIG['api_delay'] = args.delay
    progress_file = args.output.replace('.json', '_progress.json')
    
    print("=" * 70)
    print("Citation Graph Builder v2.0")
    print("=" * 70)
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}")
    print(f"  API delay: {CONFIG['api_delay']}s")
    print("=" * 70)
    
    # 1. 加载数据
    print("\n[1/4] Loading taxonomy...")
    
    if args.resume and os.path.exists(progress_file):
        print(f"  📂 Resuming from {progress_file}")
        root_paper, reference_papers = load_progress(progress_file)
        # 重建all_titles
        all_titles = {normalize_title_for_matching(root_paper.title_cleaned)}
        for p in reference_papers:
            all_titles.add(normalize_title_for_matching(p.title_cleaned))
        
        # 统计已完成的数量
        completed = sum(1 for p in [root_paper] + reference_papers if p.search_attempted)
        print(f"  Loaded progress: {completed}/{len(reference_papers)+1} papers already processed")
    else:
        root_paper, reference_papers, all_titles = load_taxonomy(args.input)
    
    print(f"  Total papers: {len(reference_papers) + 1} (1 root + {len(reference_papers)} references)")
    
    # 2. 获取论文信息
    print(f"\n[2/4] Fetching paper information from Semantic Scholar...")
    
    all_papers = [root_paper] + reference_papers
    total = len(all_papers)
    found_count = 0
    
    for i, paper in enumerate(all_papers):
        if paper.search_attempted:
            if paper.found:
                found_count += 1
            continue
        
        paper_type = "root" if paper.ref_num == -1 else f"ref #{paper.ref_num}"
        print(f"\n  [{i+1}/{total}] ({paper_type}) {paper.title_cleaned[:55]}...")
        
        if fetch_paper_info(paper, all_titles):
            found_count += 1
        
        # 定期保存进度
        if (i + 1) % CONFIG['save_interval'] == 0:
            save_progress(root_paper, reference_papers, progress_file)
        
        time.sleep(CONFIG['api_delay'])
    
    print(f"\n  📊 Summary: Found {found_count}/{total} papers ({100*found_count/total:.1f}%)")
    
    # 保存最终进度
    save_progress(root_paper, reference_papers, progress_file)
    
    # 3. 构建边
    print("\n[3/4] Building citation edges...")
    edges = build_edges(root_paper, reference_papers)
    
    root_edges = len(reference_papers)
    inter_edges = len(edges) - root_edges
    print(f"  Total edges: {len(edges)}")
    print(f"    - Root → References: {root_edges}")
    print(f"    - Inter-reference:   {inter_edges}")
    
    # 4. 保存结果
    print("\n[4/4] Saving final output...")
    save_final_output(root_paper, reference_papers, edges, args.output)
    
    # 清理进度文件
    if os.path.exists(progress_file):
        os.remove(progress_file)
        print(f"  🗑️  Removed progress file")
    
    print("\n" + "=" * 70)
    print("✅ Done!")
    print("=" * 70)
    
    # 打印使用提示
    print(f"""
📖 Output files:
   - {args.output} (JSON with full text data)
   - {args.output.replace('.json', '.pt')} (PyTorch Geometric format)

📝 To load in PyTorch Geometric:
   import torch
   from torch_geometric.data import Data
   
   # Load graph structure
   data = torch.load('{args.output.replace('.json', '.pt')}')
   
   # Load text data
   import json
   with open('{args.output}') as f:
       text_data = json.load(f)
   
   # Access nodes
   for node in text_data['nodes']:
       print(node['title'], node['abstract'][:100])
""")

if __name__ == "__main__":
    main()