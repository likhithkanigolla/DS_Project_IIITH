"""Graph loading, generation and partitioning helpers."""
from typing import List, Tuple


def load_graph(path: str) -> Tuple[int, List[Tuple[int,int,float]]]:
    """Load graph from a simple edge list file.
    
    Expected format:
    Line 1: num_nodes num_edges
    Remaining lines: u v w (edge from u to v with weight w)
    """
    edges = []
    with open(path, 'r') as fh:
        lines = fh.readlines()
        if not lines:
            return 0, []
        
        # First line: num_nodes num_edges
        first_line = lines[0].strip().split()
        num_nodes = int(first_line[0])
        
        # Remaining lines: edges
        for line in lines[1:]:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [p.strip() for p in line.split()]
            if len(parts) < 3:
                continue
            u = int(parts[0])
            v = int(parts[1])
            w = float(parts[2])
            edges.append((u, v, w))
    
    return num_nodes, edges


def generate_graph(n: int, extra_edges: int = None, seed: int = 42):
    """Generate a connected random weighted graph with NetworkX.

    Returns (num_nodes, edges).
    """
    try:
        import networkx as nx
        import random
    except Exception as e:
        raise RuntimeError("NetworkX is required for graph generation: pip install networkx")

    random.seed(seed)
    # Start with a random tree to ensure connectivity (GUARANTEED connected graph)
    G = nx.random_labeled_tree(n, seed=seed)
    # Add some extra random edges to increase density
    if extra_edges is None:
        extra_edges = max(0, n // 2)
    nodes = list(G.nodes())
    for _ in range(extra_edges):
        u = random.choice(nodes)
        v = random.choice(nodes)
        if u == v or G.has_edge(u, v):
            continue
        G.add_edge(u, v)

    # Assign random weights
    for (u, v) in G.edges():
        G[u][v]['weight'] = random.uniform(1.0, 100.0)

    edges = []
    for (u, v, data) in G.edges(data=True):
        edges.append((int(u), int(v), float(data.get('weight', 1.0))))
    print(f"[graph_utils] Generated CONNECTED graph: {n} nodes, {len(edges)} edges (tree + {len(edges)-n+1} extra)", flush=True)
    return n, edges


def partition_graph(num_nodes: int, edges: List[Tuple[int,int,float]], rank: int, size: int):
    """Partition nodes evenly across ranks and return local nodes and local edges.

    Strategy: contiguous ranges of node ids assigned to ranks. An edge (u,v,w)
    is owned by the rank that owns u (source-based partition). This keeps
    edges distributed roughly by nodes.
    Returns (local_nodes, local_edges)
    """
    per = num_nodes // size
    extras = num_nodes % size
    offsets = []
    start = 0
    for i in range(size):
        cnt = per + (1 if i < extras else 0)
        offsets.append((start, start+cnt))
        start += cnt

    local_start, local_end = offsets[rank]
    local_nodes = list(range(local_start, local_end))

    local_edges = []
    for (u, v, w) in edges:
        if local_start <= u < local_end:
            local_edges.append((u, v, w))
        elif local_start <= v < local_end:
            # keep edges incident to local nodes as well (so we see edges where v is local)
            # ensure we store with local endpoint first for consistency
            local_edges.append((v, u, w))
    return local_nodes, local_edges
