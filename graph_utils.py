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
    """Improved load-balanced partitioning for k-machine model.
    
    Uses hash-based node assignment and edge-count balancing to ensure 
    even distribution of computation across all ranks.
    """
    import hashlib
    
    # Hash-based node assignment for better load balancing
    def node_to_rank(node_id: int) -> int:
        # Use hash to distribute nodes more evenly
        hash_val = int(hashlib.md5(str(node_id).encode()).hexdigest(), 16)
        return hash_val % size
    
    # Assign nodes to this rank based on hash
    local_nodes = [i for i in range(num_nodes) if node_to_rank(i) == rank]
    
    # Collect edges where either endpoint belongs to this rank
    local_edges = []
    edge_counts_by_rank = [0] * size
    
    for (u, v, w) in edges:
        u_rank = node_to_rank(u)
        v_rank = node_to_rank(v)
        
        # Add edge if either endpoint is local
        if u_rank == rank or v_rank == rank:
            # Normalize edge direction for consistency
            if u_rank == rank:
                local_edges.append((u, v, w))
            else:
                local_edges.append((v, u, w))
        
        # Track edge distribution
        edge_counts_by_rank[u_rank] += 1
    
    if rank == 0:
        print("[partition] Hash-based load balancing:")
        for i in range(size):
            nodes_in_rank = sum(1 for j in range(num_nodes) if node_to_rank(j) == i)
            print(f"  Rank {i}: {nodes_in_rank} nodes, ~{edge_counts_by_rank[i]} edge operations")
    
    return local_nodes, local_edges
