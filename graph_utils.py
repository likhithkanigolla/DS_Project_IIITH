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


def generate_graph(n: int, extra_edges: int = None, seed: int = 42, save_visualization: bool = True, output_path = None):
    """Generate a connected random weighted graph with NetworkX.

    Args:
        n: Number of nodes
        extra_edges: Number of extra edges beyond the spanning tree (default: n//2)
        seed: Random seed for reproducibility
        save_visualization: If True and n <= 50, save a visualization of the graph
        output_path: Path to save the visualization (default: 'graph_visualization.png')

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
    
    # Save visualization if requested and graph is not too large
    if save_visualization and n <= 50:
        try:
            import matplotlib.pyplot as plt
            
            if output_path is None:
                output_path = 'graph_visualization.png'
            
            # Create figure with good size
            plt.figure(figsize=(12, 10))
            
            # Use spring layout for better visualization
            pos = nx.spring_layout(G, seed=seed, k=1/n**0.5, iterations=50)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                   node_size=500, alpha=0.9, linewidths=2, 
                                   edgecolors='darkblue')
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, alpha=0.5, width=2, edge_color='gray')
            
            # Draw node labels
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
            
            # Draw edge weights
            edge_labels = {(u, v): f'{data["weight"]:.1f}' for u, v, data in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, 
                                         bbox=dict(boxstyle='round,pad=0.3', 
                                                  facecolor='yellow', alpha=0.7))
            
            plt.title(f'Generated Graph: {n} nodes, {len(edges)} edges', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.axis('off')
            plt.tight_layout()
            
            # Save the figure
            plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            print(f"[graph_utils] Graph visualization saved to: {output_path}")
        except Exception as e:
            print(f"[graph_utils] Warning: Could not save visualization: {e}")
    elif save_visualization and n > 50:
        print(f"[graph_utils] Skipping visualization (n={n} > 50 nodes)")
    
    return n, edges


def partition_graph(num_nodes: int, edges: List[Tuple[int,int,float]], rank: int, size: int, 
                    save_visualization: bool = True, output_path = None):
    """Improved load-balanced partitioning for k-machine model.
    
    Uses hash-based node assignment and edge-count balancing to ensure 
    even distribution of computation across all ranks.
    
    Args:
        num_nodes: Total number of nodes in the graph
        edges: List of edges (u, v, weight)
        rank: Current MPI rank
        size: Total number of MPI ranks
        save_visualization: If True and rank==0 and num_nodes<=50, save partition visualization
        output_path: Path to save the visualization (only used by rank 0)
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
        
        # Save partition visualization (only rank 0 creates it)
        if save_visualization and num_nodes <= 50:
            try:
                import networkx as nx
                import matplotlib.pyplot as plt
                import matplotlib.colors as mcolors
                
                if output_path is None:
                    output_path = 'partition_visualization.png'
                
                # Build NetworkX graph
                G = nx.Graph()
                for u, v, w in edges:
                    G.add_edge(u, v, weight=w)
                
                # Create figure
                plt.figure(figsize=(14, 10))
                
                # Use spring layout with same seed for consistency
                pos = nx.spring_layout(G, seed=42, k=1/num_nodes**0.5, iterations=50)
                
                # Color nodes by rank assignment
                # Use a colormap with distinct colors
                cmap = plt.cm.get_cmap('tab10' if size <= 10 else 'tab20')
                node_colors = [cmap(node_to_rank(node) / max(size-1, 1)) for node in G.nodes()]
                
                # Draw edges first
                nx.draw_networkx_edges(G, pos, alpha=0.3, width=1.5, edge_color='gray')
                
                # Draw nodes colored by rank
                nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                       node_size=600, alpha=0.9, linewidths=2.5, 
                                       edgecolors='black')
                
                # Draw node labels
                nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
                
                # Create legend showing rank assignments
                from matplotlib.patches import Patch
                legend_elements = []
                for i in range(size):
                    nodes_in_rank = sum(1 for j in range(num_nodes) if node_to_rank(j) == i)
                    color = cmap(i / max(size-1, 1))
                    legend_elements.append(
                        Patch(facecolor=color, edgecolor='black', linewidth=1.5,
                              label=f'Rank {i}: {nodes_in_rank} nodes')
                    )
                
                plt.legend(handles=legend_elements, loc='upper left', 
                          fontsize=10, framealpha=0.9, title='Machine Assignment',
                          title_fontsize=12)
                
                plt.title(f'Graph Partitioning across {size} Machines\n'
                         f'{num_nodes} nodes distributed via hash-based assignment', 
                         fontsize=16, fontweight='bold', pad=20)
                plt.axis('off')
                plt.tight_layout()
                
                # Save the figure
                plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                plt.close()
                
                print(f"[partition] Partition visualization saved to: {output_path}")
            except Exception as e:
                print(f"[partition] Warning: Could not save partition visualization: {e}")
        elif save_visualization and num_nodes > 50:
            print(f"[partition] Skipping partition visualization (n={num_nodes} > 50 nodes)")
    
    return local_nodes, local_edges
