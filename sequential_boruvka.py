#!/usr/bin/env python3
"""Plain MPI MST: Sequential algorithm running on rank 0 only."""
from mpi4py import MPI
import time
from typing import List, Tuple
from dsu import DSU
from metrics import Metrics
import os
import json


def run_plain_mpi(num_nodes: int, edges: List[Tuple[int, int, float]], out_dir: str):
    """Run sequential Borůvka on rank 0; other ranks idle."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank != 0:
        return  # Non-zero ranks just exit
    
    # Ensure output directory exists before starting
    os.makedirs(out_dir, exist_ok=True)
    
    try:
        print("[plain-mpi] Starting sequential Borůvka on rank 0...")
        
        # Initialize DSU and metrics
        dsu = DSU()
        for i in range(num_nodes):
            dsu.make_set(i)
        
        metrics = Metrics()
        metrics.start()
        
        mst_edges: List[Tuple[int,int,float]] = []
        iteration_snapshots: List[List[Tuple[int,int,float]]] = []
        iteration_unions: List[List[Tuple[int,int]]] = []
        component_counts: List[int] = []
        iteration = 0
        
        # Main Borůvka loop
        while True:
            iteration += 1
            metrics.start_iteration()
            
            # Find minimum outgoing edge per component
            local_best = {}
            for (u, v, w) in edges:
                ru = dsu.find(u)
                rv = dsu.find(v)
                if ru == rv:
                    continue
                
                # Consider edge for component ru (u -> v)
                cur = local_best.get(ru)
                if cur is None or w < cur[2]:
                    local_best[ru] = (u, v, w)
                
                # Consider edge for component rv (v -> u)  
                cur = local_best.get(rv)
                if cur is None or w < cur[2]:
                    local_best[rv] = (v, u, w)
            
            union_pairs = []
            # Select edges and union components
            for (u, v, w) in local_best.values():
                ru = dsu.find(u)
                rv = dsu.find(v)
                if ru != rv:
                    dsu.union(ru, rv)
                    mst_edges.append((u, v, w))
                    union_pairs.append((ru, rv))
            
            iteration_unions.append(list(union_pairs))
            metrics.end_iteration()
            
            iteration_snapshots.append(list(mst_edges))
            comps = dsu.num_components()
            component_counts.append(comps)
            
            print(f"[plain-mpi] iter {iteration}, edges {len(mst_edges)}/{num_nodes-1}, components {comps}")
            
            if comps <= 1 or len(mst_edges) >= num_nodes - 1:
                break
        
        metrics.stop()
        
        # Save results
        total_w = sum(w for (_, _, w) in mst_edges)

        if not os.path.exists(os.path.join(out_dir, "mst_sequential.txt")):
            print("File not getting created!!")
        
        # MST file
        with open(os.path.join(out_dir, 'mst_sequential.txt'), 'w') as f:
            f.write("SEQUENTIAL BORŮVKA MINIMUM SPANNING TREE\n")
            f.write("=" * 45 + "\n")
            f.write("Edge List (Source -> Destination: Weight)\n")
            f.write("-" * 45 + "\n")
            for (u, v, w) in mst_edges:
                f.write(f"{u:3d} -> {v:3d}: {w:12.6f}\n")
            f.write("-" * 45 + "\n")
            f.write(f"Total MST Weight: {total_w:.6f}\n")
            f.write(f"Number of Edges: {len(mst_edges)}\n")
        
        # Save MST visualization (if graph is not too large)
        if num_nodes <= 50:
            try:
                import networkx as nx
                import matplotlib.pyplot as plt
                
                mst_viz_path = os.path.join(out_dir, 'mst_visualization.png')
                
                # Build full graph
                G = nx.Graph()
                for u, v, w in edges:
                    G.add_edge(u, v, weight=w)
                
                # Create figure
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
                
                # Use spring layout for both plots
                pos = nx.spring_layout(G, seed=42, k=1/num_nodes**0.5, iterations=50)
                
                # LEFT PLOT: Original graph with all edges
                ax1.set_title(f'Original Graph\n{num_nodes} nodes, {len(edges)} edges', 
                             fontsize=14, fontweight='bold', pad=15)
                nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                      node_size=500, alpha=0.9, linewidths=2, 
                                      edgecolors='darkblue', ax=ax1)
                nx.draw_networkx_edges(G, pos, alpha=0.3, width=1.5, edge_color='gray', ax=ax1)
                nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax1)
                ax1.axis('off')
                
                # RIGHT PLOT: MST only
                ax2.set_title(f'Minimum Spanning Tree\n{len(mst_edges)} edges, weight = {total_w:.2f}', 
                             fontsize=14, fontweight='bold', pad=15)
                
                # Draw all nodes
                nx.draw_networkx_nodes(G, pos, node_color='lightgreen', 
                                      node_size=500, alpha=0.9, linewidths=2, 
                                      edgecolors='darkgreen', ax=ax2)
                
                # Draw MST edges in red with thicker lines
                mst_edges_list = [(u, v) for u, v, w in mst_edges]
                nx.draw_networkx_edges(G, pos, edgelist=mst_edges_list, 
                                      edge_color='red', width=3, alpha=0.8, ax=ax2)
                
                # Draw node labels
                nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax2)
                
                # Draw edge weights for MST edges
                mst_edge_labels = {(u, v): f'{w:.1f}' for u, v, w in mst_edges}
                nx.draw_networkx_edge_labels(G, pos, mst_edge_labels, font_size=7,
                                            bbox=dict(boxstyle='round,pad=0.2', 
                                                     facecolor='yellow', alpha=0.7), ax=ax2)
                ax2.axis('off')
                
                plt.suptitle(f'Sequential Borůvka MST ({iteration} iterations)', 
                            fontsize=16, fontweight='bold', y=0.98)
                plt.tight_layout()
                
                # Save the figure
                plt.savefig(mst_viz_path, dpi=150, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                plt.close()
                
                print(f"[plain-mpi] MST visualization saved to: {mst_viz_path}")
            except Exception as e:
                print(f"[plain-mpi] Warning: Could not save MST visualization: {e}")
        elif num_nodes > 50:
            print(f"[plain-mpi] Skipping MST visualization (n={num_nodes} > 50 nodes)")
        
        # Metrics file
        summary = metrics.summary()
        with open(os.path.join(out_dir, 'metrics.txt'), 'w') as f:
            f.write("SEQUENTIAL BORŮVKA PERFORMANCE METRICS\n")
            f.write("=" * 40 + "\n")
            for k, v in summary.items():
                if k == 'total_time':
                    f.write(f"{k}: {float(v):.6f} seconds\n")
                elif k == 'avg_iter_time':
                    f.write(f"{k}: {float(v):.6f} seconds\n")
                else:
                    f.write(f"{k}: {v}\n")
        
        # Iteration log
        with open(os.path.join(out_dir, 'iteration_log.jsonl'), 'w') as f:
            for i, snap in enumerate(iteration_snapshots, 1):
                unions = iteration_unions[i-1] if i-1 < len(iteration_unions) else []
                comps = component_counts[i-1] if i-1 < len(component_counts) else None
                obj = {"iteration": i, "mst_size": len(snap), "components": comps, "unions": unions}
                f.write(json.dumps(obj) + "\n")
    
        print(f"[plain-mpi] Complete: {len(mst_edges)} edges, weight {total_w:.6f}")
        print(f"[plain-mpi] Results: {out_dir}")
        
        return {
            'mst_edges': mst_edges, 
            'total_weight': total_w, 
            'iterations': iteration,
            'metrics': summary,
            'snapshots': iteration_snapshots,
            'unions': iteration_unions,
            'components': component_counts
        }
    
    except Exception as e:
        # If anything fails, still create output files with error information
        print(f"[plain-mpi] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Create error files so main.py can continue
        with open(os.path.join(out_dir, 'mst_sequential.txt'), 'w') as f:
            f.write("SEQUENTIAL BORŮVKA - ERROR OCCURRED\n")
            f.write("=" * 45 + "\n")
            f.write(f"Error: {str(e)}\n")
        
        with open(os.path.join(out_dir, 'metrics.txt'), 'w') as f:
            f.write("SEQUENTIAL BORŮVKA - ERROR\n")
            f.write("total_time: 0.0 seconds\n")
            f.write("iterations: 0\n")
        
        with open(os.path.join(out_dir, 'iteration_log.jsonl'), 'w') as f:
            f.write(json.dumps({"error": str(e)}) + "\n")
        
        print(f"[plain-mpi] Error files created in: {out_dir}")
        return None
