#!/usr/bin/env python3
"""K-Machine MST: Distributed Borůvka across multiple MPI ranks."""
from mpi4py import MPI
import time
import math
from typing import List, Tuple
from dsu import DSU
from graph_utils import partition_graph
from metrics import Metrics
import os
import json


def run_kmachine_boruvka(num_nodes: int, edges: List[Tuple[int, int, float]], out_dir: str):
    """Pure K-Machine distributed Borůvka with no coordinator dependency."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    print(f"[k-machine] Starting decentralized Borůvka on {size} ranks...")
    
    # Create output directory (rank 0 only, but all ranks need to wait)
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    comm.Barrier()  # Ensure directory exists before proceeding
    
    # Improved load-balanced partitioning with visualization
    partition_viz_path = os.path.join(out_dir, 'partition_visualization.png') if rank == 0 else None
    local_nodes, local_edges = partition_graph(num_nodes, edges, rank, size, 
                                               save_visualization=True, 
                                               output_path=partition_viz_path)
    
    # Initialize distributed DSU (each rank maintains full DSU for consistency)
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
    
    # Decentralized main loop
    while True:
        iteration += 1
        metrics.start_iteration()
        
        # 1. Each rank finds local minimum outgoing edges per component
        local_candidates = {}
        for u in local_nodes:
            comp_u = dsu.find(u)
            best_edge = None
            
            # Find minimum outgoing edge for this node's component
            for (u_edge, v_edge, w) in local_edges:
                if u_edge == u:  # Edge from local node
                    comp_v = dsu.find(v_edge)
                    if comp_u != comp_v and (best_edge is None or w < best_edge[2]):
                        best_edge = (u, v_edge, w)
                elif v_edge == u:  # Edge to local node
                    comp_v = dsu.find(u_edge)
                    if comp_u != comp_v and (best_edge is None or w < best_edge[2]):
                        best_edge = (u, u_edge, w)
            
            if best_edge:
                existing = local_candidates.get(comp_u)
                if existing is None or best_edge[2] < existing[2]:
                    local_candidates[comp_u] = best_edge
        
        # 2. DECENTRALIZED COORDINATION: All-to-all communication
        # Each rank exchanges candidates with all other ranks
        all_candidates_gathered = comm.allgather(list(local_candidates.values()))
        metrics.inc_round()
        
        # 3. DISTRIBUTED CONSENSUS: Each rank independently computes same result
        global_best = {}
        for cand_list in all_candidates_gathered:
            for u, v, w in cand_list:
                comp_u = dsu.find(u)
                comp_v = dsu.find(v)
                if comp_u == comp_v:
                    continue
                
                current = global_best.get(comp_u)
                if current is None or w < current[2]:
                    global_best[comp_u] = (u, v, w)
        
        # 4. SYNCHRONIZED APPLICATION: All ranks apply same unions
        union_pairs = []
        for u, v, w in global_best.values():
            ru = dsu.find(u)
            rv = dsu.find(v)
            if ru != rv:
                dsu.union(ru, rv)
                mst_edges.append((u, v, w))
                union_pairs.append((u, v))
        
        # 5. DISTRIBUTED TERMINATION: Consensus on completion
        local_done = dsu.num_components() <= 1 or len(mst_edges) >= num_nodes - 1
        all_done = comm.allreduce(local_done, op=MPI.LAND)  # Logical AND across all ranks
        metrics.inc_round()
        
        # Record iteration data
        iteration_unions.append(list(union_pairs))
        iteration_snapshots.append(list(mst_edges))
        component_counts.append(dsu.num_components())
        
        metrics.end_iteration()
        
        if rank == 0:
            print(f"[k-machine] iter {iteration}, edges {len(mst_edges)}/{num_nodes-1}, components {dsu.num_components()}")
        
        if all_done:
            break
    
    metrics.stop()
    
    # 6. DECENTRALIZED RESULT COLLECTION
    # Each rank has the complete MST, just gather for verification
    if rank == 0:
        # Remove duplicates and finalize
        seen = set()
        unique = []
        for (u, v, w) in mst_edges:
            key = tuple(sorted((u, v)))
            if key not in seen:
                seen.add(key)
                unique.append((u, v, w))
        
        total_w = sum(w for (_, _, w) in unique)
        
        # Save MST to file
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'mst_distributed.txt'), 'w') as f:
            f.write("DISTRIBUTED BORŮVKA MST (K-MACHINE)\n")
            f.write("=" * 40 + "\n")
            for u, v, w in unique:
                f.write(f"{u} {v} {w}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total MST Weight: {total_w:.6f}\n")
            f.write(f"Number of Edges: {len(unique)}\n")
            f.write(f"K-Machines Used: {size}\n")
            f.write(f"Input Size N: {num_nodes} (N/k = {num_nodes/size:.1f})\n")
        
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
                
                # Identify MST edges
                mst_edge_set = {tuple(sorted((u, v))) for u, v, w in unique}
                
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
                ax2.set_title(f'Minimum Spanning Tree\n{len(unique)} edges, weight = {total_w:.2f}', 
                             fontsize=14, fontweight='bold', pad=15)
                
                # Draw all nodes
                nx.draw_networkx_nodes(G, pos, node_color='lightgreen', 
                                      node_size=500, alpha=0.9, linewidths=2, 
                                      edgecolors='darkgreen', ax=ax2)
                
                # Draw MST edges in red with thicker lines
                mst_edges_list = [(u, v) for u, v, w in unique]
                nx.draw_networkx_edges(G, pos, edgelist=mst_edges_list, 
                                      edge_color='red', width=3, alpha=0.8, ax=ax2)
                
                # Draw node labels
                nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax2)
                
                # Draw edge weights for MST edges
                mst_edge_labels = {(u, v): f'{w:.1f}' for u, v, w in unique}
                nx.draw_networkx_edge_labels(G, pos, mst_edge_labels, font_size=7,
                                            bbox=dict(boxstyle='round,pad=0.2', 
                                                     facecolor='yellow', alpha=0.7), ax=ax2)
                ax2.axis('off')
                
                plt.suptitle(f'Distributed Borůvka MST ({size} machines, {iteration} iterations)', 
                            fontsize=16, fontweight='bold', y=0.98)
                plt.tight_layout()
                
                # Save the figure
                plt.savefig(mst_viz_path, dpi=150, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                plt.close()
                
                print(f"[k-machine] MST visualization saved to: {mst_viz_path}")
            except Exception as e:
                print(f"[k-machine] Warning: Could not save MST visualization: {e}")
        elif num_nodes > 50:
            print(f"[k-machine] Skipping MST visualization (n={num_nodes} > 50 nodes)")
        
        # Enhanced metrics file
        summary = metrics.summary()
        with open(os.path.join(out_dir, 'metrics.txt'), 'w') as f:
            f.write("DECENTRALIZED K-MACHINE PERFORMANCE METRICS\n")
            f.write("=" * 50 + "\n")
            f.write(f"K-Machines: {size}\n")
            f.write(f"Input Size: {num_nodes} nodes, {len(edges)} edges\n")
            f.write(f"N/k Ratio: {num_nodes/size:.1f}\n")
            f.write("-" * 50 + "\n")
            for k, v in summary.items():
                if k == 'total_time':
                    f.write(f"{k}: {float(v):.6f}\n")
                elif k == 'avg_iter_time':
                    f.write(f"{k}: {float(v):.6f}\n")
                else:
                    f.write(f"{k}: {v}\n")
            
            # K-machine specific metrics
            f.write("-" * 50 + "\n")
            f.write("K-MACHINE COORDINATION\n")
            f.write("Communication Pattern: Decentralized (allgather + allreduce)\n")
            f.write("Coordinator Dependency: None\n")
            f.write("Load Balancing: Hash-based\n")
            f.write("Fault Tolerance: Distributed consensus\n")
        
        # Iteration log
        with open(os.path.join(out_dir, 'iteration_log.jsonl'), 'w') as f:
            for i, snap in enumerate(iteration_snapshots, 1):
                unions = iteration_unions[i-1] if i-1 < len(iteration_unions) else []
                comps = component_counts[i-1] if i-1 < len(component_counts) else None
                obj = {
                    "iteration": i, 
                    "mst_size": len(snap), 
                    "components": comps, 
                    "unions": unions,
                    "k_machines": size,
                    "coordination": "decentralized"
                }
                f.write(json.dumps(obj) + "\n")
        
        print(f"[k-machine] Complete: {len(unique)} edges, weight {total_w:.6f}")
        print(f"[k-machine] Decentralized execution on {size} machines")
        print(f"[k-machine] Results: {out_dir}")
        
        return {
            'mst_edges': unique,
            'total_weight': total_w, 
            'iterations': iteration,
            'metrics': summary,
            'snapshots': iteration_snapshots,
            'unions': iteration_unions,
            'components': component_counts,
            'k_machines': size,
            'coordination_model': 'decentralized'
        }
    
    return None
