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
    os.makedirs(out_dir, exist_ok=True)
    total_w = sum(w for (_, _, w) in mst_edges)
    
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