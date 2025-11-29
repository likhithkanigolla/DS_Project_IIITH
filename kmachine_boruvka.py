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
    """Run distributed Borůvka across ranks (K-Machine)."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    print(f"[k-machine] Starting distributed Borůvka on {size} ranks...")
    
    # Partition graph among ranks
    local_nodes, local_edges = partition_graph(num_nodes, edges, rank, size)
    
    # Initialize local and global DSU
    local_dsu = DSU()
    for u in local_nodes:
        local_dsu.make_set(u)
    
    if rank == 0:
        global_dsu = DSU()
        for i in range(num_nodes):
            global_dsu.make_set(i)
    
    metrics = Metrics()
    metrics.start()
    
    mst_edges: List[Tuple[int,int,float]] = []
    iteration_snapshots: List[List[Tuple[int,int,float]]] = []
    iteration_unions: List[List[Tuple[int,int]]] = []
    component_counts: List[int] = []
    iteration = 0
    
    # Main distributed loop
    while True:
        iteration += 1
        metrics.start_iteration()
        
        # 1. Each rank finds minimum outgoing edges for its local nodes
        local_candidates = {}
        for u in local_nodes:
            comp_u = local_dsu.find(u)
            best_edge = None
            for (u_edge, v_edge, w) in local_edges:
                if u_edge == u:  # Edge starts from local node u
                    comp_v = local_dsu.find(v_edge)
                    if comp_u != comp_v:  # Outgoing edge
                        if best_edge is None or w < best_edge[2]:
                            best_edge = (u, v_edge, w)
                elif v_edge == u:  # Edge ends at local node u  
                    comp_v = local_dsu.find(u_edge)
                    if comp_u != comp_v:  # Outgoing edge
                        if best_edge is None or w < best_edge[2]:
                            best_edge = (u, u_edge, w)
            
            if best_edge:
                local_candidates[comp_u] = best_edge
        
        # 2. Gather all candidates to rank 0
        all_candidates = comm.gather(list(local_candidates.values()), root=0)
        metrics.inc_round()
        
        union_pairs = []
        done = False
        if rank == 0:
            # 3. Select one minimum edge per component globally
            global_best = {}
            for cand_list in all_candidates:
                for u, v, w in cand_list:
                    comp_u = global_dsu.find(u)
                    comp_v = global_dsu.find(v)
                    if comp_u == comp_v:
                        continue  # Skip if already connected
                    
                    # Only consider this edge for component comp_u
                    current = global_best.get(comp_u)
                    if current is None or w < current[2]:
                        global_best[comp_u] = (u, v, w)
            
            # 4. Perform unions and add to MST
            for u, v, w in global_best.values():
                ru = global_dsu.find(u)
                rv = global_dsu.find(v)
                if ru != rv:
                    global_dsu.union(ru, rv)
                    mst_edges.append((u, v, w))
                    # Store the actual nodes that need to be unioned, not representatives
                    union_pairs.append((u, v))
            
            # Check termination
            if global_dsu.num_components() <= 1 or len(mst_edges) >= num_nodes - 1:
                done = True
        
        # Record unions before broadcast
        if rank == 0:
            iteration_unions.append(list(union_pairs))
        
        # 5. Broadcast union operations to all ranks
        union_pairs = comm.bcast(union_pairs, root=0)
        done = comm.bcast(done, root=0)
        metrics.inc_round()
        
        # 6. Apply unions locally  
        for (a, b) in union_pairs:
            # Union the representatives directly without resetting
            local_dsu.union(a, b)
        
        metrics.end_iteration()
        
        if rank == 0:
            iteration_snapshots.append(list(mst_edges))
            comps = global_dsu.num_components()
            component_counts.append(comps)
            print(f"[k-machine] iter {iteration}, edges {len(mst_edges)}/{num_nodes-1}, components {comps}")
        
        if done:
            break
    
    metrics.stop()
    
    # Gather final MST to rank 0
    local_mst = mst_edges if rank == 0 else []
    all_mst = comm.gather(local_mst, root=0)
    
    if rank == 0:
        # Process final results
        final_edges = []
        for lst in all_mst:
            final_edges.extend(lst)
        
        # Remove duplicates
        seen = set()
        unique = []
        for (u, v, w) in final_edges:
            key = tuple(sorted((u, v)))
            if key in seen:
                continue
            seen.add(key)
            unique.append((u, v, w))
        
        total_w = sum(w for (_, _, w) in unique)
        
        # Save results
        os.makedirs(out_dir, exist_ok=True)
        
        # MST file
        with open(os.path.join(out_dir, 'mst_kmachine.txt'), 'w') as f:
            f.write('u v w\n')
            for (u, v, w) in unique:
                f.write(f"{u} {v} {w}\n")
            f.write(f"Total weight: {total_w}\n")
        
        # Metrics file
        summary = metrics.summary()
        with open(os.path.join(out_dir, 'metrics.txt'), 'w') as f:
            for k, v in summary.items():
                f.write(f"{k}: {v}\n")
        
        # Iteration log
        with open(os.path.join(out_dir, 'iteration_log.jsonl'), 'w') as f:
            for i, snap in enumerate(iteration_snapshots, 1):
                unions = iteration_unions[i-1] if i-1 < len(iteration_unions) else []
                comps = component_counts[i-1] if i-1 < len(component_counts) else None
                obj = {"iteration": i, "mst_size": len(snap), "components": comps, "unions": unions}
                f.write(json.dumps(obj) + "\n")
        
        print(f"[k-machine] Complete: {len(unique)} edges, weight {total_w:.6f}")
        print(f"[k-machine] Results: {out_dir}")
        
        return {
            'mst_edges': unique,
            'total_weight': total_w, 
            'iterations': iteration,
            'metrics': summary,
            'snapshots': iteration_snapshots,
            'unions': iteration_unions,
            'components': component_counts
        }
    
    return None