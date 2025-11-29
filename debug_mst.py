#!/usr/bin/env python3
"""Debug MST differences by implementing reference algorithms and tracing execution."""

import sys
from typing import List, Tuple
from graph_utils import load_graph
from dsu import DSU


def kruskal_reference(num_nodes: int, edges: List[Tuple[int, int, float]]) -> Tuple[List[Tuple[int, int, float]], float]:
    """Reference Kruskal's algorithm implementation."""
    # Sort edges by weight
    sorted_edges = sorted(edges, key=lambda x: x[2])
    
    dsu = DSU()
    for i in range(num_nodes):
        dsu.make_set(i)
    
    mst_edges = []
    total_weight = 0.0
    
    print("=== KRUSKAL'S ALGORITHM (REFERENCE) ===")
    for i, (u, v, w) in enumerate(sorted_edges):
        ru = dsu.find(u)
        rv = dsu.find(v)
        if ru != rv:
            dsu.union(ru, rv)
            mst_edges.append((u, v, w))
            total_weight += w
            print(f"Edge {len(mst_edges)}: ({u}, {v}) = {w:.6f}, total = {total_weight:.6f}")
            
            if len(mst_edges) == num_nodes - 1:
                break
    
    return mst_edges, total_weight


def boruvka_reference(num_nodes: int, edges: List[Tuple[int, int, float]]) -> Tuple[List[Tuple[int, int, float]], float]:
    """Reference Borůvka's algorithm implementation with detailed tracing."""
    dsu = DSU()
    for i in range(num_nodes):
        dsu.make_set(i)
    
    mst_edges = []
    total_weight = 0.0
    iteration = 0
    
    print("=== BORŮVKA'S ALGORITHM (REFERENCE) ===")
    
    while dsu.num_components() > 1 and len(mst_edges) < num_nodes - 1:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")
        print(f"Current components: {dsu.num_components()}")
        
        # Find minimum outgoing edge per component
        component_min_edge = {}
        
        for u, v, w in edges:
            ru = dsu.find(u)
            rv = dsu.find(v)
            
            if ru == rv:
                continue  # Same component
            
            # Check if this is minimum for component ru
            if ru not in component_min_edge or w < component_min_edge[ru][2]:
                component_min_edge[ru] = (u, v, w)
            
            # Check if this is minimum for component rv
            if rv not in component_min_edge or w < component_min_edge[rv][2]:
                component_min_edge[rv] = (v, u, w)
        
        # Print candidates
        print("Component minimum edges:")
        for comp, (u, v, w) in component_min_edge.items():
            print(f"  Component {comp}: ({u}, {v}) = {w:.6f}")
        
        # Add all selected edges (avoiding duplicates)
        selected_edges = set()
        union_pairs = []
        
        for u, v, w in component_min_edge.values():
            edge_key = (min(u, v), max(u, v))
            if edge_key in selected_edges:
                continue
            selected_edges.add(edge_key)
            
            ru = dsu.find(u)
            rv = dsu.find(v)
            if ru != rv:
                dsu.union(ru, rv)
                mst_edges.append((u, v, w))
                total_weight += w
                union_pairs.append((ru, rv))
                print(f"  Added: ({u}, {v}) = {w:.6f}, unions: {ru}-{rv}")
        
        print(f"Edges added this iteration: {len(union_pairs)}")
        print(f"Total MST edges: {len(mst_edges)}")
    
    return mst_edges, total_weight


def trace_plain_mpi_logic(num_nodes: int, edges: List[Tuple[int, int, float]]):
    """Trace the exact logic used in plain_mpi.py"""
    print("\n=== TRACING PLAIN MPI LOGIC ===")
    
    dsu = DSU()
    for i in range(num_nodes):
        dsu.make_set(i)
    
    mst_edges = []
    iteration = 0
    
    while dsu.num_components() > 1 and len(mst_edges) < num_nodes - 1:
        iteration += 1
        print(f"\n--- Plain MPI Iteration {iteration} ---")
        
        # This mirrors the exact logic in plain_mpi.py
        local_best = {}
        for (u, v, w) in edges:
            ru = dsu.find(u)
            rv = dsu.find(v)
            if ru == rv:
                continue
            cur = local_best.get(ru)
            if cur is None or w < cur[2]:
                local_best[ru] = (u, v, w)
        
        print("Local best edges per component:")
        for comp, (u, v, w) in local_best.items():
            print(f"  Component {comp}: ({u}, {v}) = {w:.6f}")
        
        union_pairs = []
        for (u, v, w) in local_best.values():
            ru = dsu.find(u)
            rv = dsu.find(v)
            if ru != rv:
                dsu.union(ru, rv)
                mst_edges.append((u, v, w))
                union_pairs.append((ru, rv))
                print(f"  Selected: ({u}, {v}) = {w:.6f}")
        
        print(f"Components after iteration: {dsu.num_components()}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python debug_mst.py <graph_file>")
        sys.exit(1)
    
    graph_file = sys.argv[1]
    num_nodes, edges = load_graph(graph_file)
    
    print(f"Graph: {num_nodes} nodes, {len(edges)} edges")
    print("\nEdges sorted by weight:")
    sorted_edges = sorted(edges, key=lambda x: x[2])
    for i, (u, v, w) in enumerate(sorted_edges):
        print(f"  {i+1:2d}. ({u}, {v}) = {w:.6f}")
    
    print("\n" + "="*60)
    
    # Run reference Kruskal's algorithm
    kruskal_mst, kruskal_weight = kruskal_reference(num_nodes, edges)
    
    print("\n" + "="*60)
    
    # Run reference Borůvka's algorithm  
    boruvka_mst, boruvka_weight = boruvka_reference(num_nodes, edges)
    
    print("\n" + "="*60)
    
    # Trace Plain MPI logic
    trace_plain_mpi_logic(num_nodes, edges)
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print(f"Kruskal MST weight:  {kruskal_weight:.6f}")
    print(f"Borůvka MST weight:  {boruvka_weight:.6f}")
    print(f"Weights match: {abs(kruskal_weight - boruvka_weight) < 1e-9}")
    
    if abs(kruskal_weight - boruvka_weight) > 1e-9:
        print("ERROR: Reference algorithms produce different weights!")
    else:
        print("✓ Reference algorithms agree on optimal weight")


if __name__ == "__main__":
    main()