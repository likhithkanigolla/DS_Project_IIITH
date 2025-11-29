#!/usr/bin/env python3
"""Validate MSTs: compute optimal MST with Kruskal and compare with outputs."""
import sys
import os
from typing import List, Tuple
from dsu import DSU

def load_graph(path: str) -> Tuple[int, List[Tuple[int,int,float]]]:
    edges = []
    max_node = -1
    with open(path, 'r') as fh:
        first = fh.readline().strip().split()
        if len(first) >= 2 and all(x.isdigit() for x in first[:2]):
            # header present
            try:
                n = int(first[0])
            except:
                n = None
        else:
            # no header, treat first line as edge
            fh.seek(0)
            n = None
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            u = int(parts[0]); v = int(parts[1]); w = float(parts[2])
            edges.append((u, v, w))
            max_node = max(max_node, u, v)
    if n is None:
        n = max_node + 1
    return n, edges

def kruskal_mst(num_nodes: int, edges: List[Tuple[int,int,float]]):
    ds = DSU()
    for i in range(num_nodes):
        ds.make_set(i)
    sorted_edges = sorted(edges, key=lambda x: x[2])
    mst = []
    for u,v,w in sorted_edges:
        if ds.find(u) != ds.find(v):
            ds.union(u, v)
            mst.append((u,v,w))
            if len(mst) >= num_nodes-1:
                break
    total = sum(w for (_,_,w) in mst)
    return mst, total

def read_mst_file(filename):
    """Read MST from output file, skipping header lines."""
    edges = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('=') or line.startswith('-') or 'MST' in line or 'Edge List' in line or 'Total' in line or 'Number' in line or 'K-Machines' in line or 'Input Size' in line:
                continue
            if '->' in line:  # Sequential format: "u -> v: w"
                parts = line.replace('->', ' ').replace(':', ' ').split()
                if len(parts) >= 3:
                    u, v, w = int(parts[0]), int(parts[1]), float(parts[2])
                    edges.append((u, v, w))
            else:  # Distributed format: "u v w"
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        u, v, w = int(parts[0]), int(parts[1]), float(parts[2])
                        edges.append((u, v, w))
                    except ValueError:
                        continue  # Skip non-edge lines
    
    total_weight = sum(w for u, v, w in edges)
    return edges, total_weight

def main():
    if len(sys.argv) < 4:
        print("Usage: validate_mst.py <graph_file> <plain_mpi_mst> <kmachine_mst>")
        sys.exit(1)
    graph_file = sys.argv[1]
    plain_file = sys.argv[2]
    km_file = sys.argv[3]
    num_nodes, edges = load_graph(graph_file)
    opt_mst, opt_weight = kruskal_mst(num_nodes, edges)
    p_edges, p_weight = read_mst_file(plain_file)
    k_edges, k_weight = read_mst_file(km_file)

    print(f"Graph: {graph_file} nodes={num_nodes} edges={len(edges)}")
    print(f"Optimal (Kruskal): weight={opt_weight:.6f} edges={len(opt_mst)}")
    print(f"Plain MPI: weight={p_weight:.6f} edges={len(p_edges)}")
    print(f"K-Machine: weight={k_weight:.6f} edges={len(k_edges)}")

    def close_enough(a,b):
        return abs(a-b) < 1e-6

    if close_enough(opt_weight, p_weight):
        print("Plain MPI matches optimal MST.")
    else:
        print("Plain MPI DOES NOT match optimal MST.")
    if close_enough(opt_weight, k_weight):
        print("K-Machine matches optimal MST.")
    else:
        print("K-Machine DOES NOT match optimal MST.")

if __name__ == '__main__':
    main()
