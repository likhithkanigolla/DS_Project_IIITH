#!/usr/bin/env python3
"""Sample and convert SNAP edge-lists into weighted repo-compatible graphs.

Usage examples:
  python sample_snap_subset.py snap.txt graphs/out.txt --method node --target-nodes 2000 --weight hash --seed 42 --keep-largest-cc
  python sample_snap_subset.py snap.txt graphs/out.txt --method edge --target-edges 10000 --weight random --seed 123

Supported sampling methods: node, edge, snowball
Supported weight methods: random, hash, degree
"""
import sys
import random
import hashlib
import argparse
from collections import defaultdict, deque


def read_snap_edges(path):
    edges = []
    nodes = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                a = int(parts[0]); b = int(parts[1])
            except:
                continue
            if a == b:
                continue
            edges.append((a, b))
            nodes.add(a); nodes.add(b)
    return edges, nodes


def node_induced_sample(edges, nodes, target_n, seed):
    random.seed(seed)
    node_list = list(nodes)
    if target_n >= len(node_list):
        chosen = set(node_list)
    else:
        chosen = set(random.sample(node_list, target_n))
    sampled_edges = [(u, v) for (u, v) in edges if u in chosen and v in chosen]
    return sampled_edges, chosen


def edge_sample(edges, target_m, seed):
    random.seed(seed)
    if target_m >= len(edges):
        sampled = list(edges)
    else:
        sampled = random.sample(edges, target_m)
    chosen = set()
    for u, v in sampled:
        chosen.add(u); chosen.add(v)
    return sampled, chosen


def snowball_sample(edges, nodes, target_n, seed):
    # build adjacency (undirected)
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    random.seed(seed)
    if not nodes:
        return [], set()
    start = random.choice(list(nodes))
    q = deque([start])
    chosen = {start}
    while q and len(chosen) < target_n:
        x = q.popleft()
        for nbr in adj.get(x, []):
            if nbr not in chosen:
                chosen.add(nbr)
                q.append(nbr)
                if len(chosen) >= target_n:
                    break
    sampled_edges = [(u, v) for (u, v) in edges if u in chosen and v in chosen]
    return sampled_edges, chosen


def keep_largest_cc(sampled_edges):
    # build adjacency
    adj = defaultdict(list)
    for u, v in sampled_edges:
        adj[u].append(v); adj[v].append(u)
    visited = set()
    best_comp = []
    for n in adj:
        if n in visited:
            continue
        comp = []
        stack = [n]
        visited.add(n)
        while stack:
            x = stack.pop()
            comp.append(x)
            for nbr in adj.get(x, []):
                if nbr not in visited:
                    visited.add(nbr); stack.append(nbr)
        if len(comp) > len(best_comp):
            best_comp = comp
    chosen = set(best_comp)
    new_edges = [(u, v) for (u, v) in sampled_edges if u in chosen and v in chosen]
    return new_edges, chosen


def assign_weights(sampled_edges, weight_method, seed, chosen_nodes):
    if weight_method == 'random':
        random.seed(seed)
        return [(u, v, round(random.uniform(1.0, 100.0), 6)) for (u, v) in sampled_edges]
    elif weight_method == 'hash':
        out = []
        for u, v in sampled_edges:
            s = f"{u}-{v}"
            h = hashlib.md5(s.encode('utf-8')).hexdigest()
            # map hex to float in [1,100]
            val = int(h[:8], 16) / 0xFFFFFFFF
            w = 1.0 + val * 99.0
            out.append((u, v, round(w, 6)))
        return out
    elif weight_method == 'degree':
        # compute degrees in induced graph
        deg = defaultdict(int)
        for u, v in sampled_edges:
            deg[u] += 1; deg[v] += 1
        out = []
        for u, v in sampled_edges:
            w = 1.0 / (deg[u] + deg[v] + 1e-6)  # small eps
            # scale to [1,100]
            w_scaled = 1.0 + (1.0 / (w + 1e-12)) % 100
            out.append((u, v, round(w_scaled, 6)))
        return out
    else:
        raise ValueError("Unknown weight method")


def remap_and_write(weighted_edges, out_path):
    id_map = {}
    next_id = 0
    remapped = []
    for u, v, w in weighted_edges:
        if u not in id_map:
            id_map[u] = next_id; next_id += 1
        if v not in id_map:
            id_map[v] = next_id; next_id += 1
        remapped.append((id_map[u], id_map[v], w))
    n = next_id; m = len(remapped)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"{n} {m}\n")
        for u, v, w in remapped:
            f.write(f"{u} {v} {w}\n")
    return n, m


def main():
    p = argparse.ArgumentParser(description='Sample and convert SNAP edge-list to weighted repo graph format')
    p.add_argument('input', help='SNAP edge list path')
    p.add_argument('output', help='Output graph file path')
    p.add_argument('--method', choices=['node','edge','snowball'], default='node')
    p.add_argument('--target-nodes', type=int, default=5000)
    p.add_argument('--target-edges', type=int, default=10000)
    p.add_argument('--weight', choices=['random','hash','degree'], default='random')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--keep-largest-cc', action='store_true')
    args = p.parse_args()

    edges, nodes = read_snap_edges(args.input)
    print(f"Read {len(edges)} edges and {len(nodes)} unique node ids from {args.input}")

    if args.method == 'node':
        sampled_edges, chosen = node_induced_sample(edges, nodes, args.target_nodes, args.seed)
    elif args.method == 'edge':
        sampled_edges, chosen = edge_sample(edges, args.target_edges, args.seed)
    else:
        sampled_edges, chosen = snowball_sample(edges, nodes, args.target_nodes, args.seed)

    print(f"Sampled: {len(chosen)} nodes, {len(sampled_edges)} edges (before LCC)")

    if args.keep_largest_cc:
        sampled_edges, chosen = keep_largest_cc(sampled_edges)
        print(f"After keeping LCC: {len(chosen)} nodes, {len(sampled_edges)} edges")

    weighted = assign_weights(sampled_edges, args.weight, args.seed, chosen)
    n, m = remap_and_write(weighted, args.output)
    print(f"Wrote {args.output} with n={n}, m={m}")


if __name__ == '__main__':
    main()
