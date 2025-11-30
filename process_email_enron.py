#!/usr/bin/env python3
"""Process Email-Enron.txt: aggregate message counts into weights, sample subgraph, and write repo format.

Usage:
  python process_email_enron.py --input Email-Enron.txt --output graphs/email_enron_2k_count.txt --target-nodes 2000 --seed 42 --keep-largest-cc
"""
import argparse
import os
from collections import defaultdict
import random
import hashlib


def aggregate_counts(path):
    counts = defaultdict(int)
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
            u, v = (a, b) if a <= b else (b, a)
            counts[(u, v)] += 1
            nodes.add(a); nodes.add(b)
    edges = [(u, v, float(w)) for (u, v), w in counts.items()]
    return edges, nodes


def node_induced_sample_edges(edges, nodes, target_n, seed):
    random.seed(seed)
    node_list = list(nodes)
    if target_n >= len(node_list):
        chosen = set(node_list)
    else:
        chosen = set(random.sample(node_list, target_n))
    sampled = [(u, v, w) for (u, v, w) in edges if u in chosen and v in chosen]
    return sampled, chosen


def keep_largest_cc(sampled_edges):
    # build adjacency
    adj = defaultdict(list)
    for u, v, w in sampled_edges:
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
    new_edges = [(u, v, w) for (u, v, w) in sampled_edges if u in chosen and v in chosen]
    return new_edges, chosen


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
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"{n} {m}\n")
        for u, v, w in remapped:
            f.write(f"{u} {v} {w}\n")
    return n, m


def add_deterministic_jitter(edges):
    """Add a small deterministic fractional jitter to weights so identical counts are distinguishable.

    Jitter is derived from MD5(u-v) and is in range (0, 0.9). This keeps weights reproducible.
    """
    out = []
    for u, v, w in edges:
        s = f"{u}-{v}"
        h = hashlib.md5(s.encode('utf-8')).hexdigest()
        val = int(h[:8], 16) / 0xFFFFFFFF
        jitter = val * 0.9
        out.append((u, v, float(w) + jitter))
    return out


def map_weights_to_range(edges, min_w=1.0, max_w=100.0):
    """Deterministically map each edge to a weight in [min_w, max_w].

    Uses MD5(u-v) to produce a reproducible pseudo-random value in [0,1),
    then scales to the requested range. Returns new edge list with mapped
    float weights.
    """
    out = []
    # Use 16 hex chars (64 bits) for better granularity
    for u, v, w in edges:
        s = f"{u}-{v}"
        h = hashlib.md5(s.encode('utf-8')).hexdigest()
        val = int(h[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
        mapped = float(min_w) + val * (float(max_w) - float(min_w))
        out.append((u, v, mapped))
    return out


def main():
    p = argparse.ArgumentParser(description='Process Email-Enron: aggregate counts, sample, and write weighted graph')
    p.add_argument('--input', default='Email-Enron.txt')
    p.add_argument('--output', default='graphs/email_enron_2k_count.txt')
    p.add_argument('--target-nodes', type=int, default=2000)
    p.add_argument('--method', choices=['node','snowball','edge'], default='node')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--keep-largest-cc', action='store_true')
    p.add_argument('--min-weight', type=float, default=1.0, help='Minimum mapped weight (default: 1.0)')
    p.add_argument('--max-weight', type=float, default=100.0, help='Maximum mapped weight (default: 100.0)')
    args = p.parse_args()

    print(f"Reading and aggregating counts from {args.input} ...")
    edges, nodes = aggregate_counts(args.input)
    print(f"Aggregated {len(edges)} unique undirected pairs from {len(nodes)} nodes")

    print(f"Sampling ({args.method}) subgraph with target {args.target_nodes} nodes (seed={args.seed})...")
    if args.method == 'node':
        sampled_edges, chosen = node_induced_sample_edges(edges, nodes, args.target_nodes, args.seed)
    elif args.method == 'snowball':
        # build adjacency (undirected) from aggregated edges
        adj = defaultdict(set)
        for u, v, w in edges:
            adj[u].add(v); adj[v].add(u)
        # snowball seed selection
        random.seed(args.seed)
        start = random.choice(list(nodes))
        from collections import deque
        q = deque([start])
        chosen = {start}
        while q and len(chosen) < args.target_nodes:
            x = q.popleft()
            for nbr in adj.get(x, []):
                if nbr not in chosen:
                    chosen.add(nbr)
                    q.append(nbr)
                    if len(chosen) >= args.target_nodes:
                        break
        sampled_edges = [(u, v, w) for (u, v, w) in edges if u in chosen and v in chosen]
    else:
        # edge sampling: sample by number of edges approximate target via random sampling
        # target-nodes parameter will be ignored; sample up to target_nodes edges
        random.seed(args.seed)
        M = args.target_nodes
        if M >= len(edges):
            sampled_edges = list(edges)
        else:
            sampled_edges = list(edges)
            sampled_edges = random.sample(sampled_edges, M)
        chosen = set()
        for u,v,w in sampled_edges:
            chosen.add(u); chosen.add(v)
    print(f"Sampled {len(chosen)} nodes and {len(sampled_edges)} edges before LCC")

    if args.keep_largest_cc:
        sampled_edges, chosen2 = keep_largest_cc(sampled_edges)
        print(f"After keeping LCC: {len(chosen2)} nodes, {len(sampled_edges)} edges")
    # Map weights deterministically into the desired range (Option B)
    sampled_edges = map_weights_to_range(sampled_edges, args.min_weight, args.max_weight)

    print(f"Writing weighted graph to {args.output} ...")
    n, m = remap_and_write(sampled_edges, args.output)
    print(f"Wrote {args.output} with n={n}, m={m}")


if __name__ == '__main__':
    # avoid import issues for defaultdict in keep_largest_cc
    from collections import defaultdict
    main()
