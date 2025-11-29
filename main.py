#!/usr/bin/env python3
"""Main orchestrator: Generate graph, run both algorithms, create animations and comparisons."""
import argparse
import os
import subprocess
import sys
import time
import networkx as nx
from datetime import datetime
from graph_utils import generate_graph
from visualization import save_animation, build_gif, save_growth_chart, save_iteration_tables


def save_graph_file(num_nodes: int, edges, filename: str):
    """Save graph to file."""
    with open(filename, 'w') as f:
        f.write(f"{num_nodes} {len(edges)}\n")
        for (u, v, w) in edges:
            f.write(f"{u} {v} {w}\n")


def run_command(cmd):
    """Execute shell command."""
    print(f"🚀 {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        sys.exit(result.returncode)
    return result.stdout


def write_comparison(mpi_dir: str, kmachine_dir: str, comparison_file: str, verification=None):
    """Write platform comparison file."""
def read_metric(file_path, metric_name):
    """Extract a specific metric value from metrics file."""
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip().startswith(f'{metric_name}:'):
                    value = line.split(':')[1].strip()
                    # Remove 'seconds' suffix if present
                    if value.endswith(' seconds'):
                        value = value.replace(' seconds', '')
                    return value
    except FileNotFoundError:
        return None
    return None

def verify_mst(graph_file, sequential_mst, distributed_mst):
    """Verify MST correctness using Kruskal's algorithm as ground truth."""
    from graph_utils import load_graph
    
    # Load graph
    num_nodes, edges = load_graph(graph_file)
    
    # Create NetworkX graph and compute optimal MST
    G = nx.Graph()
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
    
    optimal_mst = nx.minimum_spanning_tree(G, algorithm='kruskal')
    optimal_weight = sum(data['weight'] for u, v, data in optimal_mst.edges(data=True))
    
    # Read MST files
    def read_mst_edges(filename):
        edges = []
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('=') or line.startswith('-') or 'MST' in line or 'Edge List' in line or 'Total' in line or 'Number' in line or 'K-Machines' in line or 'Input Size' in line:
                    continue
                if '->' in line:  # Sequential format
                    parts = line.replace('->', ' ').replace(':', ' ').split()
                    if len(parts) >= 3:
                        u, v, w = int(parts[0]), int(parts[1]), float(parts[2])
                        edges.append((u, v, w))
                else:  # Distributed format
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            u, v, w = int(parts[0]), int(parts[1]), float(parts[2])
                            edges.append((u, v, w))
                        except ValueError:
                            continue
        return edges, sum(w for u, v, w in edges)
    
    seq_edges, seq_weight = read_mst_edges(sequential_mst)
    dist_edges, dist_weight = read_mst_edges(distributed_mst)
    
    # Verification results
    seq_valid = abs(seq_weight - optimal_weight) < 1e-5
    dist_valid = abs(dist_weight - optimal_weight) < 1e-5
    weights_match = abs(seq_weight - dist_weight) < 1e-5
    
    return {
        'optimal_weight': optimal_weight,
        'sequential_weight': seq_weight,
        'distributed_weight': dist_weight,
        'sequential_valid': seq_valid,
        'distributed_valid': dist_valid,
        'weights_match': weights_match,
        'num_edges': len(seq_edges)
    }

def write_comparison(mpi_dir: str, kmachine_dir: str, comparison_file: str, verification=None):
    mpi_metrics = os.path.join(mpi_dir, 'metrics.txt')
    km_metrics = os.path.join(kmachine_dir, 'metrics.txt')
    
    if os.path.exists(mpi_metrics) and os.path.exists(km_metrics):
        mpi_time = read_metric(mpi_metrics, 'total_time')
        km_time = read_metric(km_metrics, 'total_time')
        mpi_rounds = read_metric(mpi_metrics, 'comm_rounds')
        km_rounds = read_metric(km_metrics, 'comm_rounds')
        
        with open(comparison_file, 'w') as f:
            f.write("MST Algorithm Platform Comparison\n")
            f.write("=" * 50 + "\n\n")
            
            # MST Verification Results
            if verification:
                f.write("MST VERIFICATION\n")
                f.write("-" * 20 + "\n")
                f.write(f"Optimal MST Weight (Kruskal): {verification['optimal_weight']:.6f}\n")
                f.write(f"Sequential Borůvka Weight: {verification['sequential_weight']:.6f}\n")
                f.write(f"Distributed Borůvka Weight: {verification['distributed_weight']:.6f}\n")
                f.write(f"Sequential Algorithm: {'✅ OPTIMAL' if verification['sequential_valid'] else '❌ SUBOPTIMAL'}\n")
                f.write(f"Distributed Algorithm: {'✅ OPTIMAL' if verification['distributed_valid'] else '❌ SUBOPTIMAL'}\n")
                f.write(f"Weights Match: {'✅ YES' if verification['weights_match'] else '❌ NO'}\n")
                f.write(f"MST Edges: {verification['num_edges']}\n\n")
            
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Sequential Borůvka total_time: {float(mpi_time):.6f} seconds\n")
            f.write(f"Distributed Borůvka total_time: {float(km_time):.6f} seconds\n")
            f.write(f"Sequential Borůvka rounds: {mpi_rounds if mpi_rounds else 'N/A'}\n")
            f.write(f"Distributed Borůvka rounds: {km_rounds if km_rounds else 'N/A'}\n\n")
            
            # Speed ratio
            try:
                if mpi_time and km_time:
                    ratio = float(km_time) / float(mpi_time)
                    f.write("PERFORMANCE ANALYSIS\n")
                    f.write("-" * 20 + "\n")
                    f.write(f"Speed ratio (Distributed/Sequential): {ratio:.3f}x\n")
                    if ratio > 1:
                        f.write(f"Sequential is {ratio:.1f}x faster than Distributed\n")
                    else:
                        f.write(f"Distributed is {1/ratio:.1f}x faster than Sequential\n")
            except (ValueError, TypeError, ZeroDivisionError):
                pass
            except ValueError:
                pass
        
        print(f"📊 Comparison written: {comparison_file}")
        
        # Print verification summary
        if verification:
            print("\n" + "=" * 60)
            print("🔍 MST VERIFICATION SUMMARY")
            print("=" * 60)
            print(f"Sequential Algorithm: {'✅ OPTIMAL' if verification['sequential_valid'] else '❌ SUBOPTIMAL'}")
            print(f"Distributed Algorithm: {'✅ OPTIMAL' if verification['distributed_valid'] else '❌ SUBOPTIMAL'}")
            print(f"Algorithms Match: {'✅ YES' if verification['weights_match'] else '❌ NO'}")
            print(f"Optimal Weight: {verification['optimal_weight']:.6f}")


def main():
    p = argparse.ArgumentParser(description="Complete MST Analysis: Plain MPI vs K-Machine")
    p.add_argument("--nodes", type=int, default=40, help="Number of nodes")
    p.add_argument("--seed", type=int, default=42, help="Random seed")  
    p.add_argument("--ranks", type=int, default=4, help="MPI ranks")
    p.add_argument("--animate", action="store_true", help="Generate animations")
    args = p.parse_args()
    
    print("=" * 60)
    print("🌲 MST ANALYSIS: Sequential vs Distributed Borůvka")
    print("=" * 60)
    
    # Step 1: Generate graph
    print(f"🎲 Generating graph: {args.nodes} nodes, seed {args.seed}")
    num_nodes, edges = generate_graph(args.nodes, seed=args.seed)
    
    # Step 2: Create results directory
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    results_dir = os.path.join('results', timestamp)
    
    # Save graph file in results directory
    graph_file = os.path.join(results_dir, f"graph_n{args.nodes}_s{args.seed}.txt")
    os.makedirs(results_dir, exist_ok=True)
    save_graph_file(num_nodes, edges, graph_file)
    print(f"📁 Graph saved: {graph_file}")
    
    mpi_dir = os.path.join(results_dir, 'sequential')
    kmachine_dir = os.path.join(results_dir, 'distributed')
    
    # Step 3: Run Sequential Borůvka
    print("\n🖥️  Running Sequential Borůvka...")
    run_command(f"mpiexec -n {args.ranks} python -c \"" +
                "from sequential_boruvka import run_plain_mpi; " +
                "from graph_utils import load_graph; " +
                f"num_nodes, edges = load_graph('{graph_file}'); " +
                f"run_plain_mpi(num_nodes, edges, '{mpi_dir}')\"")
    
    # Step 4: Run Distributed Borůvka  
    print("\n🌐 Running Distributed Borůvka...")
    run_command(f"mpiexec -n {args.ranks} python -c \"" +
                "from distributed_boruvka import run_kmachine_boruvka; " +
                "from graph_utils import load_graph; " +
                f"num_nodes, edges = load_graph('{graph_file}'); " +
                f"run_kmachine_boruvka(num_nodes, edges, '{kmachine_dir}')\"")
    
    # Step 5: Verify MST correctness
    print("\n🔍 Verifying MST correctness...")
    verification = verify_mst(
        graph_file, 
        os.path.join(mpi_dir, 'mst_sequential.txt'),
        os.path.join(kmachine_dir, 'mst_distributed.txt')
    )
    
    # Step 6: Generate comparison
    comparison_file = os.path.join(results_dir, 'platform_comparison.txt')
    write_comparison(mpi_dir, kmachine_dir, comparison_file, verification)
    
    # Step 7: Generate animations and charts (if requested)
    if args.animate:
        print("\n🎬 Generating visualizations...")
        try:
            import json
            kmachine_log = os.path.join(kmachine_dir, 'iteration_log.jsonl')
            
            if os.path.exists(kmachine_log):
                snapshots = []
                component_counts = []  # NEW: Track convergence
                mst_sizes = []        # NEW: Track MST growth
                
                with open(kmachine_log, 'r') as f:
                    for line in f:
                        if line.strip():
                            try:
                                data = json.loads(line.strip())
                                snapshots.append(data.get('unions', []))
                                if 'components' in data:
                                    component_counts.append(data['components'])
                                if 'mst_size' in data:
                                    mst_sizes.append(data['mst_size'])
                            except json.JSONDecodeError as e:
                                continue  # Skip malformed lines
            else:
                snapshots = []
                component_counts = []
                mst_sizes = []

            # 1. Generate Growth/Convergence Chart (NEW)
            print("   📊 Generating convergence chart...")
            if component_counts and mst_sizes:
                chart_path = save_growth_chart(component_counts, mst_sizes, kmachine_dir)
                if chart_path:
                    print(f"   ✅ Convergence chart saved: {chart_path}")
            else:
                print("   ⚠️  No convergence data available")

            # 2. Generate Animation (Existing)
            print("   🎥 Generating MST animation...")
            if snapshots:
                # Convert union data to weighted edge format for animation
                edge_weights = {(min(u, v), max(u, v)): w for u, v, w in edges}
                weighted_snapshots = []
                for unions in snapshots:
                    weighted_edges = []
                    for u, v in unions:
                        key = (min(u, v), max(u, v))
                        if key in edge_weights:
                            weighted_edges.append((u, v, edge_weights[key]))
                    weighted_snapshots.append(weighted_edges)
                
                frames_dir = os.path.join(kmachine_dir, 'frames')
                save_animation(num_nodes, edges, weighted_snapshots, frames_dir, 
                             title='K-Machine Borůvka', size=args.ranks)
                
                gif_path = os.path.join(kmachine_dir, 'mst_animation.gif')
                frame_files = [os.path.join(frames_dir, f"frame_{i:03d}.png") 
                              for i in range(1, len(snapshots)+1)]
                actual_frames = [f for f in frame_files if os.path.exists(f)]
                if actual_frames:
                    build_gif(actual_frames, gif_path, duration=2.0)
                    print(f"   ✅ Animation saved: {gif_path}")
                else:
                    print("   ⚠️  No animation frames found")
            else:
                print("   ⚠️  No snapshot data available")
        except Exception as e:
            print(f"⚠️  Animation failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Step 7: Summary
    print("\\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"📁 Results directory: {results_dir}")
    print("📊 Files created:")
    print(f"   • {mpi_dir}/mst_sequential.txt")
    print(f"   • {kmachine_dir}/mst_distributed.txt") 
    print(f"   • {comparison_file}")
    if args.animate:
        print(f"   • {kmachine_dir}/mst_animation.gif")
    print("=" * 60)


if __name__ == "__main__":
    main()