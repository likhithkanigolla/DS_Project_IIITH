#!/usr/bin/env python3
"""Main orchestrator: Generate graph, run both algorithms, create animations and comparisons."""
import argparse
import os
import subprocess
import sys
import time
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


def write_comparison(mpi_dir: str, kmachine_dir: str, comparison_file: str):
    """Write platform comparison file."""
    def read_metric(path, key):
        try:
            with open(path, 'r') as f:
                for line in f:
                    if line.startswith(f"{key}:"):
                        return line.split(':', 1)[1].strip()
        except Exception:
            return None
        return None
    
    mpi_metrics = os.path.join(mpi_dir, 'metrics.txt')
    km_metrics = os.path.join(kmachine_dir, 'metrics.txt')
    
    if os.path.exists(mpi_metrics) and os.path.exists(km_metrics):
        mpi_time = read_metric(mpi_metrics, 'total_time')
        km_time = read_metric(km_metrics, 'total_time')
        mpi_rounds = read_metric(mpi_metrics, 'rounds')
        km_rounds = read_metric(km_metrics, 'rounds')
        
        with open(comparison_file, 'w') as f:
            f.write("MST Algorithm Platform Comparison\n")
            f.write("=" * 40 + "\n")
            f.write(f"Plain MPI total_time: {mpi_time}\n")
            f.write(f"K-Machine total_time: {km_time}\n")
            f.write(f"Plain MPI rounds: {mpi_rounds}\n")
            f.write(f"K-Machine rounds: {km_rounds}\n")
            
            # Speed ratio
            try:
                if mpi_time and km_time:
                    ratio = float(km_time) / float(mpi_time)
                    f.write(f"Speed ratio (K-Machine/MPI): {ratio:.3f}\n")
            except ValueError:
                pass
        
        print(f"📊 Comparison written: {comparison_file}")


def main():
    p = argparse.ArgumentParser(description="Complete MST Analysis: Plain MPI vs K-Machine")
    p.add_argument("--nodes", type=int, default=40, help="Number of nodes")
    p.add_argument("--seed", type=int, default=42, help="Random seed")  
    p.add_argument("--ranks", type=int, default=4, help="MPI ranks")
    p.add_argument("--animate", action="store_true", help="Generate animations")
    args = p.parse_args()
    
    print("=" * 60)
    print("🌲 MST ANALYSIS: Plain MPI vs K-Machine Borůvka")
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
    
    mpi_dir = os.path.join(results_dir, 'plain_mpi')
    kmachine_dir = os.path.join(results_dir, 'kmachine')
    
    # Step 3: Run Plain MPI
    print("\n🖥️  Running Plain MPI...")
    run_command(f"mpiexec -n {args.ranks} python -c \"" +
                "from plain_mpi import run_plain_mpi; " +
                "from graph_utils import load_graph; " +
                f"num_nodes, edges = load_graph('{graph_file}'); " +
                f"run_plain_mpi(num_nodes, edges, '{mpi_dir}')\"")
    
    # Step 4: Run K-Machine  
    print("\n🌐 Running K-Machine...")
    run_command(f"mpiexec -n {args.ranks} python -c \"" +
                "from kmachine_boruvka import run_kmachine_boruvka; " +
                "from graph_utils import load_graph; " +
                f"num_nodes, edges = load_graph('{graph_file}'); " +
                f"run_kmachine_boruvka(num_nodes, edges, '{kmachine_dir}')\"")
    
    # Step 5: Generate comparison
    comparison_file = os.path.join(results_dir, 'platform_comparison.txt')
    write_comparison(mpi_dir, kmachine_dir, comparison_file)
    
    # Step 6: Generate animations (if requested)
    if args.animate:
        print("\n🎬 Generating animations...")
        try:
            # Load K-Machine iteration data for animation
            import json
            kmachine_log = os.path.join(kmachine_dir, 'iteration_log.jsonl')
            if os.path.exists(kmachine_log):
                snapshots = []
                with open(kmachine_log, 'r') as f:
                    for line in f:
                        json.loads(line.strip())  # validate format
                        # Reconstruct snapshot (simplified)
                        snapshots.append([])  # Would need actual edge data
                
                # Generate frames and GIF
                frames_dir = os.path.join(kmachine_dir, 'frames')
                save_animation(num_nodes, edges, snapshots, frames_dir, 
                             title='K-Machine Borůvka', size=args.ranks)
                
                gif_path = os.path.join(kmachine_dir, 'mst_animation.gif')
                frame_files = [os.path.join(frames_dir, f"frame_{i:03d}.png") 
                              for i in range(1, len(snapshots)+1)]
                build_gif(frame_files, gif_path, duration=2.0)
                print(f"🎥 Animation: {gif_path}")
        except Exception as e:
            print(f"⚠️  Animation failed: {e}")
    
    # Step 7: Summary
    print("\\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"📁 Results directory: {results_dir}")
    print("📊 Files created:")
    print(f"   • {mpi_dir}/mst_plain_mpi.txt")
    print(f"   • {kmachine_dir}/mst_kmachine.txt") 
    print(f"   • {comparison_file}")
    if args.animate:
        print(f"   • {kmachine_dir}/mst_animation.gif")
    print("=" * 60)


if __name__ == "__main__":
    main()