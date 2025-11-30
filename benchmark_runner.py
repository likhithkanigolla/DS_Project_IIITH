#!/usr/bin/env python3
"""
Benchmark Runner for MST Algorithm Performance Analysis

Generates comprehensive performance visualizations:
1. Performance Comparison (Time vs Input Size)
2. Scalability Analysis (Speedup vs Number of Machines)  
3. Communication Complexity (Rounds vs Input Size)
"""

import subprocess
import os
import re
import json
import time
import matplotlib.pyplot as plt

# Configuration
NODES_TESTS = [50, 100, 200, 500, 1000]  # Sizes to test for Time vs N
RANKS_TESTS = [2, 4, 6, 8]              # Ranks to test for Speedup vs K
FIXED_N_FOR_SPEEDUP = 1000              # Graph size for Speedup test
FIXED_K_FOR_TIME = 4                    # Ranks for Time test
RESULTS_FILE = "benchmark_results.json"
SEED = 42                               # Fixed seed for reproducibility

def run_experiment(nodes, ranks):
    """Run a single MST experiment and extract metrics."""
    print(f"  Running N={nodes}, K={ranks}...")
    
    # Run main.py (disable animation to save time)
    cmd = f"python main.py --nodes {nodes} --ranks {ranks} --seed {SEED}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"    FAILED: {result.stderr}")
        return None
    
    # Parse the output directory from stdout - use simpler pattern
    match = re.search(r"Results directory: (results/[0-9-]+)", result.stdout)
    if not match:
        print("    FAILED to find results directory")
        print(f"    Last 200 chars: {result.stdout[-200:]}")
        return None
        
    res_dir = match.group(1)
    return extract_metrics(res_dir)

def extract_metrics(res_dir):
    """Extract metrics from result directory."""
    metrics = {'seq_time': 0, 'dist_time': 0, 'rounds': 0, 'seq_rounds': 0}
    
    # Read Sequential Time
    seq_metrics_file = os.path.join(res_dir, 'sequential/metrics.txt')
    if os.path.exists(seq_metrics_file):
        try:
            with open(seq_metrics_file) as f:
                for line in f:
                    if 'total_time' in line:
                        time_str = line.split(':')[1].strip()
                        if time_str.endswith(' seconds'):
                            time_str = time_str.replace(' seconds', '')
                        metrics['seq_time'] = float(time_str)
                    elif 'comm_rounds' in line:
                        metrics['seq_rounds'] = int(line.split(':')[1].strip())
        except Exception as e:
            print(f"    Warning: Could not read sequential metrics: {e}")

    # Read Distributed Time & Rounds
    dist_metrics_file = os.path.join(res_dir, 'distributed/metrics.txt')
    if os.path.exists(dist_metrics_file):
        try:
            with open(dist_metrics_file) as f:
                for line in f:
                    if 'total_time' in line:
                        time_str = line.split(':')[1].strip()
                        if time_str.endswith(' seconds'):
                            time_str = time_str.replace(' seconds', '')
                        metrics['dist_time'] = float(time_str)
                    elif 'comm_rounds' in line:
                        metrics['rounds'] = int(line.split(':')[1].strip())
        except Exception as e:
            print(f"    Warning: Could not read distributed metrics: {e}")
    
    print(f"    SUCCESS: Sequential: {metrics['seq_time']:.6f}s, Distributed: {metrics['dist_time']:.6f}s, Rounds: {metrics['rounds']}")
    return metrics

def plot_results(data, output_dir):
    """Generate all three benchmark plots in the specified output directory."""
    plt.style.use('default')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Performance Comparison (Time vs N)
    print("Generating performance comparison plot...")
    ns = [d['n'] for d in data['time_vs_n']]
    seq_times = [d['metrics']['seq_time'] for d in data['time_vs_n']]
    dist_times = [d['metrics']['dist_time'] for d in data['time_vs_n']]
    
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(ns, seq_times, 'o-', label='Sequential (Rank 0)', color='red', linewidth=2, markersize=8)
    plt.plot(ns, dist_times, 's-', label=f'Distributed (K={FIXED_K_FOR_TIME})', color='blue', linewidth=2, markersize=8)
    plt.xlabel('Number of Nodes (N)')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Performance Comparison: Sequential vs Distributed Borůvka')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale for better visibility
    
    # 2. Communication Overhead (Rounds vs N)
    print("Generating communication complexity plot...")
    rounds = [d['metrics']['rounds'] for d in data['time_vs_n']]
    
    plt.subplot(2, 2, 2)
    plt.plot(ns, rounds, 'o--', color='green', linewidth=2, markersize=8)
    plt.xlabel('Number of Nodes (N)')
    plt.ylabel('Communication Rounds')
    plt.title('Communication Complexity vs Input Size')
    plt.grid(True, alpha=0.3)
    
    # Add theoretical O(log N) reference line
    import numpy as np
    log_n = [np.log2(n) for n in ns]
    # Scale to match data roughly
    if rounds:
        scale_factor = max(rounds) / max(log_n)
        scaled_log_n = [scale_factor * ln for ln in log_n]
        plt.plot(ns, scaled_log_n, 'k--', alpha=0.5, label='O(log N) reference')
        plt.legend()

    # 3. Scalability (Speedup vs K)
    print("Generating scalability analysis plot...")
    ks = [d['k'] for d in data['speedup_vs_k']]
    # Speedup = Seq_Time / Dist_Time
    # Use sequential time from same experiment size
    speedups = []
    for d in data['speedup_vs_k']:
        if d['metrics']['seq_time'] > 0 and d['metrics']['dist_time'] > 0:
            speedup = d['metrics']['seq_time'] / d['metrics']['dist_time']
            speedups.append(speedup)
        else:
            speedups.append(0)
    
    plt.subplot(2, 2, 3)
    plt.plot(ks, speedups, 'o-', color='purple', linewidth=2, markersize=8)
    plt.plot(ks, [1]*len(ks), 'k--', label='No Speedup (1.0x)', alpha=0.5) # Reference line
    plt.xlabel('Number of Machines (K)')
    plt.ylabel('Speedup Factor (Sequential/Distributed)')
    plt.title(f'Scalability Analysis (N={FIXED_N_FOR_SPEEDUP})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Efficiency vs Ranks
    print("Generating efficiency analysis plot...")
    efficiencies = [speedup/k for speedup, k in zip(speedups, ks)]
    
    plt.subplot(2, 2, 4)
    plt.plot(ks, efficiencies, 'o-', color='orange', linewidth=2, markersize=8)
    plt.plot(ks, [1]*len(ks), 'k--', label='Perfect Efficiency (1.0)', alpha=0.5)
    plt.xlabel('Number of Machines (K)')
    plt.ylabel('Parallel Efficiency (Speedup/K)')
    plt.title('Parallel Efficiency vs Number of Machines')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    combined_plot = os.path.join(output_dir, 'benchmark_analysis.png')
    plt.savefig(combined_plot, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate individual high-resolution plots
    save_individual_plots(data, output_dir)
    
    return combined_plot

def save_individual_plots(data, output_dir):
    """Save individual plots for detailed analysis in the output directory."""
    
    # Performance plot
    ns = [d['n'] for d in data['time_vs_n']]
    seq_times = [d['metrics']['seq_time'] for d in data['time_vs_n']]
    dist_times = [d['metrics']['dist_time'] for d in data['time_vs_n']]
    
    plt.figure(figsize=(10, 6))
    plt.plot(ns, seq_times, 'o-', label='Sequential Borůvka', color='red', linewidth=2, markersize=8)
    plt.plot(ns, dist_times, 's-', label=f'Distributed Borůvka (K={FIXED_K_FOR_TIME})', color='blue', linewidth=2, markersize=8)
    plt.xlabel('Number of Nodes (N)')
    plt.ylabel('Execution Time (seconds)')
    plt.title('MST Algorithm Performance Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    performance_plot = os.path.join(output_dir, 'plot_performance.png')
    plt.savefig(performance_plot, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Communication plot
    rounds = [d['metrics']['rounds'] for d in data['time_vs_n']]
    plt.figure(figsize=(10, 6))
    plt.plot(ns, rounds, 'o-', color='green', linewidth=2, markersize=8)
    plt.xlabel('Number of Nodes (N)')
    plt.ylabel('Communication Rounds')
    plt.title('Communication Complexity of Distributed Borůvka Algorithm')
    plt.grid(True, alpha=0.3)
    rounds_plot = os.path.join(output_dir, 'plot_rounds.png')
    plt.savefig(rounds_plot, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Speedup plot
    ks = [d['k'] for d in data['speedup_vs_k']]
    speedups = [d['metrics']['seq_time'] / d['metrics']['dist_time'] 
                for d in data['speedup_vs_k'] 
                if d['metrics']['seq_time'] > 0 and d['metrics']['dist_time'] > 0]
    
    plt.figure(figsize=(10, 6))
    plt.plot(ks, speedups, 'o-', color='purple', linewidth=2, markersize=8)
    plt.plot(ks, [1]*len(ks), 'k--', label='No Speedup (1.0x)', alpha=0.5)
    plt.xlabel('Number of Machines (K)')
    plt.ylabel('Speedup Factor')
    plt.title(f'Scalability Analysis: Speedup vs Number of Machines (N={FIXED_N_FOR_SPEEDUP})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    speedup_plot = os.path.join(output_dir, 'plot_speedup.png')
    plt.savefig(speedup_plot, dpi=300, bbox_inches='tight')
    plt.close()
    
    return performance_plot, rounds_plot, speedup_plot

def print_summary(data):
    """Print a summary of benchmark results."""
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    # Performance Analysis
    print("\nPERFORMANCE ANALYSIS:")
    for d in data['time_vs_n']:
        n = d['n']
        seq_t = d['metrics']['seq_time']
        dist_t = d['metrics']['dist_time']
        overhead = dist_t / seq_t if seq_t > 0 else 0
        print(f"  N={n:4d}: Sequential={seq_t:.6f}s, Distributed={dist_t:.6f}s, Overhead={overhead:.1f}x")
    
    # Scalability Analysis  
    print("\nSCALABILITY ANALYSIS:")
    for d in data['speedup_vs_k']:
        k = d['k']
        seq_t = d['metrics']['seq_time'] 
        dist_t = d['metrics']['dist_time']
        speedup = seq_t / dist_t if dist_t > 0 else 0
        efficiency = speedup / k if k > 0 else 0
        print(f"  K={k}: Speedup={speedup:.2f}x, Efficiency={efficiency:.2f}")
    
    # Communication Analysis
    print("\nCOMMUNICATION ANALYSIS:")
    for d in data['time_vs_n']:
        n = d['n']
        rounds = d['metrics']['rounds']
        theoretical = int(n.bit_length()) if n > 0 else 0  # Rough log2(n)
        print(f"  N={n:4d}: Rounds={rounds}, ~O(log N)≈{theoretical}")

def main():
    """Run comprehensive benchmark suite."""
    # Create timestamped results directory
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    benchmark_dir = os.path.join('results', f'benchmark_{timestamp}')
    os.makedirs(benchmark_dir, exist_ok=True)
    
    print("="*60)
    print("MST ALGORITHM BENCHMARK SUITE")
    print("="*60)
    print("Test Configuration:")
    print(f"   • Input Sizes (N): {NODES_TESTS}")
    print(f"   • Machine Counts (K): {RANKS_TESTS}")
    print(f"   • Fixed K for Time Analysis: {FIXED_K_FOR_TIME}")
    print(f"   • Fixed N for Speedup Analysis: {FIXED_N_FOR_SPEEDUP}")
    print(f"   • Random Seed: {SEED}")
    print(f"   • Results Directory: {benchmark_dir}")
    
    results = {'time_vs_n': [], 'speedup_vs_k': []}
    
    print("\n" + "="*60)
    print("Phase 1: Performance vs Input Size")
    print("="*60)
    for n in NODES_TESTS:
        m = run_experiment(n, FIXED_K_FOR_TIME)
        if m: 
            results['time_vs_n'].append({'n': n, 'k': FIXED_K_FOR_TIME, 'metrics': m})
        time.sleep(1)  # Brief pause between runs
        
    print("\n" + "="*60)
    print("Phase 2: Scalability vs Number of Machines")
    print("="*60)
    for k in RANKS_TESTS:
        m = run_experiment(FIXED_N_FOR_SPEEDUP, k)
        if m: 
            results['speedup_vs_k'].append({'n': FIXED_N_FOR_SPEEDUP, 'k': k, 'metrics': m})
        time.sleep(1)  # Brief pause between runs
        
    # Save raw data in benchmark directory
    results_file = os.path.join(benchmark_dir, 'benchmark_results.json')
    print("\nSaving benchmark data...")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   SUCCESS: Raw data saved to: {results_file}")
        
    # Generate plots in benchmark directory
    print("\nGenerating visualizations...")
    combined_plot = plot_results(results, benchmark_dir)
    print(f"   SUCCESS: Combined analysis: {combined_plot}")
    print(f"   SUCCESS: Individual plots saved in: {benchmark_dir}")
    
    # Print summary
    print_summary(results)
    
    print("\n" + "="*60)
    print("BENCHMARKING COMPLETE!")
    print("="*60)
    print("Generated Files:")
    print(f"   • {results_file} (Raw data)")
    print(f"   • {benchmark_dir}/benchmark_analysis.png (Combined analysis)")
    print(f"   • {benchmark_dir}/plot_performance.png (Performance comparison)")
    print(f"   • {benchmark_dir}/plot_speedup.png (Scalability analysis)")
    print(f"   • {benchmark_dir}/plot_rounds.png (Communication complexity)")
    print("\nNext Steps:")
    print("   • Run: python main.py --nodes 100 --animate")
    print("   • This will generate convergence charts and animations")
    print("="*60)

if __name__ == "__main__":
    main()