# Distributed Borůvka MST Implementation

This project implements and compares two versions of Borůvka's Minimum Spanning Tree algorithm:
- **Sequential Borůvka**: Traditional single-threaded implementation 
- **Distributed Borůvka**: Parallel implementation using MPI across multiple ranks

Both algorithms are validated against Kruskal's algorithm to ensure correctness.

## 🚀 Quick Start

```bash
# Install dependencies
python -m pip install -r requirements.txt

# Run comparison with 10 nodes, 2 MPI ranks
source venv/bin/activate
python main.py --nodes 10 --seed 42 --ranks 2

# Validate results
python validate_mst.py results/[timestamp]/graph_n10_s42.txt \
    results/[timestamp]/sequential/mst_sequential.txt \
    results/[timestamp]/distributed/mst_distributed.txt
```

## 📋 Requirements
- Python 3.8+
- `mpi4py` (MPI implementation)
- `networkx` (graph generation)
- `matplotlib` (visualization)
- `imageio` (GIF animation)

## 🏗️ Architecture

### Core Files
- **`main.py`** – Main orchestrator that generates graphs and runs both algorithms
- **`sequential_boruvka.py`** – Sequential Borůvka implementation (rank 0 only)
- **`distributed_boruvka.py`** – Distributed Borůvka across multiple MPI ranks
- **`validate_mst.py`** – Validation script using Kruskal's algorithm

### Supporting Libraries
- **`dsu.py`** – Disjoint Set Union with path compression and union-by-rank
- **`graph_utils.py`** – Graph generation, loading, and partitioning utilities
- **`metrics.py`** – Performance timing and communication metrics
- **`visualization.py`** – Animation generation and result visualization

## 📊 Usage Examples

### Basic Comparison
```bash
python main.py --nodes 50 --seed 123 --ranks 4
```

### With Animation
```bash
python main.py --nodes 20 --seed 42 --ranks 2 --animate
```

### Larger Scale
```bash
python main.py --nodes 100 --seed 456 --ranks 8
```

## 🔬 Algorithm Analysis

### Sequential Borůvka
- Runs entirely on rank 0
- Other MPI ranks remain idle
- Iteratively finds minimum outgoing edges per component
- Components merge until single connected component

### Distributed Borůvka  
- Graph partitioned across MPI ranks
- Each rank finds local minimum outgoing edges
- Rank 0 coordinates global minimum selection
- All ranks synchronize component unions
- Achieves parallelism through distributed computation

### Validation
The `validate_mst.py` script ensures both algorithms produce optimal MSTs by:
1. Computing optimal MST using Kruskal's algorithm
2. Comparing weights with Sequential and Distributed outputs
3. Reporting which algorithms match the optimal solution

## 📁 Output Structure

Results are saved to `results/[timestamp]/`:
```
graph_n[N]_s[seed].txt          # Input graph
platform_comparison.txt         # Performance comparison
sequential/
├── mst_sequential.txt          # Sequential MST result
├── metrics.txt                 # Timing metrics
└── iteration_log.jsonl         # Per-iteration details
distributed/
├── mst_distributed.txt         # Distributed MST result  
├── metrics.txt                 # Timing metrics
└── iteration_log.jsonl         # Per-iteration details
```

## 🎯 Key Features

### ✅ Correctness Validation
- Both algorithms validated against Kruskal's optimal solution
- Fixed critical bugs in edge selection and component management
- Ensures identical optimal MST weights

### 📈 Performance Metrics
- Execution time comparison
- Iteration count analysis  
- Communication round tracking
- Speedup ratio calculation

### 🎬 Visualization Support
- Per-iteration MST growth animations
- Directed edges colored by source rank
- HTML tables with detailed iteration data

### 🐛 Bug Fixes Applied
1. **Distributed Algorithm**: Fixed component edge overwriting issue
2. **Sequential Algorithm**: Fixed bidirectional edge consideration  
3. **File I/O**: Corrected newline handling in graph files

## 📐 Graph Format

Input graphs use simple edge list format:
```
n m
u1 v1 w1
u2 v2 w2
...
```
Where `n` = nodes, `m` = edges, each line contains source, destination, weight.

## 🔧 Technical Implementation

### Borůvka's Algorithm Steps
1. **Initialize**: Each node as separate component
2. **Find**: Minimum outgoing edge per component  
3. **Merge**: Union components via selected edges
4. **Repeat**: Until single connected component

### Distributed Coordination
1. **Partition**: Graph edges distributed across ranks
2. **Local Computation**: Each rank finds local minimum edges
3. **Global Selection**: Rank 0 selects optimal edges per component
4. **Synchronization**: All ranks apply same component unions

### Performance Characteristics
- **Sequential**: Fast for small graphs, no communication overhead
- **Distributed**: Scales with more ranks, but has MPI coordination cost
- **Typical Speedup**: Distributed often slower due to small graph sizes and communication overhead

## 🧪 Validation Results

Latest validation confirms both algorithms now produce optimal MSTs:
```
Optimal (Kruskal): weight=292.206990 edges=9
Sequential: weight=292.206990 edges=9 ✅  
Distributed: weight=292.206990 edges=9 ✅
```

## 🚀 Future Enhancements
- Cluster deployment scripts
- Larger graph benchmarking
- Advanced visualization options
- Performance optimization for distributed version

