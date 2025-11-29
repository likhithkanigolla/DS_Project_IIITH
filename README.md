# Distributed Boruvka MST (MPI, Python)

This project implements a distributed version of Boruvka's Minimum Spanning Tree algorithm
using `mpi4py`. It runs locally (multiple MPI ranks on one machine) and on the ADA SLURM cluster.

## Installation

```
python -m pip install -r requirements.txt
```

## Requirements
- Python 3.8+
- `mpi4py` (required)
- `networkx` (for `--generate` random graphs)
- `matplotlib` (visualization frames)
- `imageio` (GIF animation)

## Usage Examples

Local existing graph:
```
mpirun -np <K> python main.py --nodes <K> --input graph.txt
```

Generate random graph (rank 0 generates):
```
mpirun -np <K> python main.py --nodes <K> --generate 100
```

Cluster (ADA):
```
module load openmpi/4.1.5
mpirun -np <K> python main.py --nodes <K> --input graph.txt --mode cluster
```

Debug logging:
```
mpirun -np 4 python main.py --nodes 4 --generate 50 --debug
```

Disable animation:
```
mpirun -np 4 python main.py --nodes 4 --generate 50 --no-animate
```

## Files
- `main.py` – CLI entrypoint
- `boruvka.py` – distributed Boruvka implementation + results persistence
- `kruskal.py` – sequential Kruskal for correctness & performance comparison
- `graph_utils.py` – load / generate / partition helpers
- `dsu.py` – Union-Find (path compression + union-by-rank)
- `metrics.py` – timing & communication metrics
- `visualization.py` – frame rendering & GIF builder
- `slurm_template.sh` – sample SLURM script

## Visualization & Output
After each run results appear under `results/<timestamp>/`:
- `mst_boruvka.txt`: Boruvka MST edges + total weight
- `mst_kruskal.txt`: Kruskal MST edges + total weight (reference)
- `comparison.txt`: weight comparison (should be equal for connected graphs)
- `performance.txt`: Boruvka total time, Kruskal time, speedup ratio
- `metrics.txt`: iteration timing & communication rounds
- `frames/`: per-iteration PNG frames + `final_mst.png`
- `mst_animation.gif`: animated MST growth (unless `--no-animate`)
- `iteration_log.jsonl`: one JSON object per iteration with fields:
	- `iteration`: 1-based iteration index
	- `mst_size`: number of MST edges after that iteration
	- `unions`: list of component-root pairs `[[a, b], ...]` applied that iteration

## Input Format
Plain text edge list (one per line):
```
u v w
```
Example:
```
0 1 3.5
1 2 1.2
0 2 4.0
```
Nodes should be contiguous integers starting at 0.

## ADA Cluster Notes
- Non-interactive: rely solely on CLI args.
- Load MPI module before running.
- Use `--no-animate` if avoiding matplotlib/imageio overhead on large jobs.

