#!/bin/bash
#SBATCH --job-name=boruvka_scaling
#SBATCH --output=scaling_experiment_%j.out
#SBATCH --error=scaling_experiment_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=00:45:00
#SBATCH --partition=debug

# Load required modules (adjust based on your cluster)
module load python/3.12.1
module load openmpi/4.1.1

# Activate conda environment if needed
# source activate trial

# Install required Python packages (only if not already installed)
pip install --user networkx matplotlib mpi4py imageio 2>/dev/null || true

# Fixed parameters
NODES=10000
SEED=42


RANK_COUNTS=(2 4 6 8 10 12 14 16 20 24 28 32 40 48 56 64)

echo ""
echo "Starting scaling experiments..."
echo "Testing with ranks: ${RANK_COUNTS[@]}"
echo ""

# Run experiments for each rank count
# for RANKS in "${RANK_COUNTS[@]}"
# do
#     echo "========================================"
#     echo "Running with ${RANKS} ranks"
#     echo "========================================"
#     
#     python3 main.py --nodes ${NODES} --ranks ${RANKS} --seed ${SEED} 2>&1 | tee ${RUN_DIR}/run.log
# 
# done
# 
# echo "Experiment complete!"

python3 main.py --nodes 10000 --ranks 5 --seed 42

echo "Experiment complete!"
