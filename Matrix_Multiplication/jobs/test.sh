#!/bin/bash
#SBATCH --job-name="MM"
#SBATCH --get-user-env
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --exclusive
#SBATCH --output="test.out"
#SBATCH --account=ict24_dssc_cpu
#SBATCH --mem=480000          # memory per node out of 512000MB (512GB)
#SBATCH --ntasks-per-node=1

date
pwd
hostname

echo "Loading modules..."

ml load nvhpc/23.11
ml load openblas/0.3.24--nvhpc--23.11
ml load openmpi/4.1.6--nvhpc--23.11
ml load cuda/12.1

cd ~/ADV_HPC/matmul

echo "Compiling..."
make clean
make

echo "Running..."

mpirun -np 1 --map-by ppr:1:node:pe=112 --report-bindings ./run.x -i -e0 -n 1000 -o ./data/test_cpu_data.csv

mpirun -np 1 --map-by ppr:1:node:pe=112 --report-bindings ./run.x -i -e1 -n 1000 -o ./data/test_cpu_data.csv



date


###     --display-map
###