#!/bin/bash
date 

ml load nvhpc/23.11
ml load openblas/0.3.24--nvhpc--23.11
ml load openmpi/4.1.6--nvhpc--23.11
ml load cuda/12.1


export NUM_PROC=$((4*$NUM_NODES))

echo "############################################"
echo "Matrix size:           $MATRIX_SIZE"
echo "Number of nodes:       $NUM_NODES"
echo "Number of processes:   $NUM_PROC"
echo "Number of threads:     $NUM_THREADS (per process)"
echo "Type:                  $TYPE"
echo "############################################"



mpirun -np $NUM_PROC --map-by ppr:4:node:pe=$NUM_THREADS ./run.x -r -n $MATRIX_SIZE -o ./data/gpu_data.csv

date


# mpirun -np 1 --map-by ppr:4:node:pe=8 nsys profile -o ${PWD}/output_%q{OMPI_COMM_WORLD_RANK} -f true --stats=true --cuda-memory-usage=true ./run.x -r -n $MATRIX_SIZE -o ./data/gpu_data.csv
