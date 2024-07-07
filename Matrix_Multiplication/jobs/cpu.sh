#!/bin/bash
date 

ml load nvhpc/23.11
ml load openblas/0.3.24--nvhpc--23.11
ml load openmpi/4.1.6--nvhpc--23.11
ml load cuda/12.1


echo "############################################"
echo "Matrix size:           $MATRIX_SIZE"
echo "Number of nodes:       $NUM_NODES"
echo "Number of processes:   $NUM_NODES"
echo "Number of threads:     $NUM_THREADS"
echo "Type:                  $TYPE"
echo "Matmul or BLAS:        $MATMUL_BLAS"
echo "############################################"


mpirun -np $NUM_NODES --map-by ppr:1:node:pe=$NUM_THREADS ./run.x -i -n $MATRIX_SIZE -e $MATMUL_BLAS -o ./data/cpu_data_$MATMUL_BLAS.csv

date



# mpirun -np 1 --map-by ppr:1:node:pe=112 ./run.x -i -n 56000 -e 1 -o ./data/cpu_data_1.csv
