#!/bin/bash
date 
ml load nvhpc/24.3
ml load openmpi/4.1.6--nvhpc--24.3

echo "############################################"
echo "Matrix size:           $MATRIX_SIZE"
echo "Number of iterations:  $NUM_ITER"
echo "Number of nodes:       $NUM_NODES"
echo "Number of processes:   $NUM_NODES"
echo "Number of threads:     $NUM_THREADS"
echo "############################################"


mpirun -np $NUM_NODES --map-by ppr:1:node:pe=$NUM_THREADS  ./oneside_run.x "$MATRIX_SIZE" "$NUM_ITER" 10 10

date

# mpirun -np 1 --map-by ppr:1:node:pe=24  ./oneside_run.x 10 2 10 10
