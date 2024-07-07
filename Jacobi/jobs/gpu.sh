#!/bin/bash
date

export NUM_PROC=$((4*$NUM_NODES))

cd ~/Adv_HPC/Jacobi

ml load nvhpc/24.3
ml load openmpi/4.1.6--nvhpc--24.3


echo "############################################"
echo "Matrix size:           $MATRIX_SIZE"
echo "Number of iterations:  $NUM_ITER"
echo "Number of nodes:       $NUM_NODES"
echo "Number of processes:   $NUM_PROC"
echo "Number of threads:     $NUM_THREADS"
echo "############################################"


mpirun -np $NUM_PROC --map-by ppr:4:node:pe=$NUM_THREADS ./gpu_run.x "$MATRIX_SIZE" "$NUM_ITER" 10 10

date