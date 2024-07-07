#!/bin/bash

date 

ml load nvhpc/23.11
ml load openblas/0.3.24--nvhpc--23.11
ml load openmpi/4.1.6--nvhpc--23.11
ml load cuda/12.1


make clean
mpirun -np 1 make

date