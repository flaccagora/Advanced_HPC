#!/bin/bash

date 

ml load nvhpc/24.3
ml load openmpi/4.1.6--nvhpc--24.3

make -f ./makefiles/Makefile_$TYPE clean
mpirun -np 1 make -f ./makefiles/Makefile_$TYPE 

date