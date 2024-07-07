#!/bin/bash

# input size of matrix
export MATRIX_SIZE=$1
export NUM_ITER=$2

if [ $3 == "cpu" ] || [ $3 == "gpu" ] || [ $3 == "oneside" ]
then
    export TYPE=$3
else
    echo "Invalid type. Choose between cpu or gpu."
    exit 0
fi

if [ $3 == "cpu" ] || [ $3 == "oneside" ]
then
    export NUM_THREADS=112
else
    export NUM_THREADS=8
fi

echo "Matrix size:           $MATRIX_SIZE"
echo "Number of iterations:  $NUM_ITER"
echo "file:                  $TYPE"
echo "Number of threads:     $NUM_THREADS"

# create output directory
cd ~/Adv_HPC/Jacobi
mkdir -p slurm_out


for i in {1,2,4,8,16,32,64}
do
    export NUM_NODES=$i


    if [ "$TYPE" == "cpu" ]
    then
        sbatch -N $i -c112 -p dcgp_usr_prod -A ict24_dssc_cpu --mem=160G --output="./slurm_out/cpu_$i.out"                                             ./jobs/$TYPE.sh
    fi

    if [ "$TYPE" == "oneside" ]
    then
        sbatch -N $i -c112 -p dcgp_usr_prod -A ict24_dssc_cpu --mem=160G --output="./slurm_out/oneside_$i.out"                                          ./jobs/$TYPE.sh
    fi


    if [ "$TYPE" == "gpu" ] 
    then
        sbatch -N $i --ntasks-per-node=4 -c8 -p boost_usr_prod -A ict24_dssc_gpu --gres=gpu:4 --mem=160G --exclusive --output="./slurm_out/gpu_$i.out" ./jobs/$TYPE.sh
    fi
done

