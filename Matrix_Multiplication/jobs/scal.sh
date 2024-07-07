#!/bin/bash

# input size of matrix
export MATRIX_SIZE=$1

if [ $2 == "cpu" ] || [ $2 == "gpu" ]
then
    export TYPE=$2
else
    echo "Invalid type. Choose between cpu or gpu."
    exit 0
fi

if [ $TYPE == "cpu" ]
then
    if [ $3 == "0" ] || [ $3 == "1" ]
    then
        export MATMUL_BLAS=$3
    else
        echo "Invalid type. Choose between 0 or 1 (MATMUL or BLAS)."
    fi
fi

if [ $TYPE == "cpu" ]
then
    export NUM_THREADS=112
else 
    export NUM_THREADS=8
fi


echo "Matrix size:           $MATRIX_SIZE"
echo "Type:                  $TYPE"

if [ $TYPE == "cpu" ]
then
echo "Matmul or BLAS:        $MATMUL_BLAS"
fi


# create output directory
cd ~/Adv_HPC/Matrix_Multiplication
mkdir -p slurm_out

for i in {1,2,4,8,16,32,64}
do
    export NUM_NODES=$i

    if [ "$TYPE" == "cpu" ]
    then
        if [ $MATMUL_BLAS == "0" ]
        then
            sbatch -N $i -c112 -p dcgp_usr_prod -A ict24_dssc_cpu --mem=160G --gres=tmpfs:100g --output="./slurm_out/cpu_matmul_$i.out"                 ./jobs/$TYPE.sh
        else
            sbatch -N $i -c112 -p dcgp_usr_prod -A ict24_dssc_cpu --mem=160G --gres=tmpfs:100g --output="./slurm_out/cpu_blas_$i.out"                    ./jobs/$TYPE.sh
        fi
    fi

    if [ "$TYPE" == "gpu" ] 
    then
        sbatch -N $i --ntasks-per-node=4 -c8 -p boost_usr_prod -A ict24_dssc_gpu --gres=gpu:4 --mem=160G --exclusive --output="./slurm_out/gpu_$i.out" ./jobs/$TYPE.sh
    fi
done

