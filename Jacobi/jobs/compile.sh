#!/bin/bash
if [ $1 == "cpu" ] || [ $1 == "gpu" ] || [ $1 == "oneside" ]
then
    export TYPE=$1
else
    echo "Invalid type. Choose between cpu gpu or oneside."
    exit 0

fi

echo "Compiling $TYPE version"
if [ "$TYPE" == "cpu" ]
then
    sbatch -N1 -n1 -p dcgp_usr_prod -A ict24_dssc_cpu  --output="./slurm_out/cpu_compile.out"          ./jobs/compile_run.sh
fi

if [ "$TYPE" == "gpu" ] 
then
    sbatch -N1 -n1 -p boost_usr_prod --qos=boost_qos_dbg -A ict24_dssc_gpu --gres=gpu:1 --output="./slurm_out/gpu_compile.out" ./jobs/compile_run.sh
fi

if [ "$TYPE" == "oneside" ]
then
    sbatch -N1 -n1 -p dcgp_usr_prod -A ict24_dssc_cpu  --output="./slurm_out/oneside_compile.out"          ./jobs/compile_run_oneside.sh
fi
