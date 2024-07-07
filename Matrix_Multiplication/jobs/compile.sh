#!/bin/bash

echo "Compiling..."

sbatch -N1 -n1 -p boost_usr_prod --qos=boost_qos_dbg -A ict24_dssc_gpu --gres=gpu:1 --output="./slurm_out/compile.out" ./jobs/compile_run.sh

