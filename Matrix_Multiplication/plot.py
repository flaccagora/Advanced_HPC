import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def main(args):

    N = args.N
    type = args.type

    if args.type=='MATMUL':
        path = f"./data/cpu_data_0.csv"
    elif args.type=='C-BLAS':
        path = f"./data/cpu_data_1.csv"
    elif args.type=="CU-BLAS":
        path = f"./data/gpu_data.csv"
    else:
        raise ValueError("Invalid type or naive flag")
    
    
    data = pd.read_csv(path)
    data = data.groupby(["type","n_proc","n_thread","N"]).mean().reset_index()

    os.system(f"mkdir -p plots/{type}")
    os.chdir(f"plots/{type}")

    data[data['N']==N].plot(x='n_proc', y=['init','computation','communication'], kind='bar', stacked=True, title=f'{type} N={N}')
    plt.savefig(f'{type}_{N}.png')
    data[data['N']==N].plot(x='n_proc', y=['init','computation','communication'], kind='bar', stacked=False, logy=True)
    plt.savefig(f'{type}_{N}_log.png')

    data[data['N']==N].plot(x='n_proc', y=['init'], kind='line', title=f'{type} N={N} - INIT')
    plt.savefig(f'{type}_{N}_init.png')

    data[data['N']==N].plot(x='n_proc', y=['computation'], kind='line', title=f'{type} N={N} - COMPUTATION')
    plt.savefig(f'{type}_{N}_computation.png')

    data[data['N']==N].plot(x='n_proc', y=['communication'], kind='line', title=f'{type} N={N} - COMMUNICATION')
    plt.savefig(f'{type}_{N}_communication.png')
                




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='What to plot.')
    parser.add_argument('--N', type=int, help='size of the matrix')
    parser.add_argument('--n_thread', type=int,default=112, help='number of threads')
    parser.add_argument('--type', type=str,default="matmul", options=['MATMUL','C-BLAS', 'CU-BLAS'], help='type of the data to plot')
    args = parser.parse_args()

    main(args)