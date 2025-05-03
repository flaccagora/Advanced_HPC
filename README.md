# Advanced High Performance Computing

Solutions to the exercises of the course Advanced High Performance Computing (2024). This repository focuses on implementing and optimizing distributed parallel algorithms for two key computational problems.

## Overview

The repository contains implementations of:

1. **Jacobi Method** - A parallel implementation of the iterative Jacobi method for solving systems of linear equations
2. **Matrix Multiplication** - Distributed implementations of matrix multiplication algorithms

## Assignment 1: Jacobi Method

The Jacobi method is an iterative algorithm for determining the solutions of a diagonally dominant system of linear equations. In this assignment, we parallelize and optimize the algorithm using:

- MPI for distributed memory parallelism
- OpenMP for shared memory parallelism
- CUDA for GPU acceleration
- One-sided MPI communication as an optimization strategy

### Running the Jacobi Solver

To compile the program:
```bash
cd Jacobi
bash jobs/compile.sh [cpu|gpu|oneside]
```
Where `cpu`, `gpu`, or `oneside` specifies the version to compile.

To run a scaling study:
```bash
bash jobs/scal.sh [MATRIX_SIZE] [ITERATIONS] [cpu|gpu|oneside]
```

Parameters:
- `MATRIX_SIZE`: Size of the matrix (N×N)
- `ITERATIONS`: Number of Jacobi iterations to perform
- `cpu|gpu|oneside`: Implementation to use

## Assignment 2: Matrix Multiplication

This assignment implements and compares different approaches to distributed matrix multiplication:

- Naive implementation (basic distributed algorithm)
- CBLAS implementation (CPU-optimized using optimized linear algebra library)
- CUBLAS implementation (GPU-accelerated using NVIDIA's linear algebra library)

All versions distribute computation across multiple nodes while optimizing for performance.

### Running Matrix Multiplication

To compile:
```bash
cd Matrix_Multiplication
bash jobs/compile.sh
```

To run a scaling study:
```bash
bash jobs/scal.sh [MATRIX_SIZE] [cpu|gpu]
```

For CPU implementation, specify an additional argument:
```bash
bash jobs/scal.sh [MATRIX_SIZE] cpu [0|1]
```

Parameters:
- `MATRIX_SIZE`: Size of the matrices to multiply
- `cpu|gpu`: Platform to use
- `0|1`: When using CPU, specifies Naive (0) or CBLAS (1) implementation

## Repository Structure

- [`Jacobi`](Jacobi) - Jacobi method implementations (CPU, GPU, One-sided)
- [`Matrix_Multiplication`](Matrix_Multiplication) - Matrix multiplication implementations
- [`report`](report) - Performance analysis and documentation

## Performance Analysis

Both assignments include performance analysis with:
- Strong scaling measurements
- Communication vs. computation time breakdown
- Performance comparison across implementations
- Efficiency metrics at different scales

## Requirements

The code is designed to run on HPC clusters with:
- MPI implementation (for distributed computing)
- CUDA toolkit (for GPU implementations)
- BLAS libraries (for optimized CPU matrix operations)

## Results

Detailed performance analysis, scalability charts and implementation explanations are available in the [Report](./report/report.pdf).