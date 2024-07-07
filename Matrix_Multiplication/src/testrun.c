#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cblas.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>


// salloc -p boost_usr_prod -A ict24_dssc_gpu -N1 -n4 -t 00:30:00 --gres=gpu:4 --qos=boost_qos_dbg
// mpirun -np 4 --map-by ppr:4:node:pe=1 --report-bindings ./run.x -r -n 10

void test_copy_gather(double *B, double *B_loc,double *B_mult,unsigned long long int N_loc,unsigned long long  int N,unsigned long long  int ncol, int *nrows, int *displs);
int* test_get_nrows_displs(unsigned long long int N, unsigned long long int ncol, int size);


/**
 * @brief Performs matrix multiplication AxB=C, initializing A to be the identity and B to be integers from 0 to N*N-1.
 * 
 * 
 * @param N The number of rows and columns of matrix A and B.
 * @param fname The name of the file containing the matrix values (bin file).
 * @param datacsv The name of the file where to save timings.
*/
void testrun(unsigned long long int N, char *fname, char *datacsv)
{

    // MPI initialization
    int mpi_provided_thread_level;

    MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &mpi_provided_thread_level);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // cuda visible devices
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    cudaSetDevice(rank % nDevices);

    double t1 = MPI_Wtime(); 

    // Data initialization
    unsigned long long int N_loc = N/size + 1*(rank < N%size);
    double *A = identity_initialize(N, N, size, rank, fname);
    double *B = sequence_initialize(N, N, size, rank, fname);
    double *C = (double*)calloc(N_loc*N,sizeof(double));

    double t2 = MPI_Wtime(); // data created

    // allocate memory on GPU for A, C, B_mult
    // copy A to GPU

    double *d_A, *d_B_mult,*d_B_mult_tmp , *d_C;
    cudaMalloc((void **)&d_A, N_loc * N * sizeof(double));
    cudaMalloc((void **)&d_C, N_loc * N * sizeof(double));

    unsigned long long int max_ncol = N/size + 1;
    cudaMalloc((void **)&d_B_mult, N * max_ncol * sizeof(double));
    cudaMalloc((void **)&d_B_mult_tmp, N * max_ncol * sizeof(double));

    double *B_loc = (double *)malloc(sizeof(double) * N_loc * max_ncol);
    double *B_mult = (double *)calloc(N * max_ncol, sizeof(double));
    
    cudaMemcpy(d_A, A, N_loc * N * sizeof(double), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const double alpha = 1.0f;
    const double beta = 0.0f;

    double computational_time = 0.0;
    double communication_time = 0.0;

    // Matrix multiplication
    for (int i = 0; i < size; i++)
    {
        unsigned long long int ncol, offset;
        int* nrows;
        double t3 = MPI_Wtime();
        
        if (i == 0){

            ncol = N/size + 1*(i < N%size);
            offset = ncol*i + N%size*(i >= N%size);
            nrows = test_get_nrows_displs(N, ncol, size);

            // get the rows of B to be multiplied in a single matrix and copy them to GPU
            test_copy_gather(&B[offset],B_loc,B_mult, N_loc, N, ncol, nrows, &nrows[size]);
            cudaMemcpy(d_B_mult, B_mult, N * ncol * sizeof(double), cudaMemcpyHostToDevice);
        }
        
        
        double t4 = MPI_Wtime();

        // Perform the actual (block) matrix multiplication
        #pragma omp parallel
        {
            #pragma omp single nowait private(ncol,offset)
            {
                cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ncol, N_loc, N, &alpha, d_B_mult, ncol, d_A, N, &beta, &d_C[offset], N);
            }

            #pragma omp single
            {
                int j = i+1;
                ncol = N/size + 1*(j < N%size);
                offset = ncol*j + N%size*(j >= N%size);
                nrows = test_get_nrows_displs(N, ncol, size);
                #pragma omp task 
                {

                    test_copy_gather(&B[offset],B_loc,B_mult, N_loc, N, ncol, nrows, &nrows[size]);
                    cudaMemcpy(d_B_mult_tmp, B_mult, N * ncol * sizeof(double), cudaMemcpyHostToDevice);
                }
            }
            
        }
        d_B_mult = d_B_mult_tmp;
        cudaDeviceSynchronize();

        double t5 = MPI_Wtime();
        printf("i=%d\n",i);

        computational_time += t5-t4;
        communication_time += t4-t3;

        free(nrows);
    }

    double t6 = MPI_Wtime(); 

    cudaFree(d_A);
    cudaFree(d_B_mult);
    cudaFree(d_C);
    cublasDestroy(handle);

    free(B_loc);
    free(B_mult);

    double init_time = t2-t1;
    double global = t6-t1;
    double computational_global, communication_global, init_global, global_max;

    MPI_Reduce(&computational_time,&computational_global,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&communication_time,&communication_global,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&global,&global_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&init_time,&init_global,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

    if (rank == 0)
        write_csv(2,size,N,global_max,computational_global,communication_global,init_global,datacsv);


    MPI_Finalize();

}


/**
 * @brief Copies ncol columns of matrix B stored by rows on each process in contigous memory B_mult.
 * 
 * @param B The matrix to be copied.
 * @param N_loc The number of rows of matrix B the process has access to.
 * @param ncol The number of columns to be copied.
 * @param nrows counts of elements to be received from each process for the Allgather.
 * @param displs displacements where to place the received elements for the Allgather.
*/
void test_copy_gather(double *B,double *B_loc,double *B_mult,unsigned long long int N_loc,unsigned long long int N,unsigned long long int ncol, int *nrows, int *displs)
{
    #pragma omp parallel for //collapse(2)
    for (int k = 0; k < N_loc; k++){
        for (int j = 0; j < ncol; j++){
            B_loc[k * ncol + j] = B[N * k + j];
        }
    }

    MPI_Allgatherv(B_loc, N_loc * ncol, MPI_DOUBLE, B_mult, nrows, displs, MPI_DOUBLE, MPI_COMM_WORLD);
}

int* test_get_nrows_displs(unsigned long long int N, unsigned long long int ncol, int size){
    int *nrows=(int*)calloc(2*size, sizeof(int));

    for(int i=0;i<size;i++){
        nrows[i] = (N/size + 1*(i < N%size)) * ncol;
    }

    for(int i=size+1;i<2*size;i++){
        nrows[i] += nrows[i-size-1] + nrows[i-1];
    }

    return nrows;
}
