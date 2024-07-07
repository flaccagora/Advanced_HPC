#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cblas.h>
#include <omp.h>

#include "./initialize.c"
#include "./utils.c"

void matmul(double* restrict A,double* restrict B,double* restrict C,const unsigned long long N, const unsigned long long M,const unsigned long long P);
int* cpu_get_nrows_displs(unsigned long long N, unsigned long long ncol, int size);
void cpu_copy_gather(double* restrict B, double* restrict B_loc, double* restrict B_mult, unsigned long long N_loc, unsigned long long N, unsigned long long ncol, int *nrows, int *displs);


/**
 * @brief Performs matrix multiplication AxB=C, initializing A to be the identity and B to be integers from 0 to (N*N)-1.
 * 
 * @param N The number of rows and columns of matrix A and B.
 * @param e Whether to use the custom matrix multiplication (e=0) or the BLAS library (e=1).
 * @param fname The name of the file containing the matrix values (bin file).
 * @param datacsv The name of the file where to save timings.
*/
void cpu_mult(unsigned long int N, int e,char *fname, char *datacsv )
{    

    // MPI initialization
    int mpi_provided_thread_level;
    
    MPI_Init_thread( NULL, NULL, MPI_THREAD_FUNNELED,&mpi_provided_thread_level);

    int rank, size; 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double t1 = MPI_Wtime();

    // Data initialization
    unsigned long long N_loc = N/size + 1*(rank < N%size);
    double *A = identity_initialize(N, N, size, rank, fname); 
    double *B = sequence_initialize(N, N, size, rank, fname); 
    double *C = (double*)calloc(N_loc*N,sizeof(double));

    unsigned long long max_ncol = N/size + 1;
    double *B_loc = (double*)malloc(sizeof(double)*N_loc*max_ncol);
    double *B_mult = (double*)calloc(N*max_ncol, sizeof(double));


    double t2 = MPI_Wtime(); // data created

    double computational_time = 0.0;
    double communication_time = 0.0;

    // Matrix multiplication
    for (int i = 0; i < size; i++)
    {
        double t3 = MPI_Wtime(); 
   
        unsigned long long ncol = N/size + 1*(i < N%size);
        unsigned long long offset = ncol*i + N%size*(i >= N%size);
        int* nrows = cpu_get_nrows_displs(N, ncol, size);

        // get the rows of B to be multiplied in a single matrix
        if(size != 1)
        cpu_copy_gather(&B[offset],B_loc,B_mult, N_loc, N, ncol, nrows, &nrows[size]);
        else
        B_mult = B;
        
        double t4 = MPI_Wtime(); 

        // Perform the actual (block) matrix multiplication
        if (e == 0)
        matmul(A,B_mult,&C[offset],N_loc,N,ncol);
        else if (e == 1)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N_loc, ncol, N, 1.0, A, N, B_mult, ncol, 0.0, &C[offset], N);
        
        double t5 = MPI_Wtime(); 
        
        computational_time += t5-t4;
        communication_time += t4-t3;

        free(nrows);
    }

    double t6 = MPI_Wtime(); 

    
    double init_time = t2-t1; 
    double global = t6-t1;
    double computational_global, communication_global, init_global, global_max;

    MPI_Reduce(&computational_time,&computational_global,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&communication_time,&communication_global,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&global,&global_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&init_time,&init_global,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

    if (rank == 0)
    write_csv(e,size,N,global_max,computational_global,communication_global,init_global,datacsv);

    MPI_Finalize();
    
}


/**
 * @brief Copies ncol columns of matrix B stored by rows on each process in contigous memory B_mult.
 * 
 * @param B The matrix to be copied.
 * @param B_loc The matrix where to store columns of B contiguosly.
 * @param B_mult The matrix where to store the copied elements from all processes allagather B_loc.
 * @param N_loc The number of rows of matrix B the process has access to.
 * @param ncol The number of columns to be copied.
 * @param nrows counts of elements to be received from each process for the Allgather.
 * @param displs displacements where to place the received elements for the Allgather.
*/
void cpu_copy_gather(double* restrict B,double* restrict B_loc, double* restrict B_mult, unsigned long long int N_loc, unsigned long long int N,unsigned long long int ncol, int *nrows, int *displs)
{
    #pragma omp parallel for collapse(2)
    for(unsigned long long int k = 0; k < N_loc; k++){
        for(unsigned long long int j = 0; j < ncol; j++){
            B_loc[k*ncol+j] = B[ N*k + j ];
        }
    }

    MPI_Allgatherv(B_loc, N_loc*ncol, MPI_DOUBLE, B_mult, nrows, displs, MPI_DOUBLE, MPI_COMM_WORLD);

}

/**
 * @brief Naive matrix multiplication AxB=C.
 * 
 * @param A The first matrix.
 * @param B The second matrix.
 * @param C The resulting matrix.
 * @param N The number of rows of matrix A and C.
 * @param M The number of columns of matrix A and rows of matrix B.
 * @param P The number of columns of matrix B and C.
*/
void matmul(double* restrict A,double* restrict B,double* restrict C,const unsigned long long int N,const unsigned long long int M,const unsigned long long int P){

    #pragma omp parallel for collapse(2)
    for(int i = 0; i < N; i++){
        for(int j = 0; j < P; j++){   
            double sum = 0.0;
            for(int k = 0; k < M; k++){
                sum += A[i*M+k] * B[k*P+j];
            }        
            C[i*M+j] = sum;  
        }
    }

}

int* cpu_get_nrows_displs(unsigned long long int N,unsigned long long int ncol, int size){
    int *nrows=(int*)calloc(2*size, sizeof(int));

    for(int i=0;i<size;i++){
        nrows[i] = (N/size + 1*(i < N%size)) * ncol;
    }

    for(int i=size+1;i<2*size;i++){
        nrows[i] += nrows[i-size-1] + nrows[i-1];
    }

    return nrows;
}
