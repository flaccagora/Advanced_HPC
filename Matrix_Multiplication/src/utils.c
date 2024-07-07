#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>


void check_equality(double* A, double* B, long int N, long int M){
    // check h_a and h_a_bis are equal
    for(int i = 0; i < N*M; i++){
        if(A[i] != B[i]){
            printf("\n\n\nERROR: A[%d] = %f, B[%d] = %f\n\n\n", i, A[i], i, B[i]);
            break;
        }
    }
    printf("No error, identical matrix ");
}

void write_csv(int type, int size, int N, double global, double computational_global, double communication_global,double init_global, char* datacsv){
    
    /*
        columns
        type 
        0 cpu matmul
        1 cpu blas
        2 gpu


        1. size (#processi)
        2. num_threads
        2. N (matrix dimension)
        3. tempo di calcolo effettivo
        4. tempo di comunicazione
        5. tempo di inizializzazione delle matrici

    */

    FILE *csv_file = fopen(datacsv, "a");
    if (csv_file == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }
    


  fprintf(csv_file, "%d,%d,%d,%d,%f,%f,%f,%f\n",type,size,omp_get_max_threads(),N,global,computational_global,communication_global, init_global);
  
  fclose(csv_file);

  return;

}

void write_csv_gpu(int type, int size, int N, double global, double computational_global, double communication_global,double init_global,double gpu_init, char* datacsv){
    
    /*
        columns
        type 
        0 cpu matmul
        1 cpu blas
        2 gpu


        1. size (#processi)
        2. num_threads
        2. N (matrix dimension)
        3. tempo di calcolo effettivo
        4. tempo di comunicazione
        5. tempo di inizializzazione delle matrici

    */

    FILE *csv_file = fopen(datacsv, "a");
    if (csv_file == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }
    


  fprintf(csv_file, "%d,%d,%d,%d,%f,%f,%f,%f,%f\n",type,size,omp_get_max_threads(),N,global,computational_global,communication_global, init_global,gpu_init);
  
  fclose(csv_file);

  return;

}

// MATRIX PRINTS

void matrix_print(double *a, int dimx, int dimy) {

    for (int i=0; i<dimx*dimy; i++) {
        printf("%f ", a[i]);
        if ((i+1)%dimy == 0) printf("\n");
    }
    printf("\n");
}

void print_loc(double *mat, int loc_size, int LDA)
{

    int i, j;
    for (i = 0; i < loc_size; i++)
    {
        for (j = 0; j < LDA; j++)
        {
            fprintf(stdout, "%.3g ", mat[j + (i * LDA)]);
        }
        fprintf(stdout, "\n");
    }
}

void distributed_print(int rank, int size, int N_loc, int LDA, double *matrix)
{

    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < size; i++)
    {
        if (rank == i)
        {
            printf("rank %d, N_loc %d\n", rank, N_loc);
            print_loc(matrix, N_loc, LDA);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}
