#include <stdlib.h>
#include <mpi.h>

/**
 * @brief Initializes a matrix at random.
 * 
 * This function initializes a matrix. It uses MPI to get the rank and size of the process. The matrix is allocated dynamically using malloc and the values are randomly set.
 * 
 * @param N The number of rows of the matrix.
 * @param M The number of columns of the matrix.
 * @param fname The name of the file containing the matrix values.
 * @return pointer to the initialized matrix.
 */
double* rand_initialize(int N, int M,  char* fname)
{
    double *A = (double*)malloc(N*M*sizeof(double));
    
    // Set random values for the matrix
    #pragma omp parallel for
    for (int i = 0; i < N*M; i++)
    {
        A[i] = (double)rand() / (double)RAND_MAX;
    }

    if (fname!=NULL){
        /*
            stampa su file la matrice in modo da poter verificare correttezza tramite hash
        */
    }
    

    return A;
}


double* identity_initialize_seq(unsigned long long int N,unsigned long long int M,char* fname){
    
    double *A = (double*)calloc(N*M, sizeof(double));
    
    // Set random values for the matrix
    #pragma omp parallel for 
    for (int i = 0; i < N; i++)
    {
                A[i*N+i] = 1.0;
    }

    if (fname!=NULL){
        /*
            stampa su file la matrice in modo da poter verificare correttezza tramite hash
        */
    }
    

    return A;

}


/**
 * @brief Initializes N by M Identity matrix.
 * 
 * This function initializes a matrix. It uses MPI to get the rank and size of the process. The matrix is allocated dynamically using malloc and the values are randomly set.
 * 
 * @param N The number of rows of the matrix.
 * @param M The number of columns of the matrix.
 * @param fname The name of the file containing the matrix values.
 * @return A pointer to the initialized matrix.
 */
double* identity_initialize(unsigned long long int N,unsigned long long int M, int size, int rank, char* fname)
{
    if (size == 1){
        return identity_initialize_seq(N, M, fname);
    }
    
    int N_loc = N/size + 1*(rank < N%size);
    int offset = rank*N_loc + N%size*(rank >= N%size);

    double *A = (double*)calloc(N_loc*N, sizeof(double));
    #pragma omp parallel for 
    for (int i = 0; i < N_loc; i++)
    {
                A[i*M+offset+i] = 1.0;
    }

    if (fname!=NULL){
        /*
            stampa su file la matrice in modo da poter verificare correttezza tramite hash
        */
    }
    

    return A;
}


/**
 * @brief Initializes N by M Identity matrix.
 * 
 * This function initializes a matrix. It uses MPI to get the rank and size of the process. The matrix is allocated dynamically using malloc and the values are randomly set.
 * 
 * @param N The number of rows of the matrix.
 * @param M The number of columns of the matrix.
 * @param fname The name of the file containing the matrix values.
 * @return A pointer to the initialized matrix.
 */
double* sequence_initialize_seq(unsigned long long int N,unsigned long long int M, char* fname)
{
    double *A = (double*)malloc(N*M*sizeof(double));
    
    // Set random values for the matrix
    #pragma omp parallel for 
    for (int i = 0; i < N*M; i++)
    {
        A[i] = i;
    }

    if (fname!=NULL){
        /*
            stampa su file la matrice in modo da poter verificare correttezza tramite hash
        */
    }

    return A;
}

/**
 * @brief Initializes N by M Identity matrix.
 * 
 * This function initializes a matrix. It uses MPI to get the rank and size of the process. The matrix is allocated dynamically using malloc and the values are randomly set.
 * 
 * @param N The number of rows of the matrix.
 * @param M The number of columns of the matrix.
 * @param fname The name of the file containing the matrix values.
 * @return A pointer to the initialized matrix.
 */
double* sequence_initialize(unsigned long long int N,unsigned long long int M,int size, int rank, char* fname)
{
 
    if (size == 1){
        return sequence_initialize_seq(N, M, fname);
    }
   
    int N_loc = N/size + 1*(rank < N%size);
    int offset = rank*N_loc + N%size*(rank >= N%size);

    double *A = (double*)malloc(N_loc*M*sizeof(double));
    
    #pragma omp parallel for 
    for (int i = 0; i < N_loc*M; i++)
    {
            A[i] = (offset*M+i);
    }

    if (fname!=NULL){
        /*
            stampa su file la matrice in modo da poter verificare correttezza tramite hash
        */
    }
    

    return A;
}


double* sequence_initialize_old(int N, int M,int size, int rank, char* fname)
{
 
    if (size == 1){
        return sequence_initialize_seq(N, M, fname);
    }
   
    int N_loc = N/size + 1*(rank < N%size);
    int offset = rank*N_loc + N%size*(rank >= N%size);

    double *A = (double*)malloc(N_loc*M*sizeof(double));
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N_loc; i++){
        for (int j = 0; j < M; j++){
            A[i*M+j] = (offset+i)*M+j;
        }
    }

    if (fname!=NULL){
        /*
            stampa su file la matrice in modo da poter verificare correttezza tramite hash
        */
    }
    

    return A;
}

