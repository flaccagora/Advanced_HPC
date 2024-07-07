#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>
#include <openacc.h>


#define OPENACC

/*** function declarations ***/

void init_matrix(const int rank,const int size, double* restrict matrix,double* restrict matrix_new, const int N_loc,const int dimension,const double increment,const int my_row);
void init_matrix_test(const int rank,const int size, double* restrict matrix,double* restrict matrix_new, const int N_loc,const int dimension,const double increment,const int my_row);

void evolve( double* restrict matrix, double* restrict matrix_new, const int N_loc, const size_t dimension);
void update_boundaries(int rank, int size, double* local_matrix, int my_row, int N);

void mpi_io(int rank, int size, double* local_matrix, int N_loc, int LDA, char* fname_snap, MPI_File file);

void print_loc( double * mat, int loc_size , int LDA);
void distributed_print(int rank, int size,int N_loc, int LDA,  double * matrix);

void write_csv(int size, int N,int iterations, double global, double computational_global, double communication_global,double init_global, double t_io, char* datacsv);

/*** end function declaration ***/


int main(int argc, char* argv[]){

  // MPI init
  MPI_Init(&argc, &argv);
  
  int rank, size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // timing variables
  double increment, t_comm=0;
  
  // initialize matrix
  double *matrix, *matrix_new, *tmp_matrix;

  size_t dimension = 0, iterations = 0, row_peek = 0, col_peek = 0;
  size_t byte_dimension = 0;

  // check on input parameters
  if(argc != 5) {
    fprintf(stderr,"\nwrong number of arguments. Usage: ./a.out dim it n m\n");
    return 1;
  }

  dimension = atoi(argv[1]);
  iterations = atoi(argv[2]);
  row_peek = atoi(argv[3]);
  col_peek = atoi(argv[4]);

  // if(rank==0){
  // printf("matrix size = %zu\n", dimension);
  // printf("number of iterations = %zu\n", iterations);
  // printf("element for checking = Mat[%zu,%zu]\n",row_peek, col_peek);
  // }
  // if((row_peek > dimension) || (col_peek > dimension)){
  //   fprintf(stderr, "Cannot Peek a matrix element outside of the matrix dimension\n");
  //   fprintf(stderr, "Arguments n and m must be smaller than %zu\n", dimension);
  //   return 1;
  // }


  int ngpu = acc_get_num_devices(acc_device_nvidia);
  int igpu = rank % ngpu;
  acc_set_device_num(igpu, acc_device_nvidia);
  acc_init(acc_device_nvidia);
  // if( !rank ) fprintf(stdout, "NUM GPU: %d\n", ngpu);
  // fprintf(stdout, "GPU ID: %d, PID: %d\n", igpu, rank);
  // //fflush( stdout );
  // printf("GPU ID: %d, PID: %d\n", igpu, rank);


  // Initialization of the matrix and the new matrix
  double t1 = MPI_Wtime(); // init data

  int N_loc = 2 + (dimension)/size + 1*(rank < (dimension)%size);

  byte_dimension = sizeof(double) * ( N_loc ) * ( dimension + 2 );
  matrix = ( double* )malloc( byte_dimension );
  matrix_new = ( double* )malloc( byte_dimension );

  #pragma acc enter data create ( matrix[ 0 : ( N_loc ) * ( dimension + 2 ) ], matrix_new[ 0 : ( N_loc ) * ( dimension + 2 ) ] )
 
  // set up incremet for boundary condition 
  increment = 100.0 / ( dimension + 1 );
  
  // starting row for each rank wrt the global matrix
  int my_row = rank*(N_loc-2) + (dimension%size)*(rank >= (dimension)%size);
  // printf("rank %d, my_row %d, N_loc %d\n", rank, my_row, N_loc);

  init_matrix(rank, size, matrix,matrix_new, N_loc, dimension, increment, my_row);
  
  // #pragma acc update self( matrix[ 0 : ( N_loc ) * ( dimension + 2 ) ])
  // distributed_print(rank, size, N_loc, dimension+2, matrix);

  double t2 = MPI_Wtime(); // init data

  // start algorithm
  for(size_t it = 0; it < iterations; ++it ){
    
    evolve( matrix, matrix_new,N_loc, dimension);

    // swap the pointers
    tmp_matrix = matrix;
    matrix = matrix_new;
    matrix_new = tmp_matrix;
    
    double t3 = MPI_Wtime(); // init data
    update_boundaries(rank, size, matrix, N_loc, dimension+2);
    double t4 = MPI_Wtime(); // init data

    t_comm += t4-t3;

  }
  
  double t5 = MPI_Wtime(); // init data
  
  mpi_io(rank, size, matrix, N_loc, dimension+2, "snap_gpu", MPI_FILE_NULL); 

  double t6 = MPI_Wtime(); // init data
  
  double t_compute = (t5-t2) - t_comm;
  double t_io = t6-t5;
  double t_init = t2-t1;
  double t_compute_max, t_comm_max, t_init_max, t_io_max;
  
  MPI_Reduce(&t_compute,&t_compute_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Reduce(&t_init,&t_init_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Reduce(&t_comm,&t_comm_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Reduce(&t_io,&t_io_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);


  double global = t_compute_max + t_init_max + t_comm_max + t_io_max;

  free( matrix );
  free( matrix_new );


  char* datacsv;
  datacsv = (char*)malloc(100);
  sprintf(datacsv, "%s","./data/gpu_data.csv");

  if (rank == 0)
    write_csv(size, dimension, iterations, global, t_compute_max,t_comm_max, t_init_max, t_io,datacsv);


  MPI_Finalize();

  return 0;
}

void init_matrix(const int rank,const int size, double* __restrict__ matrix,double* __restrict__ matrix_new, const int N_loc,const int dimension,const double increment,const int my_row){
   
  #pragma omp parallel
  {

  #pragma omp single nowait
  {
    #pragma omp task
    {
      //fill initial values (internal values are set to .5) 
    if (rank==0){
      #pragma acc parallel loop collapse(2) present( matrix )   
      for(int i = 1; i < N_loc; ++i ){
        for(int j = 1; j < dimension+1; ++j ){
        matrix[ ( i * ( dimension + 2 ) ) + j ] = 0.5;
        }
      }
    }else{
      #pragma acc parallel loop collapse(2) present( matrix )   
      for(int i = 0; i < N_loc; ++i ){
        for(int j = 1; j < dimension+1; ++j ){
        matrix[ ( i * ( dimension + 2 ) ) + j ] = 0.5;
        }
      }
    }
    }// task 1


  #pragma omp task
  {
  //fill initial values for the first column of the BIGMATRIX
  if (rank!=0 && rank!=size-1){
    #pragma acc parallel loop present( matrix, matrix_new )   
    for(int i = 0; i < N_loc; ++i ){
      matrix[ i * ( dimension + 2 ) ] = ( i + my_row ) * increment;
      matrix_new[ i * ( dimension + 2 ) ] = (i+my_row ) * increment;
    }
  }else if (rank==size-1){
    #pragma acc parallel loop present( matrix, matrix_new )   
    for(int i = 0; i < N_loc-1; ++i ){
      matrix[ i * ( dimension + 2 ) ] = ( i + my_row ) * increment;
      matrix_new[ i * ( dimension + 2 ) ] = (i+my_row) * increment;
    }
  }else if(rank == 0){
    #pragma acc parallel loop present( matrix, matrix_new )   
    for(int i = 1; i < N_loc; ++i ){
      matrix[ i * ( dimension + 2 ) ] = ( i + my_row ) * increment;
      matrix_new[ i * ( dimension + 2 ) ] = (i+my_row) * increment;
    }
  }
  
  }// task 2

  #pragma omp task
  {

  //fill initial values for the last row of the BIGMATRIX
  if(rank == size-1){
    #pragma acc parallel loop present( matrix, matrix_new )   
    for (int j = 0; j<dimension+2; ++j){
      matrix[ (N_loc)*(dimension+2) - 1-j] = j*increment;
      matrix_new[ (N_loc) * (dimension+2) -1- j ] = j*increment;;
    }
  }

    //fill initial values for the first row of the BIGMATRIX
  if(rank == 0){
    #pragma acc parallel loop present( matrix, matrix_new )   
    for (int j = 0; j<dimension+2; ++j){
      matrix[j] = 0.0;
      matrix_new[ j ] = 0.0;;
    }
  }
  }// task 3
  }// single
  }// parallel
}

void init_matrix_test(const int rank,const int size, double* restrict matrix,double* restrict matrix_new, const int N_loc,const int dimension,const double increment,const int my_row){
   
   int i, j;

  if (rank==0){
    #pragma acc parallel loop collapse(2) present( matrix )   
    for( i = 1; i < N_loc; ++i ){
      for( j = 1; j < dimension+1; ++j ){
      matrix[ ( i * ( dimension + 2 ) ) + j ] = 0.5;
      }
    }
    #pragma acc parallel loop present( matrix, matrix_new )   
    for( i = 1; i < N_loc; ++i ){
      matrix[ i * ( dimension + 2 ) ] = ( i + my_row ) * increment;
      matrix_new[ i * ( dimension + 2 ) ] = (i+my_row) * increment;
    }
    #pragma acc parallel loop present( matrix, matrix_new )   
    for (j = 0; j<dimension+2; ++j){
      matrix[j] = 0.0;
      matrix_new[ j ] = 0.0;;
    }

  }

  if (rank!=0 && rank!=size-1){
    #pragma acc parallel loop present( matrix, matrix_new )   
    for( i = 0; i < N_loc; ++i ){
      matrix[ i * ( dimension + 2 ) ] = ( i + my_row ) * increment;
      matrix_new[ i * ( dimension + 2 ) ] = (i+my_row ) * increment;
    }
    #pragma acc parallel loop collapse(2) present( matrix )   
    for( i = 0; i < N_loc; ++i ){
      for( j = 1; j < dimension+1; ++j ){
      matrix[ ( i * ( dimension + 2 ) ) + j ] = 0.5;
      }
    }

  }
  
  if (rank==size-1){
    #pragma acc parallel loop present( matrix, matrix_new )   
    for( i = 0; i < N_loc-1; ++i ){
      matrix[ i * ( dimension + 2 ) ] = ( i + my_row ) * increment;
      matrix_new[ i * ( dimension + 2 ) ] = (i+my_row) * increment;
    }
    #pragma acc parallel loop collapse(2) present( matrix )   
    for( i = 0; i < N_loc; ++i ){
      for( j = 1; j < dimension+1; ++j ){
      matrix[ ( i * ( dimension + 2 ) ) + j ] = 0.5;
      }
    }
    #pragma acc parallel loop present( matrix, matrix_new )   
    for (j = 0; j<dimension+2; ++j){
      matrix[ (N_loc)*(dimension+2) - 1-j] = j*increment;
      matrix_new[ (N_loc) * (dimension+2) -1- j ] = j*increment;;
    }


  }

}


void evolve( double* restrict matrix, double* restrict matrix_new, const int N_loc, const size_t dimension ){
  
  #pragma acc parallel loop collapse(2) present( matrix, matrix_new )  
  for(size_t i = 1 ; i < N_loc-1; ++i ){
    for(size_t j = 1; j < dimension+1; ++j ){
      matrix_new[ ( i * ( dimension + 2 ) ) + j ] = ( 0.25 ) * 
         (matrix[ ( ( i - 1 ) * ( dimension + 2 ) ) + j ] + 
          matrix[ ( i * ( dimension + 2 ) ) + ( j + 1 ) ] + 	  
          matrix[ ( ( i + 1 ) * ( dimension + 2 ) ) + j ] + 
          matrix[ ( i * ( dimension + 2 ) ) + ( j - 1 ) ]); 
    }
  }
}

void mpi_io(int rank, int size, double* local_matrix, int N_loc, int LDA, char* fname_snap, MPI_File file){
      

    char filename[256];
    sprintf(filename, "%s.bin", fname_snap);

    MPI_Offset offset = ((N_loc-2) * rank + (LDA-2)%size*(rank>=(LDA-2)%size)) * LDA * sizeof(double);
    // printf("rank %d, N_loc %d, offset %d\n", rank,N_loc, offset);

    MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);

    // Set the file pointer to the correct position
    MPI_File_seek(file, offset, MPI_SEEK_SET);

    // Set the file view
    MPI_File_set_view(file, offset, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);

    // Write the local matrix block to the file using MPI I/O
    #pragma acc host_data use_device( local_matrix )            
    MPI_File_write(file, &local_matrix[LDA], (N_loc-2)*LDA, MPI_DOUBLE, MPI_STATUS_IGNORE);

    // Close the file
    MPI_File_close(&file);


}


void update_boundaries(int rank, int size, double* local_matrix, int my_row, int N){

    MPI_Request send_request;
    MPI_Status recv_status;

    int tag1=0,tag2=1;

    int prev = rank == 0? MPI_PROC_NULL : rank-1;
    int next = rank == size-1? MPI_PROC_NULL : rank+1;

            // sending last row to rank +1
            #pragma acc host_data use_device( local_matrix )
            MPI_Isend(&local_matrix[(my_row-2)*N],N, MPI_DOUBLE, next, tag1, MPI_COMM_WORLD, &send_request);
            // sending first row to rank -1
            #pragma acc host_data use_device( local_matrix )
            MPI_Isend(&local_matrix[N],N, MPI_DOUBLE, prev, tag2, MPI_COMM_WORLD, &send_request);

            // reveiving first row from rank -1
            #pragma acc host_data use_device( local_matrix )
            MPI_Recv(local_matrix,N ,MPI_DOUBLE,prev,tag1,MPI_COMM_WORLD,&recv_status);
            // receiving last row from rank +1
            #pragma acc host_data use_device( local_matrix )            
            MPI_Recv(&local_matrix[(my_row-1)*N],N ,MPI_DOUBLE,next,tag2,MPI_COMM_WORLD,&recv_status);

}

void print_loc( double * mat, int loc_size, int LDA ){
  
  int i, j;
  for( i = 0; i < loc_size; i++ ){
    for( j = 0; j < LDA; j++ ){
      fprintf( stdout, "%.3g ", mat[ j + ( i * LDA ) ] );
    }
    fprintf( stdout, "\n" );
    
  }
}

void distributed_print(int rank, int size,int N_loc,int LDA, double * matrix){
  
  MPI_Barrier(MPI_COMM_WORLD);

  for (int i = 0; i < size; i++){
    if (rank == i){
      printf("rank %d, N_loc %d\n", rank, N_loc);
      print_loc(matrix, N_loc, LDA);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);

}

void write_csv(int size, int N,int iterations, double global, double computational_global, double communication_global,double init_global, double t_io,char* datacsv){
    
    /*

        1. size (#processi)
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
  
  fprintf(csv_file, "%d,%d,%d,%f,%f,%f,%f,%f\n", size, N, iterations, global,  computational_global,  communication_global, init_global, t_io );
  
  fclose(csv_file);

  return;

}