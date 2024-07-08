#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

/*** function declarations ***/

// save matrix to file
void mpi_io(int rank, int size, double* local_matrix, int N_loc, int LDA, char* fname_snap, MPI_File file);

// initialize matrix
void jacobi_init(double* restrict matrix, double* restrict matrix_new, int N_loc, size_t dimension, int rank, int size);
void jacobi_init_serial(double *matrix, double *matrix_new, int N_loc, int dimension);
// evolve Jacobi
void evolve( double* restrict matrix, double* restrict matrix_new, int N_loc, size_t dimension);
// void evolve_update( double* restrict matrix, double* restrict matrix_new, int N_loc, size_t dimension, int rank, int size, int prev, int next, MPI_Win win_u, MPI_Win win_d);

// return the elapsed time
double seconds( void );

// print matrix
void print_loc( double * mat, int loc_size , int LDA);
void distributed_print(int rank, int size,int N_loc, int LDA,  double * matrix);

// processes update of shared boundaries
void update_boudaries(int rank, int size, double* local_matrix, int my_row, int N);
void update_boun_oneside_lock(int rank,int size,double* matrix_new,int N_loc,int dimension,MPI_Win win_u, MPI_Win win_d);
void update_boun_oneside_fence(int rank,int size,double* matrix_new,int N_loc,int dimension,MPI_Win win_u,MPI_Win win_d);


void write_csv(int size, int N, int iterations, double global, double computational_global, double communication_global, double t_window_max, double init_global, double t_io, char* datacsv);


/*** end function declaration ***/


int main(int argc, char* argv[]){

  // MPI init
  // MPI_Init(&argc, &argv);

  int provided;
  MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);
  
  int rank, size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // timing variables
  double t_comm = 0.0;

  // indexes for loops
  size_t it;
  
  // initialize matrix
  double *matrix, *matrix_new, *tmp_matrix;

  size_t dimension = 0, iterations = 0, row_peek, col_peek;
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


// Initialization of the matrix and the new matrix
  double t1 = MPI_Wtime(); // init data

  int N_loc = 2 + (dimension)/size + 1*(rank < (dimension)%size);

  byte_dimension = sizeof(double) * ( N_loc ) * ( dimension + 2 );
  matrix = ( double* )malloc( byte_dimension );
  matrix_new = ( double* )malloc( byte_dimension );

  jacobi_init( matrix, matrix_new, N_loc, dimension, rank, size);
  
  double t2 = MPI_Wtime(); // init data


  MPI_Win win_u_n, win_d_n;
  MPI_Win win_u, win_d;

  MPI_Info info;
  MPI_Info_create(&info);
  MPI_Info_set(info, "same_size", "true");

  double t_window_max=0;

  if(size > 1){
    double t1_1 = MPI_Wtime();

    MPI_Win_create(&matrix_new[dimension+2], (MPI_Aint)(dimension+2) * sizeof(double), sizeof(double), info, MPI_COMM_WORLD, &win_u_n);
    MPI_Win_create(&matrix_new[(N_loc-2)*(dimension+2)], (MPI_Aint)(dimension+2) * sizeof(double), sizeof(double), info, MPI_COMM_WORLD, &win_d_n);

    MPI_Win_create(&matrix[dimension+2], (MPI_Aint)(dimension+2) * sizeof(double), sizeof(double), info, MPI_COMM_WORLD, &win_u);
    MPI_Win_create(&matrix[(N_loc-2)*(dimension+2)], (MPI_Aint)(dimension+2) * sizeof(double), sizeof(double), info, MPI_COMM_WORLD, &win_d);

    double t1_2 = MPI_Wtime();
    double t_window = t1_2 - t1_1;

    MPI_Reduce(&t_window,&t_window_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  }
  double t3_p = MPI_Wtime(); 

  // start algorithm
  for( it = 0; it < iterations; ++it ){
    
    evolve( matrix, matrix_new,N_loc, dimension);

    double t3 = MPI_Wtime(); 
    if (size >1 & it%2==0){
      update_boun_oneside_fence(rank, size, matrix_new, N_loc, dimension, win_u_n, win_d_n);
    }else if(size>1){  
      update_boun_oneside_fence(rank, size, matrix_new, N_loc, dimension, win_u,win_d);
    }
    double t4 = MPI_Wtime(); 

    // swap the pointers
    tmp_matrix = matrix;
    matrix = matrix_new;
    matrix_new = tmp_matrix;
    
    t_comm += t4-t3;

  }
  
  double t5 = MPI_Wtime();
  mpi_io(rank, size, matrix, N_loc, dimension+2, "snap_oneside_cpu", MPI_FILE_NULL); 
  double t6 = MPI_Wtime(); // init data

  double t_compute = (t5-t3_p) - t_comm;
  double t_io = t6-t5;
  double t_init = t2-t1;
  double t_compute_max, t_comm_max, t_init_max, t_io_max;
  
  MPI_Reduce(&t_compute,&t_compute_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Reduce(&t_init,&t_init_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Reduce(&t_comm,&t_comm_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Reduce(&t_io,&t_io_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);


  double global = t_compute_max + t_init_max + t_comm_max + t_io_max;


  // if (rank==0){
  // printf("Elapsed time: %f s\n", t_end - t_start);
  // }


  char* datacsv;
  datacsv = (char*)malloc(100);
  sprintf(datacsv, "%s","./data/oneside_data.csv");

  if (rank == 0){
    write_csv(size, dimension, iterations, global, t_compute_max,t_comm_max, t_window_max,t_init_max, t_io,datacsv);
    // printf("t_compute %f, t_comm %f, t_init %f, t_io %f\n", t_compute_max, t_comm_max, t_init_max, t_io_max);
  }

  // MPI_Win_free(&win_u);
  // MPI_Win_free(&win_d);
  // MPI_Win_free(&win_u_n);
  // MPI_Win_free(&win_d_n);
  // free( matrix );
  // free( matrix_new );

  MPI_Finalize();

  return 0;
}




void evolve( double* restrict matrix, double* restrict matrix_new, int N_loc, size_t dimension ){
  
  size_t i, j;

  //This will be a row dominant program.
  #pragma omp parallel for
  for( i = 1 ; i < N_loc-1; ++i ){
    for( j = 1; j < dimension+1; ++j ){
      matrix_new[ ( i * ( dimension + 2 ) ) + j ] = ( 0.25 ) * 
	( matrix[ ( ( i - 1 ) * ( dimension + 2 ) ) + j ] + 
	  matrix[ ( i * ( dimension + 2 ) ) + ( j + 1 ) ] + 	  
	  matrix[ ( ( i + 1 ) * ( dimension + 2 ) ) + j ] + 
	  matrix[ ( i * ( dimension + 2 ) ) + ( j - 1 ) ] ); 
    }
  }
}

void jacobi_init(double* restrict matrix, double* restrict matrix_new, int N_loc, size_t dimension, int rank, int size){
  
  if(size==1){
    jacobi_init_serial(matrix, matrix_new, N_loc, dimension);
    return;
  }

  double increment = 100.0 / ( dimension + 1 );
  
  // qui devo contare quale sia la riga di partenza di ogni processo rispetto alla matrice completa
  int my_row = rank*(N_loc-2) + (dimension%size)*(rank >= (dimension)%size);

  if (rank==0){
      // first row
      #pragma omp parallel for
      for (size_t j = 0; j<dimension+2; ++j){
        matrix[ j ] = 0.0;
        matrix_new[ j ] = 0.0;
      }

    // last column
    #pragma omp parallel for
    for(size_t i = 0; i < N_loc; ++i ){
      matrix[ i * ( dimension + 2 ) + dimension+1 ] = 0.0;
      matrix_new[ i * ( dimension + 2 ) + dimension+1 ] = 0.0;
    }
    
    #pragma omp parallel for collapse(2)
    for(size_t i = 1; i < N_loc; ++i ){
      for(size_t j = 1; j < dimension+1; ++j ){
        matrix[ ( i * ( dimension + 2 ) ) + j ] = 0.5;
      }
    }
    
    #pragma omp parallel for
    for(size_t i = 1; i < N_loc; ++i ){
      matrix[ i * ( dimension + 2 ) ] = ( i + my_row ) * increment;
      matrix_new[ i * ( dimension + 2 ) ] = (i+my_row) * increment;
    }
  
  }

  if (rank!=0 && rank!=size-1){
      // first row
      #pragma omp parallel for
      for (size_t j = 0; j<dimension+2; ++j){
        matrix[ j ] = 0.0;
        matrix_new[ j ] = 0.0;
      }

    // last column
    #pragma omp parallel for
    for(size_t i = 0; i < N_loc; ++i ){
      matrix[ i * ( dimension + 2 ) + dimension+1 ] = 0.0;
      matrix_new[ i * ( dimension + 2 ) + dimension+1 ] = 0.0;
    }

    #pragma omp parallel for
    for(size_t i = 0; i < N_loc; ++i ){
      matrix[ i * ( dimension + 2 ) ] = ( i + my_row ) * increment;
      matrix_new[ i * ( dimension + 2 ) ] = (i+my_row ) * increment;
    }

      #pragma omp parallel for collapse(2)
      for(size_t i = 0; i < N_loc; ++i ){
        for(size_t j = 1; j < dimension+1; ++j ){
          matrix[ ( i * ( dimension + 2 ) ) + j ] = 0.5;
        }
      }
  }

  if (rank==size-1){
    // first row
    #pragma omp parallel for
    for (size_t j = 0; j<dimension+2; ++j){
      matrix[ j ] = 0.0;
      matrix_new[ j ] = 0.0;
    }

    // last column
    #pragma omp parallel for
    for(size_t i = 0; i < N_loc; ++i ){
      matrix[ i * ( dimension + 2 ) + dimension+1 ] = 0.0;
      matrix_new[ i * ( dimension + 2 ) + dimension+1 ] = 0.0;
    }
    #pragma omp parallel for 
    for(size_t i = 0; i < N_loc-1; ++i ){
      matrix[ i * ( dimension + 2 ) ] = ( i + my_row ) * increment;
      matrix_new[ i * ( dimension + 2 ) ] = (i+my_row) * increment;
    }

    #pragma omp parallel for
    for (size_t j = 0; j<dimension+1; ++j){
      matrix[ (N_loc-1)*(dimension+2) + j] = 100-j*increment;
      matrix_new[ (N_loc-1) * (dimension+2) + j ] = 100-j*increment;;
    }

      #pragma omp parallel for collapse(2)
      for(size_t i = 0; i < N_loc-1; ++i ){
        for(size_t j = 1; j < dimension+1; ++j ){
          matrix[ ( i * ( dimension + 2 ) ) + j ] = 0.5;
        }
      }
  }

}

void jacobi_init_serial(double *matrix, double *matrix_new, int N_loc, int dimension){


    int rank = 0, size = 1;
    double increment = 100.0 / ( dimension + 1 );
    int my_row = rank*(N_loc-2) + (dimension%size)*(rank >= (dimension)%size);

  #pragma omp parallel
  {
    #pragma omp for nowait collapse(2)
    for(int i = 1; i < N_loc-1; ++i ){
      for(int j = 1; j < dimension+1; ++j ){
        matrix[ ( i * ( dimension + 2 ) ) + j ] = 0.5;
      }
    }

    // last row
    #pragma omp for nowait
    for (size_t j = 0; j<dimension+1; ++j){
      matrix[ (N_loc-1)*(dimension+2) + j] = 100-j*increment;
      matrix_new[ (N_loc-1) * (dimension+2) + j ] = 100-j*increment;;
    }

    // first column
    #pragma omp for nowait
    for(size_t i = 1; i < N_loc; ++i ){
      matrix[ i * ( dimension + 2 ) ] = ( i + my_row ) * increment;
      matrix_new[ i * ( dimension + 2 ) ] = (i+my_row) * increment;
    }

    // SE NON USO MEMSET DEVO INIZIALIZZARE ANCHE I BORDI TOP E RIGHT
    // first row
    #pragma omp for nowait
    for (size_t j = 0; j<dimension+2; ++j){
      matrix[ j ] = 0.0;
      matrix_new[ j ] = 0.0;
    }

    // last column
    #pragma omp for nowait
    for(size_t i = 0; i < N_loc; ++i ){
      matrix[ i * ( dimension + 2 ) + dimension+1 ] = 0.0;
      matrix_new[ i * ( dimension + 2 ) + dimension+1 ] = 0.0;
    }

  }
}


void mpi_io(int rank, int size, double* local_matrix, int N_loc, int LDA, char* fname_snap, MPI_File file){
      

    char filename[256];
    sprintf(filename, "%s.bin", fname_snap);


    MPI_Offset offset = ((N_loc-2) * rank + (LDA-2)%size*(rank>=(LDA-2)%size)) * LDA * sizeof(double);

    // printf("rank %d, N_loc %d, offset %lld\n", rank,N_loc, offset);


    MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);

    // Set the file pointer to the correct position
    MPI_File_seek(file, offset, MPI_SEEK_SET);

    // Set the file view
    MPI_File_set_view(file, offset, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);

    // Write the local matrix block to the file using MPI I/O
    MPI_File_write(file, &local_matrix[LDA], (N_loc-2)*LDA, MPI_DOUBLE, MPI_STATUS_IGNORE);

    // Close the file
    MPI_File_close(&file);


}

double seconds(){

    struct timeval tmp;
    double sec;
    gettimeofday( &tmp, (struct timezone *)0 );
    sec = tmp.tv_sec + ((double)tmp.tv_usec)/1000000.0;
    return sec;
}

void update_boudaries(int rank, int size, double* local_matrix, int my_row, int N){

    MPI_Request send_request;
    MPI_Status recv_status;

    int tag1=0,tag2=1;

    int prev = rank == 0? MPI_PROC_NULL : rank-1;
    int next = rank == size-1? MPI_PROC_NULL : rank+1;

            // sending last row to rank +1
            MPI_Isend(&local_matrix[(my_row-2)*N],N, MPI_DOUBLE, next, tag1, MPI_COMM_WORLD, &send_request);
            // sending first row to rank -1
            MPI_Isend(&local_matrix[N],N, MPI_DOUBLE, prev, tag2, MPI_COMM_WORLD, &send_request);

            // reveiving first row from rank -1
            MPI_Recv(local_matrix,N ,MPI_DOUBLE,prev,tag1,MPI_COMM_WORLD,&recv_status);
            // receiving last row from rank +1
            MPI_Recv(&local_matrix[(my_row-1)*N],N ,MPI_DOUBLE,next,tag2,MPI_COMM_WORLD,&recv_status);

}

void update_boun_oneside_lock(int rank,int size,double* matrix_new,int N_loc,int dimension,MPI_Win win_u,MPI_Win win_d){

  MPI_Barrier(MPI_COMM_WORLD);
  int prev = rank == 0? MPI_PROC_NULL : rank-1;
  int next = rank == size-1? MPI_PROC_NULL : rank+1;

  if (rank!=0){
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE,prev,MPI_MODE_NOCHECK, win_d);     
    MPI_Get(matrix_new, dimension+2, MPI_DOUBLE, prev, 0, dimension+2, MPI_DOUBLE, win_d);
    MPI_Win_unlock(prev, win_d);
  }
  if(rank!=size-1){
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE,next,MPI_MODE_NOCHECK, win_u);
    MPI_Get(&matrix_new[(N_loc-1)*(dimension+2)], dimension+2, MPI_DOUBLE, next, 0, dimension+2, MPI_DOUBLE, win_u);
    MPI_Win_unlock(next, win_u);
  }

}

void update_boun_oneside_fence(int rank,int size,double* matrix_new,int N_loc,int dimension,MPI_Win win_u,MPI_Win win_d)
{
  int prev = rank == 0? MPI_PROC_NULL : rank-1;
  int next = rank == size-1? MPI_PROC_NULL : rank+1;

  MPI_Win_fence(MPI_MODE_NOPUT, win_d);     
  MPI_Win_fence(MPI_MODE_NOPUT, win_u);
  MPI_Get(matrix_new, dimension+2, MPI_DOUBLE, prev, 0, dimension+2, MPI_DOUBLE, win_d);
  MPI_Get(&matrix_new[(N_loc-1)*(dimension+2)], dimension+2, MPI_DOUBLE, next, 0, dimension+2, MPI_DOUBLE, win_u);
  MPI_Win_fence(MPI_MODE_NOPUT, win_u);
  MPI_Win_fence(MPI_MODE_NOPUT, win_d); 

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

void write_csv(int size, int N, int iterations, double global, double computational_global, double communication_global,double t_window_max, double init_global, double t_io,char* datacsv){

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
  
  fprintf(csv_file, "%d,%d,%d,%f,%f,%f,%f,%f,%f\n", size, N, iterations, global,  computational_global,  communication_global,t_window_max, init_global, t_io );
  
  fclose(csv_file);

  return;

}


// void evolve_update( double* restrict matrix, double* restrict matrix_new, int N_loc, size_t dimension, int rank, int size, int prev, int next, MPI_Win win_u, MPI_Win win_d){

//     #pragma omp parallel
//     {
//         #pragma omp single nowait
//         {
            
//             #pragma omp task
//             {
//                 #pragma omp taskloop collapse(2) 
//                   for( size_t i = 1 ; i < 2; ++i ){
//                   for( size_t j = 1; j < dimension+1; ++j ){
//                     matrix_new[ ( i * ( dimension + 2 ) ) + j ] = ( 0.25 ) * 
//                 ( matrix[ ( ( i - 1 ) * ( dimension + 2 ) ) + j ] + 
//                   matrix[ ( i * ( dimension + 2 ) ) + ( j + 1 ) ] + 	  
//                   matrix[ ( ( i + 1 ) * ( dimension + 2 ) ) + j ] + 
//                   matrix[ ( i * ( dimension + 2 ) ) + ( j - 1 ) ] );
//                   }}

//               #pragma omp taskloop collapse(2)
//                   for( size_t i = (N_loc-2) ; i < (N_loc-1); ++i ){
//                   for( size_t j = 1; j < dimension+1; ++j ){
//                     matrix_new[ ( i * ( dimension + 2 ) ) + j ] = ( 0.25 ) * 
//                 ( matrix[ ( ( i - 1 ) * ( dimension + 2 ) ) + j ] + 
//                   matrix[ ( i * ( dimension + 2 ) ) + ( j + 1 ) ] + 	  
//                   matrix[ ( ( i + 1 ) * ( dimension + 2 ) ) + j ] + 
//                   matrix[ ( i * ( dimension + 2 ) ) + ( j - 1 ) ] );
//                   }}

//             }

//               #pragma omp taskwait
//               {
//                   MPI_Win_fence(0, win_d);     
//                   MPI_Win_fence(0, win_u);
//                   MPI_Get(matrix_new, dimension+2, MPI_DOUBLE, prev, 0, dimension+2, MPI_DOUBLE, win_d);
//                   MPI_Get(&matrix_new[(N_loc-1)*(dimension+2)], dimension+2, MPI_DOUBLE, next, 0, dimension+2, MPI_DOUBLE, win_u);
//                   MPI_Win_fence(0, win_u);
//                   MPI_Win_fence(0, win_d);
//                   // update_boudaries(rank, size, matrix_new, N_loc, dimension+2);
//               }

//         }
        
//         #pragma omp single nowait
//         {
//             #pragma omp task
//             {
//                 #pragma omp taskloop
//                 for( size_t i = 2 ; i < (N_loc-2); ++i ){
//                 for( size_t j = 1; j < dimension+1; ++j ){
//                   matrix_new[ ( i * ( dimension + 2 ) ) + j ] = ( 0.25 ) * 
//               ( matrix[ ( ( i - 1 ) * ( dimension + 2 ) ) + j ] + 
//                 matrix[ ( i * ( dimension + 2 ) ) + ( j + 1 ) ] + 	  
//                 matrix[ ( ( i + 1 ) * ( dimension + 2 ) ) + j ] + 
//                 matrix[ ( i * ( dimension + 2 ) ) + ( j - 1 ) ] );
//                 }}
//             }
            

//         }
//     }


// }
