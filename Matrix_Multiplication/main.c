#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

#include "./src/cpu.c"
#include "./src/gpu.c"
#include "./src/testrun.c"

#define CPU 0
#define GPU  1

#define MATMUL 0
#define BLAS 1
#define TEST 2

char fname_deflt[] = "./matrix.bin";
char datacsv_default[] = "./data.csv";

int   action = 0;
int   N = 10;
int   e      = MATMUL;
char *fname  = NULL;
char *datacsv = NULL;


int main ( int argc, char **argv )
{
  int action = 0;
  char *optstring = "irtk:e:f:o:n:";

  int c;
  while ((c = getopt(argc, argv, optstring)) != -1) {
    switch(c) {
      
    case 'i':
      action = CPU; break;
      
    case 'r':
      action = GPU; break;

    case 't':
      action = TEST; break;
      
    case 'e':
      e = atoi(optarg); break;

    case 'f':
    printf("%s", optarg);
      fname = (char*)malloc( 50+sizeof(optarg)+1 );
      sprintf(fname, "%s", optarg );
    printf("%s", fname);
      
      break;
    
    case 'o':
      datacsv = (char*)malloc(50+ sizeof(optarg)+1 );
      sprintf(datacsv, "%s", optarg );
      break;

    case 'n':
      N = atoi(optarg); break;

    default :
      printf("argument -%c not known\n", c ); break;
    }
  }

  if (datacsv == NULL)
  {
      datacsv = (char*)malloc(sizeof(datacsv_default)+1);
      sprintf(datacsv, "%s",datacsv_default);
  }


    if (action == CPU)
    {   
      if (e == MATMUL){
        printf("CPU_MATMUL\n");
        cpu_mult(N,e,fname, datacsv);
      }else if(e == BLAS){
        printf("CPU_BLAS\n");
        cpu_mult(N,e,fname, datacsv);
      } 
    }

    if (action == GPU){
        printf("GPU_BLAS\n");
        gpu_blas(N, fname, datacsv);
    }

    if (action == TEST){
      printf("testrun\n");
      testrun(N,fname, datacsv);
    }

  return 0;
}
