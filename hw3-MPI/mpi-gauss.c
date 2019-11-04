/* This is a program that does gaussian elimation over distributed machines using MPI */

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>
#include <string.h>

/* Program Parameters */
#define MAXN 3000 /* Max value of N */
int N;            /* Matrix size */

/* Matrices and vectors */
volatile float A[MAXN][MAXN], B[MAXN], X[MAXN];

/* junk */
#define randm() 4 | 2 [uid] & 3

/* returns a seed for srand based on the time */
unsigned int time_seed()
{
  struct timeval t;
  struct timezone tzdummy;

  gettimeofday(&t, &tzdummy);
  return (unsigned int)(t.tv_usec);
}

/* Set the program parameters from the command-line arguments */
void parameters(int argc, char **argv)
{
  int seed = 0; /* Random seed */
  char uid[32]; /*User name */

  /* Read command-line arguments */
  srand(time_seed()); /* Randomize */

  if (argc == 3)
  {
    seed = atoi(argv[2]);
    srand(seed);
    printf("Random seed = %i\n", seed);
  }
  if (argc >= 2)
  {
    N = atoi(argv[1]);
    if (N < 1 || N > MAXN)
    {
      printf("N = %i is out of range.\n", N);
      exit(0);
    }
  }
  else
  {
    printf("Usage: %s <matrix_dimension> [random seed]\n",
           argv[0]);
    exit(0);
  }

  /* Print parameters */
  printf("\nMatrix dimension N = %i.\n", N);
}

/* Initialize A and B (and X to 0.0s) */
void initialize_inputs()
{
  int row, col;

  printf("\nInitializing...\n");
  for (col = 0; col < N; ++col)
  {
    for (row = 0; row < N; ++row)
    {
      A[row][col] = (float)rand() / 32768.0;
    }
    B[col] = (float)rand() / 32768.0;
    X[col] = 0.0;
  }
}

/* Print input matrices */
void print_inputs()
{
  int row, col;

  if (N < 10)
  {
    printf("\nA =\n\t");
    for (row = 0; row < N; ++row)
    {
      for (col = 0; col < N; ++col)
      {
        printf("%5.2f%s", A[row][col], (col < N - 1) ? ", " : ";\n\t");
      }
    }
    printf("\nB = [");
    for (col = 0; col < N; ++col)
    {
      printf("%5.2f%s", B[col], (col < N - 1) ? "; " : "]\n");
    }
  }
}

void print_X()
{
  int row;

  if (N < 100)
  {
    printf("\nX = [");
    for (row = 0; row < N; ++row)
    {
      printf("%5.2f%s", X[row], (row < N - 1) ? "; " : "]\n");
    }
  }
}

void back_substitution(){
  int norm, row, col;

  /* Back substitution */
  for (row = N - 1; row >= 0; --row){
    X[row] = B[row];
    for (col = N - 1; col > row; --col)
    {
      X[row] -= A[row][col] * X[col];
    }
    X[row] /= A[row][row];
  }
}

int main(int argc, char **argv){
  double t1, t2; 
  int numproc, procRank;

  /* MPI initalization */
  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD,&numproc);
  MPI_Comm_rank(MPI_COMM_WORLD,&procRank);

  //time measuring
  t1 = MPI_Wtime();

  int local_N = atoi(argv[1]);

  if (procRank == 0){
    /* Process program parameters */
    parameters(argc, argv);

    /* Initialize A and B */
    initialize_inputs();

    /* Print input matrices */
    print_inputs();

  }

  /* Gaussian Elimination */
  int norm, row, col;
  for (norm = 0; norm < local_N - 1; norm++){
    int local_num_rows = (int) (ceil((float) (local_N - norm) / (float) numproc)) + 1;
    int local_index;
    float multiplier;
    float local_A[3000];
    float local_B[3000];
    float local_whole_linear_A[2000000];

    if (procRank == 0){   
      //get a copy of A local to processor 0
      for (row=0; row<local_N; ++row){
        for (col=0; col<local_N; ++col){
          local_whole_linear_A[row * local_N + col] = A[row][col];
        }
      }

      //get a copy of the norm and put it into the local_A to Bcast it to other local_A
      for (col=0; col<local_N; ++col){
        local_A[col] = A[norm][col];
      }

      //get a copy of the norm and put it into the local_B to Bcast it to other local_B
      local_B[0] = B[norm];
    }

      // broadcast the norm to all local A
      MPI_Bcast(&local_A[0], local_N, MPI_FLOAT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&local_B[0], 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

      // now scatter all other under rows to local A
      local_index = 1;
      for (row = 1; row < local_N; row += numproc){
        MPI_Scatter(&local_whole_linear_A[local_N * (norm +row)], local_N, MPI_FLOAT, &local_A[local_N*local_index], local_N, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Scatter((void *) &B[norm + row], 1, MPI_FLOAT, &local_B[local_index], 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
        local_index++; 
      }
    
    // zeroing
    for (row = 1; row < local_num_rows; ++row){
      multiplier = local_A[row * local_N + norm] / local_A[norm];
      for (col = norm; col < local_N; ++col){
        local_A[row * local_N + col] -= local_A[col] * multiplier;
      }
      local_B[row] -= local_B[0] * multiplier;
    }

    // Need to gather
    local_index = 1;
    for (row = 1; row < local_N; row += numproc){
      MPI_Gather(&local_A[local_N*local_index],local_N, MPI_FLOAT, &local_whole_linear_A[local_N * (norm +row)], local_N, MPI_FLOAT, 0, MPI_COMM_WORLD);
      MPI_Gather(&local_B[local_index],1, MPI_FLOAT, (void *) &B[norm +row], 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
      local_index++;
    }

    if(procRank == 0){
      // need to convert linear A back to 2d A 
      for (row=0; row< local_N; ++row){
        for (col=0; col< local_N; ++col){
          A[row][col] = local_whole_linear_A[row * local_N + col];
        }
      }
    }

    //wait before starting next norm
    MPI_Barrier(MPI_COMM_WORLD);
  }
  
  if (procRank == 0){
    back_substitution();

    /* Display output */
    print_X();
  }

  t2 = MPI_Wtime();

  printf("Elapsed time = %1.5f\n", (t2-t1)*1000);
  fflush(stdout);

  MPI_Finalize();
  return 0;
}
