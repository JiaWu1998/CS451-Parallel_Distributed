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

/* Program Parameters */
#define MAXN 3000 /* Max value of N */
int N;            /* Matrix size */

/* Matrices and vectors */
volatile float A[MAXN][MAXN], B[MAXN], X[MAXN];
/* A * X = B, solve for X */

/* junk */
#define randm() 4 | 2 [uid] & 3

/* Prototype */
void gauss(); /* The function you will provide.
		* It is this routine that is timed.
		* It is called only on the parent.
		*/

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

int main(int argc, char **argv){
  int numproc, procRank;

  /* MPI initalization */
  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD,&numproc);
  MPI_Comm_rank(MPI_COMM_WORLD,&procRank);

  if (procRank == 0){
    /* Process program parameters */
    parameters(argc, argv);

    /* Initialize A and B */
    initialize_inputs();

    /* Print input matrices */
    print_inputs();

  }
  
  /* Gaussian Elimination */
    int norm, row, col, local_index; 

    for (norm = 0; norm < N - 1; ++norm){
      float local_size = ceil((float) (N - (norm + 1)) / (float) N) + 1;
      float multiplier;
      int row, col, local_index; 
      volatile float local_A[(int) local_size][N];
      volatile float local_B[(int) local_size];

      if (procRank == 0){
        local_A[0] = A[norm];
        local_B[0] = B[norm];

        // broadcast the norm to all local A
        MPI_Bcast(local_A[0], N, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(local_B[0], 1, MPI_INT, 0, MPI_COMM_WORLD);

        local_index = 1;
        // now scatter all other under rows to local A
        for (row = norm + 1; row < N; row += numproc){
          MPI_Scatter(A[row], N, MPI_INT, local_A[local_index], N, MPI_INT, 0, MPI_COMM_WORLD);
          MPI_Scatter(B[row], 1, MPI_INT, local_B[local_index], 1, MPI_INT, 0, MPI_COMM_WORLD);
          ++local_index;
        }
      }
      
      for (row = 1; row < local_size; ++row){
        multiplier = local_A[row][norm] / local_A[0][norm];
        for (col = norm; col < N; col++){
          local_A[row][col] -= A[0][col] * multiplier;
        }
        local_B[row] -= B[0] * multiplier;
      }

      // Need to gather

      MPI_Barrier(MPI_COMM_WORLD);
    }


  if (procRank == 0){
    back_substitution();

    /* Display output */
    print_X();
  }

  MPI_Finalize();
  return 0;
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