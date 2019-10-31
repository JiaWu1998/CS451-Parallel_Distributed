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
  int numproc, procRank;

  /* MPI initalization */
  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD,&numproc);
  MPI_Comm_rank(MPI_COMM_WORLD,&procRank);

  int local_N = atoi(argv[1]);

  if (procRank == 0){
    /* Process program parameters */
    parameters(argc, argv);

    /* Initialize A and B */
    initialize_inputs();

    /* Print input matrices */
    print_inputs();

  }
  MPI_Barrier(MPI_COMM_WORLD);
  /* Gaussian Elimination */
  int norm, row, col;
  for (norm = 0; norm < local_N - 1; ++norm){
    int local_num_rows = (int) (ceil((float) (local_N - norm) / (float) numproc)) + 1;
    int local_index;
    float multiplier;
    float local_A[local_num_rows][local_N];
    float local_B[local_num_rows];
    float local_whole_linear_A[local_N*local_N];

    if (procRank == 0){   
      //get a copy of A local to processor 0
      for (row=0; row<local_N; ++row){
        for (col=0; col<local_N; ++col){
          local_whole_linear_A[row * local_N + col] = A[row][col];
        }
      }

      //get a copy of the norm and put it into the local_A to Bcast it to other local_A
      for (col=0; col<local_N; ++col){
        local_A[0][col] = A[norm][col];
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
        MPI_Scatter(&local_whole_linear_A[local_N * (norm +row)], local_N, MPI_FLOAT, &local_A[local_index], local_N, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Scatter((void *) &B[row], 1, MPI_FLOAT, &local_B[local_index], 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
        local_index++; 
      }

    // zeroing
    for (row = 1; row < local_num_rows; ++row){
      multiplier = local_A[row][norm] / local_A[0][norm];
      for (col = norm; col < local_N; ++col){
        local_A[row][col] -= local_A[0][col] * multiplier;
      }
      local_B[row] -= local_B[0] * multiplier;
    }

    // every processor needs to wait until all processors are complete with calculating
    MPI_Barrier(MPI_COMM_WORLD); 

    // Need to gather
    local_index = 1;
    for (row = 1; row < local_N; row += numproc){
      MPI_Gather();
      local_index++;
    }


    if(procRank == 0){
      printf("local A from proc %i----------------------\n", procRank);
      for (row=0; row< local_num_rows; ++row){
        for (col=0; col< local_N; ++col){
          printf("%f\t",local_A[row][col]);
        }
        printf("\n");
      }
      printf("Local B----------------\n");
      for (row=0; row< local_num_rows; ++row){
          printf("%f\t",local_B[row]);
        printf("\n");
      }
      printf("----------------\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(procRank == 1){
      printf("local A from proc %i----------------------\n", procRank);
      for (row=0; row< local_num_rows; ++row){
        for (col=0; col< local_N; ++col){
          printf("%f\t",local_A[row][col]);
        }
        printf("\n");
      }
      printf("Local B----------------\n");
      for (row=0; row< local_num_rows; ++row){
          printf("%f\t",local_B[row]);
        printf("\n");
      }
      printf("----------------\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(procRank == 2){
      printf("local A from proc %i----------------------\n", procRank);
      for (row=0; row< local_num_rows; ++row){
        for (col=0; col< local_N; ++col){
          printf("%f\t",local_A[row][col]);
        }
        printf("\n");
      }
      printf("Local B----------------\n");
      for (row=0; row< local_num_rows; ++row){
          printf("%f\t",local_B[row]);
        printf("\n");
      }
      printf("----------------\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(procRank == 3){
      printf("local A from proc %i----------------------\n", procRank);
      for (row=0; row< local_num_rows; ++row){
        for (col=0; col< local_N; ++col){
          printf("%f\t",local_A[row][col]);
        }
        printf("\n");
      }
      printf("Local B----------------\n");
      for (row=0; row< local_num_rows; ++row){
          printf("%f\t",local_B[row]);
        printf("\n");
      }
      printf("----------------\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);


    norm = local_N;    
    // MPI_Barrier(MPI_COMM_WORLD);
  }
  

  // if (procRank == 0){
  //   back_substitution();

  //   /* Display output */
  //   print_X();
  // }

  MPI_Finalize();
  return 0;
}
