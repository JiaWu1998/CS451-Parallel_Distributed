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

void zeroing(int row, int col, int norm, int local_N, int numproc){
    float multiplier;

    if (numproc == -1){
        // from other processors: only need to change one row
        multiplier = A[row][norm] / A[norm][norm];
        for (col = norm; col < local_N; col++){
            A[row][col] -= A[norm][col] * multiplier;
        }
        B[row] -= B[norm] * multiplier;
    }else{
        // from processor 0: need to skip numproc and change all other rows
        for (row=norm+1 ; row<local_N ; row += numproc){
            multiplier = A[row][norm] / A[norm][norm];
            for (col = norm; col < local_N; col++){
                A[row][col] -= A[norm][col] * multiplier;
            }
            B[row] -= B[norm] * multiplier;
        }
    }
}

int main(int argc,char *argv[]){
    double t1, t2; 
    int numproc,procRank, local_N;

	MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numproc);
	MPI_Comm_rank(MPI_COMM_WORLD,&procRank);


	if(procRank==0){
		/* Process program parameters */
        parameters(argc, argv);

        /* Initialize A and B */
        initialize_inputs();

        // /* Print input matrices */
        print_inputs();
        t1 = MPI_Wtime();
 							
	}

    local_N = atoi(argv[1]);
	
	MPI_Request req;
	MPI_Status stat;
	int procid,norm,row,col;

    //need to wait for processor 0 to complete initialization of A and read the user input
	MPI_Barrier(MPI_COMM_WORLD);

	for (norm=0;norm<local_N-1;norm++){	
		// broadcast the norm to all local A
		MPI_Bcast((void*) &A[norm][0],local_N, MPI_FLOAT, 0, MPI_COMM_WORLD);
		MPI_Bcast((void*) &B[norm],1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	
		if (procRank==0) {
			for (procid=1;procid<numproc;procid++){
		  		for (row=norm+1+procid;row<local_N;row+=numproc){
				//non block send whole A, B matrix to other processors 
				   MPI_Isend((void*) &A[row], local_N, MPI_FLOAT, procid, 0, MPI_COMM_WORLD, &req);
                   //note to self: must do MPI_Wait to only wait for after the MPI send request is 
                   //complete. Then it moves on quick.
				   MPI_Wait(&req, &stat);
				   MPI_Isend((void*) &B[row], 1, MPI_FLOAT, procid, 0, MPI_COMM_WORLD, &req);
				   MPI_Wait(&req, &stat);
		  		}
			}

			// zeroing for processor 0
            zeroing(row, col, norm, local_N, numproc);
			// for (row=norm+1 ; row<local_N ; row += numproc){
	  		// 	multiplier = A[row][norm] / A[norm][norm];
	  		// 	for (col = norm; col < local_N; col++){
	   		// 		A[row][col] -= A[norm][col] * multiplier;
	 		// 	}
	   		// 	B[row] -= B[norm] * multiplier;
			// }

			// Receiving back changed matrix from all other processors
			for (procid = 1; procid < numproc; procid++){
			  for (row = norm + 1 + procid; row < local_N; row += numproc){
			    MPI_Recv((void*) &A[row], local_N, MPI_FLOAT, procid, 1, MPI_COMM_WORLD, &stat);
			    MPI_Recv((void*) &B[row], 1, MPI_FLOAT, procid, 1, MPI_COMM_WORLD, &stat);
			  }
			}
		}


		if (procRank != 0){
			for (row = norm + 1 + procRank; row < local_N; row += numproc){
                //get norm from processor 0
				MPI_Recv((void*) &A[row], local_N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &stat);		
				MPI_Recv((void*) &B[row], 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &stat);

                //zeroing for other processors (exclude 0)
                zeroing(row, col, norm, local_N, -1);
				// multiplier = A[row][norm] / A[norm][norm];
				// for (col = norm; col < local_N; col++){
				//     A[row][col] -= A[norm][col] * multiplier;
				// }
				// B[row] -= B[norm] * multiplier;

                //send back the row to processor 0
				MPI_Isend((void*) &A[row], local_N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &req);						    
				MPI_Wait(&req, &stat);		
				MPI_Isend((void*) &B[row], 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &req);
				MPI_Wait(&req, &stat);
			}
		}

        //wait before starting next norm
		 MPI_Barrier(MPI_COMM_WORLD);
	}
	
	if(procRank==0){
		back_substitution();

        print_X();
		t2 = MPI_Wtime();

        printf("Elapsed time = %1.5f ms.\n", (t2-t1)*1000);
        fflush(stdout);
	}
	MPI_Finalize(); //Finalizing the MPI
  	return 0;
}

