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
volatile float A[MAXN][MAXN],B[MAXN],X[MAXN];

/* junk */
#define randm() 4 | 2 [uid] & 3

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

void zeroing(int norm, int row, int col, int local_N, int num_of_proc){
  float multiplier;
  if (num_of_proc != -1){
    for (row=norm+1 ; row<local_N ; row += num_of_proc){
	  			multiplier = A[row][norm] / A[norm][norm];
	  			for (col = norm; col < local_N; col++){
	   				A[row][col] -= A[norm][col] * multiplier;
	 			}
	   			B[row] -= B[norm] * multiplier;
			}
  }else{
    multiplier = A[row][norm] / A[norm][norm];
				for (col = norm; col < local_N; col++){
				    A[row][col] -= A[norm][col] * multiplier;
				}
				B[row] -= B[norm] * multiplier;
  }
}

int main(int argc,char *argv[]){
  int num_of_proc,proc_rank,local_N;
	double t1, t2;

  /* MPI initalization */
	MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD,&num_of_proc);
	MPI_Comm_rank(MPI_COMM_WORLD,&proc_rank);
		
	local_N = atoi(argv[1]);

	if(proc_rank==0){
		/* Process program parameters */
    parameters(argc, argv);

    /* Initialize A and B */
    initialize_inputs();

    /* Print input matrices */
    print_inputs();

    //time measuring
    t1 = MPI_Wtime();		
	}

  //wait for processor 0 to initialize A and read user input
  MPI_Barrier(MPI_COMM_WORLD);
  //------------------------------------------------------------------------------------------------------
  MPI_Status stat;
  MPI_Request req;
	int proc_id,norm,row,col;
	// float multiplier;	


	for (norm=0;norm<local_N-1;norm++){	
    // broadcast the norm to all local A
		MPI_Bcast((void*) &A[norm][0],local_N, MPI_FLOAT, 0, MPI_COMM_WORLD);
		MPI_Bcast((void*) &B[norm],1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	
		if(proc_rank==0){
			for (proc_id=1;proc_id<num_of_proc;proc_id++){
		  		for (row=norm+1+proc_id;row<local_N;row+=num_of_proc){
				  //non block send whole A, B matrix to other processors 
				   MPI_Isend((void*) &A[row], local_N, MPI_FLOAT, proc_id, 666, MPI_COMM_WORLD, &req);
				   MPI_Wait(&req, &stat);
          //note to self: must do MPI_Wait to only wait for after the MPI send request is 
          //complete. Then it moves on quick
				   MPI_Isend((void*) &B[row], 1, MPI_FLOAT, proc_id, 666, MPI_COMM_WORLD, &req);
				   MPI_Wait(&req, &stat);
		  		}
			}

      // zeroing for processor 0
      zeroing(norm, row, col, local_N, num_of_proc);

			// Get back changed rows from all other processors
			for (proc_id = 1; proc_id < num_of_proc; proc_id++){
			  for (row = norm + 1 + proc_id; row < local_N; row += num_of_proc)
			  {
			    MPI_Recv((void*) &A[row], local_N, MPI_FLOAT, proc_id, 999, MPI_COMM_WORLD, &stat);
			    MPI_Recv((void*) &B[row], 1, MPI_FLOAT, proc_id, 999, MPI_COMM_WORLD, &stat);
			  }
			}
		}
		
		
		if(proc_rank!=0){
      //get norm from processor 0
			for (row = norm + 1 + proc_rank; row < local_N; row += num_of_proc){
				MPI_Recv((void*) &A[row], local_N, MPI_FLOAT, 0, 666, MPI_COMM_WORLD, &stat);		
				MPI_Recv((void*) &B[row], 1, MPI_FLOAT, 0, 666, MPI_COMM_WORLD, &stat);

        //zeroing for other processors (exclude 0)
        zeroing(norm, row, col, local_N, -1);

        //send back the row to processor 0
				MPI_Isend((void*) &A[row], local_N, MPI_FLOAT, 0, 999, MPI_COMM_WORLD, &req);						    
				MPI_Wait(&req, &stat);		
				MPI_Isend((void*) &B[row], 1, MPI_FLOAT, 0, 999, MPI_COMM_WORLD, &req);
				MPI_Wait(&req, &stat);
			}
		}

    //wait before starting next norm
		 MPI_Barrier(MPI_COMM_WORLD);
	}
	//------------------------------------------------------------------------------------------------------
	
	if(proc_rank==0){
    //might be able to parallelize back sub too????
		back_substitution();
    t2 = MPI_Wtime();
    print_X();
    printf("Elapsed time = %1.5f ms.\n", (t2-t1)*1000);
    fflush(stdout);
	}
	MPI_Finalize();
  	return 0;
}
