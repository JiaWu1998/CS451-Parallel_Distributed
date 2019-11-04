#include <stdio.h>
#include <stdlib.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>
#include <mpi.h>


#define MAXN 3000
int N;

void gaussian_mpi(int N);
double A[MAXN][MAXN],B[MAXN],X[MAXN];

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

int main(int argc,char *argv[])
{
    int proc,id;
	double t1, t2; 

	MPI_Init(&argc,&argv);//Initiating MPI
	MPI_Comm_rank(MPI_COMM_WORLD,&id);//Getting rank of current processor.
	MPI_Comm_size(MPI_COMM_WORLD,&proc);//Getting number of processor in MPI_COMM_WORLD


	if(id==0)
	{
		/* Process program parameters */
        parameters(argc, argv);

        /* Initialize A and B */
        initialize_inputs();

        // /* Print input matrices */
        // print_inputs();
        t1 = MPI_Wtime();
 							
	}

    N = atoi(argv[1]);//getting matrix dimension from command line argument
	
	MPI_Request request;
	MPI_Status status;
	int p,k,i,j;
	float mp;	

	MPI_Barrier(MPI_COMM_WORLD);// waiting for all processors	

	for (k=0;k<N-1;k++)
 	{	
		//Broadcsting X's and Y's matrix from 0th rank processor to all other processors.
		MPI_Bcast(&A[k][0],N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&B[k],1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
		if(id==0)
		{
			for (p=1;p<proc;p++)
			{
		  		for (i=k+1+p;i<N;i+=proc)
		  		{
				/* Sending X and y matrix from oth to all other processors using non blocking send*/
				   MPI_Isend(&A[i], N, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, &request);
				   MPI_Wait(&request, &status);
				   MPI_Isend(&B[i], 1, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, &request);
				   MPI_Wait(&request, &status);
		  		}
			}
			// implementing gaussian elimination 
			for (i=k+1 ; i<N ; i += proc)
			{
	  			mp = A[i][k] / A[k][k];
	  			for (j = k; j < N; j++)
	 			{
	   				A[i][j] -= A[k][j] * mp;
	 			}
	   			B[i] -= B[k] * mp;
			}
			// Receiving all the values that are send by 0th processor.
			for (p = 1; p < proc; p++)
			{
			  for (i = k + 1 + p; i < N; i += proc)
			  {
			    MPI_Recv(&A[i], N, MPI_DOUBLE, p, 1, MPI_COMM_WORLD, &status);
			    MPI_Recv(&B[i], 1, MPI_DOUBLE, p, 1, MPI_COMM_WORLD, &status);
			  }
			}
		}
		
		
		else
		{
			for (i = k + 1 + id; i < N; i += proc)
			{
				MPI_Recv(&A[i], N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);		
				MPI_Recv(&B[i], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
				mp = A[i][k] / A[k][k];
				for (j = k; j < N; j++)
				{
				    A[i][j] -= A[k][j] * mp;
				}
				B[i] -= B[k] * mp;
				MPI_Isend(&A[i], N, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &request);						    
				MPI_Wait(&request, &status);		
				MPI_Isend(&B[i], 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &request);
				MPI_Wait(&request, &status);
			}
		}
		 MPI_Barrier(MPI_COMM_WORLD);//Waiting for all processors
	}
	
	if(id==0)
	{
		back_substitution();
		t2 = MPI_Wtime();

        printf("Elapsed time = %1.5f ms.\n", (t2-t1)*1000);
        fflush(stdout);
	}
	MPI_Finalize(); //Finalizing the MPI
  	return 0;
}

