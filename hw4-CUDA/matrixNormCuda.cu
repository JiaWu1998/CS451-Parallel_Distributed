/* Matrix normalization.
* Compile with "nvcc matrixNormCuda.c -lm"
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

/* Program Parameters */
#define N 6000  /* Matrix size */

/* Matrices */
volatile float A[N][N], B[N][N];

/* CUDA arrays */
volatile float *A_d, *B_d;


/* Initialize A and B*/
void initialize_inputs() {
    int row, col;
    
    srand((unsigned)time(NULL));
    for (row = 0; row < N; row++) {
        for (col = 0; col < N; col++) {
            A[row][col] = (float)rand() / 32768.0;
            B[row][col] = 0.0;
        }
    }
    
}


/* Kernel function */

__global__ void matrixNorm(float mu, float sigma, int N) {
    for (row=0; row < N; row++) {
        if (sigma == 0.0)
            B_d[row*N + col] = 0.0;
        else
            B_d[row*N + col] = (A_d[row*N + col] - mu) / sigma;
    }
}



int main(int argc, char **argv) {
    /* Timing variables */
    struct timeval start, stop;  /* Elapsed times using gettimeofday() */
    struct timezone tzdummy;
    unsigned long long runtime;
    int col, row;
    float mu, sigma;
    
    /* Initialize A and B */
    initialize_inputs();
    
    
    /* Start Clock */
    printf("\n---------------------------------------------\n");
    printf("Matrix size N = %d", N);
    printf("\nStarting clock.\n\n");
    gettimeofday(&start, &tzdummy);
    
    printf("Computing Serially.\n");
    
    /*allocating GPU space*/
    cudaError_t err1 = cudaMalloc((void **) &A_d, N);
    cudaError_t err2 = cudaMalloc((void **) &B_d, N);

    /*transfer data from host to device*/
    cudaMemcpy(A_d,A,N*N,cudaMemcpyHostToDevice);

    /* Kernal Matrix Normalization */
    for (col=0; col < N; col++) {
        mu = 0.0;
        for (row=0; row < N; row++)
            mu += A[row][col];
        mu /= (float) N;
        sigma = 0.0;
        for (row=0; row < N; row++)
            sigma += powf(A[row][col] - mu, 2.0);
        sigma /= (float) N;
        sigma = sqrt(sigma);
        matrixNorm<<< >>>(mu,sigma, N);
    }

    /*transfer data from device to host*/
    cudaMemcpy(B_d,B,N*N,cudaMemcpyDeviceToHost);
    
    /*deallocating GPU space*/
    cudaFree(A_d);
    cudaFree(B_d);
    
    /* Stop Clock */
    gettimeofday(&stop, &tzdummy);
    runtime = (unsigned long long)(stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_usec - start.tv_usec);
    
    
    /* Display timing results */
    printf("Runtime = %g ms.\n", (float)runtime/(float)1000);
    printf("\nStopped clock.");
    printf("\n---------------------------------------------\n");
    
    exit(0);
}
