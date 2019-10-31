#include "mpi.h"
#include <stdio.h>

int main(int argc, char** argv){
    int world_size, myid, i;

    //4x4
    char A[16] = {
        "a","b","c","d",
        "e","f","g","h",
        "1","2","3","4",
        "5","6","7","8"
    };

    MPI_Init(argc,argv);
    MPI_Comm_size(MPI_COMM_WORLD,&world_size);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    
    char Buf[4];
    
    MPI_Scatter(&A,4,MPI_CHAR,&Buf,4,MPI_CHAR,0,MPI_COMM_WORLD);
    
    return 0;
}