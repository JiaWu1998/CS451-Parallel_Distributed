#include "mpi.h"
#include <stdio.h>

int main(int argc, char** argv){
    int world_size, myid, i;

    //4x4
    // char A[16] = {
    //     'a','b','c','d',
    //     'e','f','g','h',
    //     '1','2','3','4',
    //     '5','6','7','8'
    // };

    char B[16] = {'0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0'};

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&world_size);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    
    char Buf[6];

    if (myid == 0){
        Buf[0] = 'a';
        Buf[1] = 'b';
        Buf[2] = 'c';
        Buf[3] = 'd';
        Buf[4] = '-';
        Buf[5] = '-';
        
    }
    if (myid == 1){
        Buf[0] = 'e';
        Buf[1] = 'f';
        Buf[2] = 'g';
        Buf[3] = 'h';
        Buf[4] = '+';
        Buf[5] = '+';
    }
    if (myid == 2){
        Buf[0] = '1';
        Buf[1] = '2';
        Buf[2] = '3';
        Buf[3] = '4';
        Buf[4] = '=';
        Buf[5] = '=';
    }
    if (myid == 3){
        Buf[0] = '5';
        Buf[1] = '6';
        Buf[2] = '7';
        Buf[3] = '8';
        Buf[4] = ')';
        Buf[5] = ')';
    }
    
    
    // MPI_Scatter(A,4,MPI_CHAR,Buf,4,MPI_CHAR,0,MPI_COMM_WORLD);
    MPI_Gather(&Buf[0], 1, MPI_CHAR, &B[0], 1, MPI_CHAR,0,MPI_COMM_WORLD);
    MPI_Gather(&Buf[1], 1, MPI_CHAR, &B[4], 1, MPI_CHAR,0,MPI_COMM_WORLD);
    // MPI_Gather(&Buf[2], 1, MPI_CHAR, &B[8], 1, MPI_CHAR,0,MPI_COMM_WORLD);
    // MPI_Gather(&Buf[3], 1, MPI_CHAR, &B[12], 1, MPI_CHAR,0,MPI_COMM_WORLD);
    // for (i=0; i<4; ++i){
    //     printf("%c\t",Buf[i]);
    // }
    // printf("\n");

    if(myid == 0){
        for (i=0; i<16; ++i){
            printf("%c\t",B[i]);
            // printf("\n") if i % 4 == 0;
        }
    }
    return 0;
}