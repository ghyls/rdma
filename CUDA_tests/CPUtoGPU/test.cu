
#include <cstring>
#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "mpi.h"


#define CHECK_ERROR( err, msg ) if( err != cudaSuccess ) { std::cerr <<"\033[1;31mERROR:" << cudaGetErrorName ( err ) << "  |  " << "ERROR DES: " <<cudaGetErrorString( err ) << "  |  " << "User msg: " << msg << "\033[0m" <<std::endl; exit( 0 ); }



// number of elements of the host buffer. All of them are sent to the device. 
#define N 41000
// number of elements that will be received on the dev buffer. M must be <= N.
#define M N

// When N = M ~ 1e5             -> Everything works.
// When N = M < 4.2e4 (0.84 MB) -> Segmentation fault (non UCX related)
// When N = M > 5.7e7 (114 MB)  -> UCX ERROR: Segmentation fault.
// When N < 4.2e4 and M << N    -> works fine. 

// 2147483647

/* After printing some memory addresses, I have noticed that the segmentation
faults occur because MPI is still trying to write in the device after the buffer
is over. Since the floats weight 4 bytes both in Patatrack and in Felk, it could
be most likely because MPI doesn't start writing the buffer right at the
beginning. I am still trying to prove this.*/


int main(int argc, const char * argv[])
{
    int rank;
    int size = argc > 1 ? atoi(argv[1]) : N;
    
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    float *buf_host = nullptr;
    float *buf_dev  = nullptr;

    // the size of both buffers (not necessarily the size of the package)
    //const long bufferSize = size*sizeof(float);
    int bufferSize = size*sizeof(float);

    // Host buffer
    if (rank==0) {
        buf_host = (float*)malloc(bufferSize);
        std::cout << "size (MB): " << bufferSize / 1048576. << std::endl;
    }

    // Device buffer
    if (rank==1) {
        printf("buf_dev before cudaMalloc on rank %d: %p\n", rank, buf_dev);
	cudaMalloc(&buf_dev, bufferSize);               // works with small pkgs
	//cudaMalloc(&buf_dev, 16*1024*1024);               // works with small pkgs
        printf("buf_dev after cudaMalloc on rank %d: %p\n", rank, buf_dev);
    }

    if (rank == 0) {
        MPI_Send(buf_host, size, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
        //printf("buf_dev after MPI_Ssend on rank %d: %p\n", rank, buf_dev);
    } else { 
        // receive into the device buffer
        printf("buf_dev before MPI_recv on rank %d: %p\n", rank, buf_dev);
        MPI_Recv(buf_dev, M, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Host buffer
    if (rank==0) {
        free(buf_host);
    }

    // Device buffer
    if (rank==1) {
	cudaFree(buf_dev);
    }

    MPI_Finalize();
}

