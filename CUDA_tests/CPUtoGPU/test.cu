

# include <iostream>
#include <stdio.h>
# include <cuda.h>
# include <cuda_runtime.h>
# include "mpi.h"


#define CHECK_ERROR( err, msg ) if( err != cudaSuccess ) { std::cerr <<"\033[1;31mERROR:" << cudaGetErrorName ( err ) << "  |  " << "ERROR DES: " <<cudaGetErrorString( err ) << "  |  " << "User msg: " << msg << "\033[0m" <<std::endl; exit( 0 ); }



// number of elements of the host buffer. All of them are sent to the device. 
#define N 16
// number of elements that will be received on the dev buffer. M must be <= N.
#define M 16

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


int main()
{
    int rank;
    
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    float *buf_host = nullptr;
    float *buf_dev  = nullptr;

    // the size of both buffers (not necessarily the size of the package)
    //const long bufferSize = N*sizeof(float);
    int bufferSize = N*sizeof(float);

    // Host buffer: The four following alternatives give the same result. I am
    // pretty sure that the error has nothing to do with the host.
    // -----
    //cudaMallocHost((void **) &buf_host, bufferSize);
    //cudaMallocHost((void **) &buf_host, bufferSize);
    buf_host = (float*)malloc(bufferSize);
    //buf_host = new float[N];
    //float buf_host[N];
    // -----

    if (rank==1) printf("buf_dev before cudaMalloc on rank %d: %p\n", rank, buf_dev);

    // Device buffer
    //cudaMalloc((void **) &buf_dev, bufferSize);   // does not work
    //CHECK_ERROR( cudaMalloc((void **)buf_dev, bufferSize), "hello");    // works with small pkgs
    //cudaMalloc((void **)buf_dev, bufferSize);    // works with small pkgs
    cudaMalloc(&buf_dev, bufferSize);               // works with small pkgs

    if (rank==1) printf("buf_dev after cudaMalloc on rank %d: %p\n", rank, buf_dev);

    if(rank == 0) {
        MPI_Ssend(buf_host, N, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
        //printf("buf_dev after MPI_Ssend on rank %d: %p\n", rank, buf_dev);
    
    }
    else{ 
        // receive into the device buffer
        printf("buf_dev before MPI_recv on rank %d: %p\n", rank, buf_dev);
        MPI_Recv(buf_dev, M, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Finalize();
}

