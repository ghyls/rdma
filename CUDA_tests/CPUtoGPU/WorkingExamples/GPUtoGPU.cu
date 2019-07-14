


# include <iostream>
# include <cuda.h>
# include "mpi.h"


// Works without CUDA support for MPI. MPI cannot send buffers
// from GPU to GPU, at least without CUDA support.

int main()
{

    int rank;
    //int size;

    MPI_Init(NULL, NULL);
    //MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int numDevs = 0;
    cudaGetDeviceCount(&numDevs);

    std::cout << numDevs << std::endl;



    float *buf_host = (float*)malloc(1e6*sizeof(float));
    //int size = 1e6*sizeof(float);
    float *buf_dev;// = (float*)malloc(1e6*sizeof(float));
    cudaMalloc(&buf_dev, 1e6*sizeof(float));

    if( 0 == rank ) {
        MPI_Send(buf_host, 1e6, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
	std::cout << "sent!" << std::endl;
    }
    else { // assume MPI rank 1
        std::cout << "receiving..." << std::endl;
        MPI_Recv(buf_dev, 1e6, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "received!" << std::endl;
    }
}

