/*

    This is a minimal example, that just sends data H → D, D → H and H → H.

    You could also use it to make sure that works also using synchronous calls
    (MPI_Ssend) and other MPI functions like MPI_Barrier.

    Compilation

        nvcc \
            -I/home/mariog/openmpi-4.0.1/build/include \
            -L/home/mariog/openmpi-4.0.1/build/lib \
            -lmpi test.cu -o main

    Runtime

        mpirun -np 2 --hostfile hosts -x UCX_MEMTYPEACHE=n -x UCX_TLS=all \
                     --mca pml ucx --mca btl ^openib main    

*/



#include <cuda.h>
#include "mpi.h"


int main()
{    
    MPI_Init(nullptr, nullptr);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    

    float *buf_host, *buf_dev;

    int nFloats = 10;

    buf_host = (float*)malloc(nFloats * sizeof(float));
    cudaMalloc((void**)&buf_dev, nFloats * sizeof(float));

    if (rank == 0) {
        printf("rank 0: Calling MPI\n");
        //MPI_Send(buf_host, size, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(buf_host, size, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //MPI_Send(buf_dev, size, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
        //MPI_Recv(buf_dev, size, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("rank 0: done\n");
    } else { 
        printf("rank 1: Calling MPI\n");
        //MPI_Recv(buf_dev, size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Ssend(buf_dev, size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        //MPI_Recv(buf_host, size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //MPI_Send(buf_host, size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        printf("rank 1: done\n");
    }
    
    MPI_Finalize();
}

