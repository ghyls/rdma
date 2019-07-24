

# include <iostream>
# include <cuda.h>
# include "mpi.h"

// number of elements of the host buffer. All of them are sent to the device. 
#define N 1e4 
// number of elements that will be received on the dev buffer. M must be <= N.
#define M 1e4

// When N = M ~ 1e5             -> Everything works.
// When N = M < 4.2e4           -> Segmentation fault (non UCX related)
// When N = M > ~1e8            -> UCX ERROR: Segmentation fault.
// When N < 4.2e4 and M << N    -> works fine. 


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
    
    float *buf_host;
    float *buf_dev;

    // the size of both buffers (not necessarily the size of the package)
    const int bufferSize = N*sizeof(float);

    // Host buffer: The four following alternatives give the same result. I am
    // pretty sure that the error has nothing to do with the host.
    // -----
    cudaMallocHost((void **) &buf_host, bufferSize);
    //float *buf_host = (float*)malloc(bufferSize);
    //buf_host = new float[N];
    //float buf_host[N];
    // -----

    // Device buffer
    cudaMalloc((void **) &buf_dev, bufferSize);

    if(rank == 0) {
        MPI_Ssend(buf_host, N, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
    }
    else{ 
        // receive into the device buffer
        MPI_Recv(buf_dev, M, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Finalize();
}

