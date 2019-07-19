

# include <iostream>
# include <cuda.h>
# include "mpi.h"


int main()
{
    int rank;
    
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    

    int N = 4.2e4; 
    int p_size = N*sizeof(float);

    float *buf_host = (float*)malloc(p_size);
    float *buf_dev;// = (float*)malloc(1e6*sizeof(float));
    cudaMalloc(&buf_dev, p_size);

    int nIter = 200;

    float t0 = MPI_Wtime();
    for (int i = 0; i < nIter; i++)
    {
        if(rank == 0) {
            MPI_Ssend(buf_host, N, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
        }
        else { // assume MPI rank 1
            MPI_Recv(buf_dev, N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    float t1 = MPI_Wtime();

    if (rank == 0)
    {
        float t_send = (t1-t0)/nIter;
        std::cout << N*sizeof(float) / 1048576. << " \t" << t_send << std::endl;
    }

    MPI_Finalize();
    //cudaFree(buf_dev);
    //free(buf_host);
}

