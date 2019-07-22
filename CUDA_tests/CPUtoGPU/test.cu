

# include <iostream>
# include <cuda.h>
# include "mpi.h"


int main()
{
    int rank;
    
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    

    const int N = 4.3e4; 
    const int M = N; 
    const int p_size = N*sizeof(float);

    float *buf_host, *buf_dev;

    //float *buf_host = (float*)malloc(p_size);
    cudaMallocHost((void **) &buf_host, p_size);
    //float *buf_host = new float[M];

    if (rank==0){
        for (int i = 0; i < N; i++){
            buf_host[i] = 4;
        }
    }
    else if (rank==1){
        for (int i = 0; i < N; i++){
            buf_host[i] = -1;
        }
    }

    cudaMalloc((void **) &buf_dev, p_size);

    const int nIter = 200;
    float t0 = MPI_Wtime();
    //for (int i = 0; i < nIter; i++)
    //{
    if(rank == 0) {
        MPI_Ssend(buf_host, N, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
        std::cout << "HELLO, " << buf_host <<  std::endl;
    }
    else { // assume MPI rank 1

        //MPI_Recv(buf_host, M, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //cudaMemcpy(buf_dev, buf_host, p_size, cudaMemcpyHostToDevice);

        MPI_Recv(buf_dev, M, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "HELLO2, " << buf_host << std::endl;

        //cudaMemcpy(buf_host, buf_dev, p_size, cudaMemcpyDeviceToHost);
        
        for (int i = 0; i < M; i++)
        {
            if (buf_host[i] != -1)
                std::cout << buf_host[i] << std::endl;
        }

    }
    //}
    float t1 = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
    {
        float t_send = (t1-t0)/nIter;
        std::cout << N*sizeof(float) / 1048576. << " \t" << t_send << std::endl;
    }

    MPI_Finalize();
    //cudaFree(buf_dev);
    //free(buf_host);
}

