


#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "mpi.h"



// Works without CUDA support for MPI. MPI cannot send buffers
// from GPU to GPU, at least without CUDA support.


void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err));
        //exit(-1);
    }
    else
    {
        printf("SUCCESSSSSSSSS");
    }             
}

int main(int argc, char *argv[])
{
    int rank;
    //int size;

    MPI_Init(NULL, NULL);
    //MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

//    bool doOneStep = 1;
//    bool doHostToHost = 0;
//    bool doHostToDevice = 1;

    bool doOneStep;
    bool doHostToHost;
    bool doHostToDevice;
    bool doOnlyMemcpy;

    if (argc != 5){std::cout << "c'mon"; return -1;}
    else if (argc > 0)
    {
        doOneStep = atoi(argv[1]);
        doHostToHost = atoi(argv[2]);
        doHostToDevice = atoi(argv[3]);
        doOnlyMemcpy = atoi(argv[4]);
    }

    if (rank == 0)
    {
        std::cout << "# One Step: " << doOneStep << std::endl; 
        std::cout << "# H to H: " << doHostToHost << std::endl; 
        std::cout << "# H to dev: " << doHostToDevice << std::endl; 
        std::cout << "# Only Memcpy: " << doOnlyMemcpy << std::endl; 
    }


    //int numDevs = 0;
    //cudaGetDeviceCount(&numDevs);
    //if (rank == 0) {std::cout << numDevs << std::endl;}

    int p_size;
    
    

    if (rank == 0){std::cout << "# p_size (MB)\t" << "time" << std::endl;}

    float t_0, t_1;
    //int nReps = 100;
    int nReps = 20;


    int M;
    //for (int N = 10; N < 1e6; N *= 2)
    for (int N = 4.2e4; N < 2e6; N *= 1.1)
    {
        M = N;
        MPI_Barrier(MPI_COMM_WORLD);
        p_size = N*sizeof(int);

        int *buf_host = (int*)malloc(N*sizeof(int));   // host buffer
        int *buf_dev;
        cudaMalloc(&buf_dev, N*sizeof(int));       // dev buffer
        
        if (doOnlyMemcpy)
        {
            t_0 = MPI_Wtime();
            for (int i = 0; i < nReps; i++)
            {
                if (rank==1)
                {
                    cudaMemcpy(buf_dev, buf_host, p_size, cudaMemcpyHostToDevice);
                }
            }
            t_1 = MPI_Wtime();
        }
        else if (doHostToHost)
        {
            // std::cout << "HERE BE DRAGONS" << std::endl; 
        
            t_0 = MPI_Wtime();
            for (int i = 0; i < nReps; i++)
            {
                if(rank == 0) {
                    MPI_Send(buf_host, N, MPI_INT, 1, 0, MPI_COMM_WORLD);
                }
                else { // assume MPI rank 1
                    MPI_Recv(buf_host, M, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
            t_1 = MPI_Wtime();
        }
        else if (doHostToDevice)
        {
            //MPI_Barrier(MPI_COMM_WORLD);
            t_0 = MPI_Wtime();
            
            for (int i = 0; i < nReps; i++)
            {

                if(rank == 0) {
                    MPI_Ssend(buf_host, N, MPI_INT, 1, 0, MPI_COMM_WORLD);
                    //std::cout << "sent!" << std::endl;                
                }
                else if (rank==1) { // assume MPI rank 1
                    if (doOneStep){
                        //std::cout << "recving.." << std::endl;
                        MPI_Recv(buf_dev, M, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        
                        //std::cout << "ahgggfggffffghhhhh" << std::endl; 
                    }
                    else{
                        //std::cout << "HERE BE DRAGONS" << std::endl; 
                        MPI_Recv(buf_host, M, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        cudaMemcpy(buf_dev, buf_host, p_size, cudaMemcpyHostToDevice);
                    }
                    //std::cout << "received!" << std::endl;
                }
            }
            t_1 = MPI_Wtime();
        }

        if (rank == 0)
        {
            float t_send = (t_1-t_0)/nReps;
            std::cout << M*sizeof(int) / 1048576. << " \t" << t_send << std::endl;
            //std::cout << N << " \t" << M << std::endl;
        }
    }
    MPI_Finalize();
}
