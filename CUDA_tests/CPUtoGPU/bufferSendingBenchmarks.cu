


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

    bool doOneStep = 1;
    bool doHostToHost = 0;
    bool doHostToDevice = 1;

    if (argc != 4){std::cout << "c'mon"; return -1;}
    else if (argc > 0)
    {
        doOneStep = atoi(argv[1]);
        doHostToHost = atoi(argv[2]);
        doHostToDevice = atoi(argv[3]);
    }

    if (rank == 0)
    {
        std::cout << "# One Step: " << doOneStep << std::endl; 
        std::cout << "# H to H: " << doHostToHost << std::endl; 
        std::cout << "# H to dev: " << doHostToDevice << std::endl; 
    }

    //if (rank == 0){std::cout << doOneStep << " " << doHostToHost << " " <<
    //doHostToDevice << std::endl;}

    // options: 
        // HtoD onestep
        // HtoH onestep
        // HtoD no onestep
        // HtoH no onestep


    //int numDevs = 0;
    //cudaGetDeviceCount(&numDevs);
    //if (rank == 0) {std::cout << numDevs << std::endl;}

    int p_size;
    
    

    if (rank == 0){std::cout << "# p_size (MB)\t" << "time" << std::endl;}

    float t_0, t_1;
    int nReps = 100;



    //for (int nInts = 4.2e4; nInts < 1e6; nInts *= 1.1)
    for (int nInts = 5e5; nInts < 1e7; nInts *= 1.2)
    {
        //MPI_Barrier(MPI_COMM_WORLD);
        p_size = nInts*sizeof(int);

        int *buf_host = (int*)malloc(nInts*sizeof(int));   // host buffer
        int *buf_dev;
        cudaMalloc(&buf_dev, nInts*sizeof(int));       // dev buffer


        if (doHostToHost)
        {
            // std::cout << "HERE BE DRAGONS" << std::endl; 
        
            t_0 = MPI_Wtime();
            for (int i = 0; i < nReps; i++)
            {
                if(rank == 0) {
                    MPI_Ssend(buf_host, nInts, MPI_INT, 1, 0, MPI_COMM_WORLD);
                }
                else { // assume MPI rank 1
                    MPI_Recv(buf_host, nInts, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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
                MPI_Barrier(MPI_COMM_WORLD);
                //cudaMemcpy(buf_dev, buf_host, p_size, cudaMemcpyHostToDevice); // not here
                
                //std::cout << i << " out of " << nReps << std::endl;
                if(rank == 0) {
                    MPI_Ssend(buf_host, nInts, MPI_INT, 1, 0, MPI_COMM_WORLD);
                    //std::cout << "sent!" << std::endl;                
                }
                else if (rank==1) { // assume MPI rank 1
                    if (doOneStep){
                        //std::cout << "recving.." << std::endl;
                        MPI_Recv(buf_dev, nInts, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        
                        //std::cout << "ahgggfggffffghhhhh" << std::endl; 
                    }
                    else{
                        //std::cout << "HERE BE DRAGONS" << std::endl; 
                        MPI_Recv(buf_host, nInts, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        cudaMemcpy(buf_dev, buf_host, p_size, cudaMemcpyHostToDevice);
                    }
                    //std::cout << "received!" << std::endl;
                }
                //std::cout << rank << ": last" << std::endl;

            }
            //std::cout << rank << ": measuring t1" << std::endl;

            t_1 = MPI_Wtime();
        }
        //std::cout << rank << ": entering the barrier" << std::endl;

        //MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0)
        {
            float t_send = (t_1-t_0)/nReps;
            std::cout << p_size / 1048576. << " \t" << t_send << std::endl;
            //std::cout << p_size << " \t" << t_send << std::endl;
        }
        //cudaFree(buf_dev);
        //free(buf_host);
        //delete buf_dev;
        //delete buf_host;
        //cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
    }
    //MPI_Finalize();
}
