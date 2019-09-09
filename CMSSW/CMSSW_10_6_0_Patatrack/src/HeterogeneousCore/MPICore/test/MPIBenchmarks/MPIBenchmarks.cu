/*
    This script outputs in two columns the tame taken to send some package as
    function of the package size. It is intended to be used to compare different
    MPI PMLs and BTLs, and to test MPI with different configuration options in
    general.

    Compilation:
        nvcc \
            -I/path/to/ompi/include \
            -L/path/to/ompi/lib \
            -lmpi MPIBenchmarks.cu -o main

    Runtime example command:

        mpirun -np 2 --hostfile hosts -x UCX_MEMTYPEACHE=n -x UCX_TLS=all \
                     --mca pml ucx --mca btl ^openib main 1 1 0 0
    
    The four boolean parameters passed in the command line do, in order,

        Don't do extra copies
        Make the transfer H → H
        Make the transfer H → D
        Make a single cudaMemcpy H → D

    So:
        1 1 0 0: H → H
        1 0 1 0: H → D
        0 0 1 0: H → H → D
        0 0 0 1: udaMemcpy H → D
*/

#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "mpi.h"


int main(int argc, char *argv[])
{
    int rank;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    bool doOneStep;
    bool doHostToHost;
    bool doHostToDevice;
    bool doOnlyMemcpy;

    if (argc != 5){std::cout << "you're missing some parameters"; return -1;}
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

    
    if (rank == 0){std::cout << "# p_size (MB)\t" << "time" << std::endl;}
    
    float t_0, t_1;
    int p_size;
    int nReps = 5;  // more statistics

    for (int N = 2; N < 4e7; N *= 1.5)
    {
        MPI_Barrier(MPI_COMM_WORLD);    // start the loop at the same time
        p_size = N*sizeof(int);

        int *buf_host = (int*)malloc(N*sizeof(int));   // host buffer
        int *buf_dev;
        if (rank==1){
            // this conditional is not really neccesary
            cudaMalloc(&buf_dev, N*sizeof(int));       // dev buffer
        }
        
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
            t_0 = MPI_Wtime();
            for (int i = 0; i < nReps; i++)
            {
                if(rank == 0) {
                    MPI_Send(buf_host, N, MPI_INT, 1, 0, MPI_COMM_WORLD);
                }
                else {
                    MPI_Recv(buf_host, N, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
            t_1 = MPI_Wtime();
        }
        else if (doHostToDevice)
        {
            t_0 = MPI_Wtime();
            for (int i = 0; i < nReps; i++)
            {
                if(rank == 0) {
                    MPI_Send(buf_host, N, MPI_INT, 1, 0, MPI_COMM_WORLD);
                }
                else if (rank==1) {
                    if (doOneStep){
                        // Recv on the Device
                        MPI_Recv(buf_dev, N, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                    else{
                        // Recv on the Host and Memcpy
                        MPI_Recv(buf_host, N, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        cudaMemcpy(buf_dev, buf_host, p_size, cudaMemcpyHostToDevice);
                    }
                }
            }
            t_1 = MPI_Wtime();
        }

        if (rank == 0)
        {
            // add a new row to the table
            float t_send = (t_1-t_0)/nReps;
            std::cout << N*sizeof(int) / 1048576. << " \t" << t_send << std::endl;
        }
        if (rank==1) {cudaDeviceReset();}
    }
    MPI_Finalize();
}
