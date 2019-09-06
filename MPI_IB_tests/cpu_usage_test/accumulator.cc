// system include files
#include <memory>
#include <iostream>
#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <thread>
#include "header.h"





int main()
{

    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::cout << "main2 rank " << rank << std::endl;
    int len; // length of the data package

    MPI_Status status;

    std::cout << "main2: Ready to receive " << rank << std::endl;
    MPI_Probe(0, 98, MPI_COMM_WORLD, &status);

    MPI_Get_count(&status, MPI_DOUBLE, &len);

    std::cout << "main2: Allocating those doubles: " << len << std::endl;
    // The array to be summed will be stored here on the GPU
    double *buf_d;
    cudaMalloc((void **) &buf_d, len * sizeof(double));

    // and this will be its sum also on the GPU
    double *sum_d;
    cudaMalloc((void **) &sum_d, sizeof(double));

    std::cout << "main2: Receiving " << std::endl;
    MPI_Recv(buf_d, len, MPI_DOUBLE, 0, 98, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    std::cout << "main2: kernel " << std::endl;
    cudaWrapper(buf_d, sum_d, len);

    std::cout << "main2: sending " << std::endl;
    MPI_Ssend(sum_d, 1, MPI_DOUBLE, 0, 101, MPI_COMM_WORLD);

    cudaFree(buf_d);
    cudaFree(sum_d);
}