/*

    This file defines a simple kernel that sums all the elements of a given
    memory buffer.

*/ 



#include <cuda.h>
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include <iostream>
#include "header.h"



//void AccumulatorGPU(double* sum_d)
__global__ 
void AccumulatorGPU(double* buf_d, double* sum_d, int len)
{
    sum_d[0] = 0;

    for (int i = 0; i < len; i++)
    {
        sum_d[0] += buf_d[i];
    }

    printf("[cudaWrapper::AccumulatorGPU]: Result: %.5f\n", sum_d[0]);
}

void cudaWrapper(double* buf_d, double* sum_d, int len)
{
    
    cudaCheck( cudaDeviceSynchronize()); // wait for everyone
    LOG("[cudaWrapper::cudaWrapper]:  calling the kernel", 1);
    AccumulatorGPU<<<1, 1>>>(buf_d, sum_d, len);
    LOG("[cudaWrapper::cudaWrapper]:  kernel called", 1);
    cudaCheck( cudaDeviceSynchronize() ); // wait for everyone
    LOG("[cudaWrapper::cudaWrapper]:  kernel finished", 1);
}
