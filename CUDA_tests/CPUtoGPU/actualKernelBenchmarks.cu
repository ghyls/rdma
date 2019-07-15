


# include <iostream>
# include <stdio.h>
# include <cuda.h>
# include <curand.h>
# include <curand_kernel.h>
# include <time.h>
# include "mpi.h"

// Works without CUDA support for MPI and Infiniband

#define NUM_BLOCKS 1024
#define THREADS_PER_BLOCK 256
#define ITER_PER_THREAD 2048

#define PI 3.14159265359


__global__ void kernel(int *count)
{
    // size(count) = number of threads * sizeof(int)

    double x, y, z;

    // find the overall ID of the thread
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    count[index] = 0;
    curandState state; // the generator of random numbers
    curand_init((unsigned long long)clock() + index, 0, 0, &state);
    for (int i = 0; i < ITER_PER_THREAD; i++)
    {
        x = curand_uniform_double(&state); // return random double form generator
        y = curand_uniform_double(&state); // return random double form generator
        z =  x * x + y * y;
 
        // if the stone falls in the circle contained in the square defined by x
        // and y, put 1. Else put 0.
        if (z <= 1)
            count[index] += 1;
    }
}

void CUDAErrorCheck()
{
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error : %s (%d)\n", cudaGetErrorString(error), error);
        exit(EXIT_FAILURE);
    }
}

int main()
{
    bool oneStep = false;
    int rank;
    
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // total number of threads
    long unsigned int n = NUM_BLOCKS * THREADS_PER_BLOCK;
    int *count_d, *count_h;

    if (rank == 0){std::cout << "#0: hello! How are you?" << std::endl;}

    // allocate memory, somewhere
    cudaMalloc(&count_d, n * sizeof(int));
    cudaMallocHost((void **) &count_h, n * sizeof(int));
    CUDAErrorCheck();


    if (rank == 1) // run the kernel and send back the results.
    {
        std::cout << "#1: Fine, I'm about to launch the kernel" << std::endl;
        
        kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(count_d);
        std::cout << "#1: Launched!" << std::endl;
        
        cudaDeviceSynchronize(); // wait for everyone
        CUDAErrorCheck();
        
        std::cout << "#1: Hey! I'm sending you the results!" << std::endl;
        if (oneStep)
        {
            MPI_Ssend(count_d, n, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
        else
        {
            cudaMemcpy(count_h, count_d, n * sizeof(int), cudaMemcpyDeviceToHost);
            MPI_Ssend(count_h, n, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }
    else if (rank==0)
    {

        // do CPU stuff until rank 1 finishes with the kernel
        std::cout << "#0: I'm waiting for you bro!" << std::endl;

        MPI_Recv(count_h, n, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "#0: Received! Nice catch!" << std::endl;
    
        long unsigned int reduced_count = 0; // number of stones inside the circle
        for(int i = 0; i < n; i++)
        {
            reduced_count += count_h[i];
        }

        // find the ratio
        long unsigned int total_iter = n * ITER_PER_THREAD;
        double pi = ((double)reduced_count / total_iter) * 4.0; // 4 pi r^2
        printf("#0: PI [%lu iterations] = %.10g\n", total_iter, pi);
        printf("#0: Error = %.10g\n", pi - PI);
    }
        
    cudaFree(count_h);    
    cudaFree(count_d);    
}
