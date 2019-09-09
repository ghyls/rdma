/*
    This scripts uses RDMA to remotely run an actual kernel. The kernel uses a
    Monte Carlo method to compute the number PI. The scripts prints a readable
    output showing the time consumed in different times of the process.

    Compilation:
        nvcc \
            -I/path/to/ompi/include \
            -L/path/to/ompi/lib \
            -lmpi actualKernelTest.cu -o main

    Runtime:

        mpirun -np 2 --hostfile hosts -x UCX_MEMTYPE_CACHE=n -x UCX_TLS=all \
                     --mca pml ucx --mca btl ^openib main

        where "hosts" is a file containing, for example,
            
            patatrack01.cern.ch slots=1
            felk40 slots=1
*/

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


__global__ void kernel(int *count)
{
    double x, y, z;

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
        // and y, write 1. Else write 0.
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
    float t0, t1, t2;
    bool oneStep = true;
    int rank;
    printf("\n");
    
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // name of the host where the program is being run
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;

    MPI_Get_processor_name(processor_name, &name_len);

    if (rank == 0){printf("#0 is %s", processor_name);}
    if (rank == 1){printf("#1 is %s", processor_name);}

    printf("\n");
    MPI_Barrier(MPI_COMM_WORLD);

    // total number of threads
    long unsigned int n = NUM_BLOCKS * THREADS_PER_BLOCK;
    int *count_d, *count_h;
    
    // allocate memory
    t0 = MPI_Wtime();
    cudaMalloc(&count_d, n * sizeof(int));
    t1 = MPI_Wtime();
    cudaMallocHost((void **) &count_h, n * sizeof(int));
    t2 = MPI_Wtime();
    CUDAErrorCheck();

    if (rank == 0){
        std::cout << "#0: hello! I'll do the CPU part of the job" << std::endl;}
    if (rank == 1){
        std::cout << "#1: hi! Leave the GPU stuff for me" << std::endl;}

    if (rank == 0){
        std::cout << "#0: allocating memory on the GPU took me " << t1-t0
        << " s, but i will not use it at all" << std::endl;
        std::cout << "#0: For the CPU, I spent " << t2-t1 << " s" << std::endl;
    }
    if (rank == 1){
        std::cout << "#1: allocating memory on the GPU took me " << t1-t0
        << " s" << std::endl;
        std::cout << "#1: For the CPU buffer, I spent " << t2-t1 << 
        " s, but I will not use it!" << std::endl;
    }


    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 1) // run the kernel and send back the results.
    {
        std::cout << "#1: Fine, I'm about to launch the kernel" << std::endl;
        
        float t0 = MPI_Wtime();
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
        float t1 = MPI_Wtime();

        std::cout << "#1: Sent! running + sending took me only " << t1-t0 << 
        " seconds!" << std::endl;
    }
    else if (rank==0)
    {
        // do CPU stuff until rank 1 finishes with the kernel
        std::cout << "#0: Now I'm waiting for your results!" << std::endl;
        float t0 = MPI_Wtime();
        MPI_Recv(count_h, n, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        float t1 = MPI_Wtime();
        std::cout << "#0: Received! I've got them!" << std::endl;
        std::cout << "#0: I've been waiting (doing anything) for " << t1-t0 << 
        " seconds. Now I will compute the final result." << std::endl;
    
        long unsigned int reduced_count = 0; // number of stones inside the circle
        for(int i = 0; i < n; i++)
        {
            reduced_count += count_h[i];
        }

        // find the ratio
        long unsigned int total_iter = n * ITER_PER_THREAD;
        double pi = ((double)reduced_count / total_iter) * 4.0; // 4 pi r^2
        std::cout << "\n---------------------------------" << std::endl;
        printf("#0: pi = %.10g\n", pi);
         
    }
    
    cudaFree(count_h);    
    cudaFree(count_d);    
}
