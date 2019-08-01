

# include <cuda.h>
# include <stdio.h>
# include <iostream>
# include "mpi.h"


#define NUM_BLOCKS 1024
#define THREADS_PER_BLOCK 256

#define TRIGER_TAG 999

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__global__ void kernel(volatile int* buf_d, int len)
{
    //printf("#k: why I am not being printed...\n");

    while (true)
    {
        //recv[0] = 178;
        if (buf_d[len] == TRIGER_TAG)
        {
            //asm("trap;");
            return;
        }

        for (int i = 0; i < 1e3; i++)
        {
            continue;
        }
    }

    // --------
    // Now you are supposed to do something useful
}


int main()
{
    int rank;
    printf("\n");
    
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // do the presentations...
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;

    MPI_Get_processor_name(processor_name, &name_len);
    printf("#%d is %s\n", rank, processor_name);
    

    int n = 1e6; // number of elements of the buffer
    
    if (rank == 0)
    {
        // prepare the buffer on the client. The trigger will be the last
        // element of the buffer. buf_h has *n* elements.
        int *buf_h;
        cudaMallocHost((void **) &buf_h, (n+1) * sizeof(int));
        
        // init the buffer
        for (int i = 0; i < n; i++){
            buf_h[i] = 23; 
        }
        
        //Activate the trigger
        buf_h[n] = TRIGER_TAG;

        // send the buffer
        printf("#0: sending the data\n");        
        MPI_Ssend(buf_h, n+1, MPI_INT,1, 0, MPI_COMM_WORLD);
        printf("#0: sent\n");        
    }

    if (rank == 1)
    {
        // create the buffer on the device
        int *buf_d;
        cudaMalloc((void **) &buf_d, (n+1) * sizeof(int));

        // launch the kernel
        kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(buf_d, n);
        printf("#1: just launched the kernel\n");

        // receive the package
        MPI_Recv(buf_d, n+1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("#1: received\n");

        // make sure that the CPU waits for the GPU to finish.
        gpuErrchk( cudaDeviceSynchronize() ); 
    }

    MPI_Finalize();
}


























































