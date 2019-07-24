// hi

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



__global__ void kernel(int* buf_d, int len)
{
    // should I run this loop on every thread?
    while (true)
    {     
        printf("%2d\t", buf_d[len]);
        if (buf_d[len] == TRIGER_TAG)
        {
            printf("now the actual kernel code starts running\n");
            printf("%2d\n", buf_d[len]);
            break;
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
        // element of the buffer.
        int *buf_h;
        cudaMallocHost((void **) &buf_h, n * sizeof(int) + sizeof(int));
        
        // init the buffer
        for (int i = 0; i < n; i++){
            buf_h[i] = 23; 
        }
        
        //Activate the trigger
        buf_h[n] = TRIGER_TAG;

        // send the buffer
        printf("sending the data\n");        
        MPI_Ssend(buf_h, n+1, MPI_INT,1, 0, MPI_COMM_WORLD);
    }

    if (rank == 1)
    {
        // create the buffer on the device
        int *buf_d;
        cudaMalloc((void **) &buf_d, (n+1) * sizeof(int));

        // launch the kernel
        kernel<<<1, 1>>>(buf_d, n);
        printf("just launched the kernel\n");

        // receive the package
        printf("receiving\n");
        MPI_Recv(buf_d, n+1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // make sure that the CPU waits for the GPU to finish.
        gpuErrchk( cudaDeviceSynchronize() ); 
    }

    MPI_Finalize();
}



Dear Felice,

    I think that I have just somehow archieved the goal you suggested me
    yesterday. Now, the core that receives the buffer, runs the kernel at the
    beggining, and the kernel keeps running a while(true) loop, until the whole
    buffer has been received.

    I don't know if it's worth, because now the core has to wait until it has
    called the kernel to start receiving the package (I don't know if it is what
    you meant yesterday). 
    
    At any rate, I am getting a "weird" error related to a printf inside the
    kernel. For some reason the code hangs if I remove it. I'm currently
    debbugging it, but it is taking longer than I expected.

    I have also simplified the code that we talked about yesterday (the one that
    sends buffers from host to device and crashes when the buffer is too small
    or too big). Now it has barely ~20 lines and it is easily readable.

    I'd like to discuss this new "progress" with you tomorrow if you have time.

    Thank you very much once again!

Best regards,
Mario


























































