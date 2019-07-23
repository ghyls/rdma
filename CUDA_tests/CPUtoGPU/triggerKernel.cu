// hi

# include <cuda.h>
# include <stdio.h>
# include <iostream>
# include "mpi.h"




#define NUM_BLOCKS 1024
#define THREADS_PER_BLOCK 256

#define TRIGER_TAG 666


__global__ void kernel(int *buf_d)
{
    // you are suppposed to write something here.
    while (true)
    {
        if (buf_d[-1] == TRIGER_TAG)
        {
            break;
        }
    }

    printf("imagine that it works");

    // do actual stuff
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

    
    if (rank == 0){printf("#0 is %s\n", processor_name);}
    else if (rank == 1){printf("#1 is %s\n", processor_name);}
    
    
    int n = 1e6;
    
    
    if (rank == 0)
    {
        printf("r0 there\n");        
        // prepare the buffer on the client. The trigger will be the last
        // element of the buffer.


        int *buf_h;
        cudaMallocHost((void **) &buf_h, n * sizeof(int) + sizeof(int));
        
        // init the buffer
        for (int i = 0; i < n; i++){
            buf_h[i] = 34; 
        }
        
        //Activate the trigger
        buf_h[-1] = TRIGER_TAG;

        // send the buffer
        printf("sending the data");        
        MPI_Ssend(buf_h, n+1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

/*    if (rank == 1)
    {
        printf("he 1");        

        // create the buffer
        int *buf_d;
        cudaMalloc(&buf_d, n * sizeof(int) + sizeof(int));

        // launch the kernel
        kernel<<<1, 1>>>(buf_d);

        // receive the package
        MPI_Recv(buf_d, n+1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("hey");
    }
*/
}































































