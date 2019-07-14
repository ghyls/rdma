


# include <iostream>
# include <cuda.h>
# include "mpi.h"

// Works without CUDA support for MPI. MPI cannot send buffers
// from GPU to GPU, at least without CUDA support.


int main(int argc, char *argv[])
{
    int rank;
    
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    bool doHostToDevice = 0;
    bool doHostToHost = 0;
    bool doOneStep = 0;

    if (argc != 4){std::cout << "c'mon"; return -1;}
    else if (argc > 0)
    {
        doOneStep = atoi(argv[1]);
        doHostToHost = atoi(argv[2]);
        doHostToDevice = atoi(argv[3]);
    }
    if (rank == 0){std::cout << doOneStep << " " << doHostToHost << " " <<
    doHostToDevice << std::endl;}

    // options:
        // HtoD onestep
        // HtoH onestep
        // HtoD no onestep
        // HtoH no onestep


    int numDevs = 0;
    cudaGetDeviceCount(&numDevs);
    //if (rank == 0) {std::cout << numDevs << std::endl;}

    int size;
    int *buf_host;
    int *buf_dev;

    if (rank == 0){std::cout << "# size (MB)\t" << "time" << std::endl;}

    for (int nInts = 2; nInts < 5e6; nInts *= 2)
    {
        size = nInts*sizeof(int);
        //if (rank == 0) {std::cout << "hi there!" << std::endl;}

        buf_host = (int*)malloc(size);      // host buffer
        cudaMalloc(&buf_dev, size);         // dev buffer

        float t_0, t_1;
        int nReps = 100;


        if (doHostToHost)
        {
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
            t_0 = MPI_Wtime();
            for (int i = 0; i < nReps; i++)
            {
                if(rank == 0) {
                    MPI_Ssend(buf_host, nInts, MPI_INT, 1, 0, MPI_COMM_WORLD);
                }
                else { // assume MPI rank 1

                    if (doOneStep){
                        MPI_Recv(buf_dev, nInts, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                    else{
                        MPI_Recv(buf_host, nInts, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        cudaMemcpy(buf_dev, buf_host, size, cudaMemcpyHostToDevice);
                    }

                }
            }
            t_1 = MPI_Wtime();
        }
        
        if (rank == 0)
        {
            float t_send = (t_1-t_0)/nReps;
            std::cout << size * 1e-6 << " \t" << t_send << std::endl;
        }
        cudaFree(buf_dev);
        free(buf_host);
    }
}
