#include <memory>
#include <iostream>
#include <mpi.h>
#include <chrono>
#include <thread>
#include "header.h"
#include "cudaCheck.h"
#include <cuda_runtime.h>




void work_rank_0(int rank, int SLEEP_TIME, int SIZE)
{
    double *data = (double*)malloc(SIZE * sizeof(double));   // host buffer

    // plant the seed
    srand(40);

    // init it with a random seed
    for (uint32_t i = 0; i < SIZE; i++) 
    {
        data[i] = (double) rand() / RAND_MAX;
    }

    std::cout << "rank " << rank << " sending 98" << std::endl;
    MPI_Ssend(data, SIZE, MPI_DOUBLE, 1, 98, MPI_COMM_WORLD);

    int flag = false;
    //std::cout << "rank " << rank << " polling" << std::endl;
    //std::cout << "measure" << std::endl;
    //while (not flag)
    //{
    //    MPI_Iprobe(1, 101, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
    //    std::this_thread::sleep_for(std::chrono::microseconds(100000));
    //    std::cout << "rank " << rank << ". . ." << std::endl;
    //}

    int len = 0;
    MPI_Status status;

    std::cout << "rank " << rank << " probing 101" << std::endl;
    MPI_Probe(1, 101, MPI_COMM_WORLD, &status);
    MPI_Get_count(&status, MPI_DOUBLE, &len);

    // Create the result vector (will be filled with Server's output)
    double *result = (double*)malloc(len * sizeof(double));   // host buffer

    // recive the result from the accumulator
    std::cout << "rank " << rank << " receiving 101" << std::endl;        
    MPI_Recv(result, 1, MPI_DOUBLE, 1, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    std::cout << "rank " << rank << " offloader done" << std::endl;

}

void work_rank_1(int rank, int SLEEP_TIME, int SIZE)
{
    std::cout << "rank " << rank << " start" << std::endl;
    int len; // length of the data package

    MPI_Status status;

    std::cout << "rank " << rank << " probing 98" << std::endl;
    MPI_Probe(0, 98, MPI_COMM_WORLD, &status);

    MPI_Get_count(&status, MPI_DOUBLE, &len);


    std::cout << "rank " << rank << " allocating " << len <<  std::endl;
    // The array to be summed will be stored here on the GPU
    double *buf_d;
    cudaCheck( cudaMalloc((void **) &buf_d, len * sizeof(double)) );
    // and this will be its sum also on the GPU
    double *sum_d;
    cudaCheck( cudaMalloc((void**)&sum_d, sizeof(double)) );

    // another buffer on the host
    double *buf_h = (double*)malloc(len * sizeof(double));   // host buffer

    std::cout << "rank " << rank << " recving 98" << std::endl;
    
    //MPI_Recv(buf_d, len, MPI_DOUBLE, 0, 98, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Recv(buf_h, len, MPI_DOUBLE, 0, 98, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    cudaMemcpy(buf_d, buf_h, len * sizeof(double), cudaMemcpyHostToDevice);            

    std::cout << "rank " << rank << " kernel" << std::endl;
    cudaWrapper(buf_d, sum_d, len);

    std::cout << "rank " << rank << " sending 101" << std::endl;
    MPI_Ssend(sum_d, 1, MPI_DOUBLE, 0, 101, MPI_COMM_WORLD);

    std::cout << "rank " << rank << " freee" << std::endl;
    
    //cudaFree(buf_d);
    //cudaFree(sum_d);
}


int main()
{

    MPI_Init(NULL, NULL);

    // Init the MPI stuff
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int SIZE = 1e4;
    int SLEEP_TIME = 1;


    float t0;
    float t1;
    float TIME;

    // = = = = = = = = = = = = = =
    // int SLEEP_TIME = 1; // us
    // int SIZE = 1e4;     // doubles
    // = = = = = = = = = = = = = =

    //work(rank, SLEEP_TIME, SIZE);
    
    int min_sleep_time = 1;
    int max_sleep_time = 200;
    
    if (rank == 0) {work_rank_0(rank, SLEEP_TIME, SIZE);}
    if (rank == 1) {work_rank_1(rank, SLEEP_TIME, SIZE);}

    /*
    if (rank==0){
        std::cout << -1 << " ";
        for (SLEEP_TIME = min_sleep_time; SLEEP_TIME < max_sleep_time; SLEEP_TIME += 20){
            std::cout << SLEEP_TIME << " ";
        }
        printf("\n");
    }

    for (SIZE = 10; SIZE < 15000; SIZE *= 1.1)
    {
        if (rank==0) std::cout << SIZE << " ";
        for (SLEEP_TIME = min_sleep_time; SLEEP_TIME < max_sleep_time; SLEEP_TIME += 20)
        {
            t0 = MPI_Wtime();
            //work(rank, SLEEP_TIME, SIZE);
            t1 = MPI_Wtime();

            TIME = t1 - t0;
            if (rank==0) std::cout << TIME << " ";
        }
        if (rank==0) printf("\n");
    }
    */
    

}


