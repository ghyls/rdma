/*
    This script communicates two ranks on different nodes outside CMSSW and
    calls a kernel in one of them. Works from H -> H -> D and H -> D. It does it
    for many package sizes and "check for the result" intervals in the offloader
    waiting loop (see the code). Outputs a well formated table that can easily
    be read on Python for making plots, etc.

    It works in the following way:

        - rank 0 fills a std::vector and sends it to rank 1
        - rank 1 receives the std::vector and calls a kernel written in
            cudaWrapper.cu, which returns the sum of all of its elements.
        - rank 1 sends the result back to rank 0

    Compilation command:

        nvcc -I/path/to/ompi/include -L/path/to/ompi/lib -lmpi \
             fullTest.cc cudaWrapper.cu -o main
    
    Sample runtime command:

        mpirun -np 2 --hostfile hosts -x UCX_MEMTYPE_CACHE=n -x UCX_TLS=all
                     --mca pml ucx --mca btl ^openib -x UCX_RNDV_SEND_NBR_THRESH=0K
                     -x UCX_LOG_LEVEL=func main

        where hosts is a file containing, for example,
        
            patatrack01.cern.ch slots=1
            felk40 slots=1
*/



#include <memory>
#include <iostream>
#include <stdio.h>
#include <mpi.h>
#include <chrono>
#include <thread>
#include "header.h"
#include "cudaCheck.h"
#include <cuda_runtime.h>
#include <vector>



void work_rank_0(int SLEEP_TIME, int SIZE)
{
    int baseTag_ = 40;


    // Create the data, initialized to zeros
    auto dataProducer = std::make_unique<std::vector<double>>(SIZE);

    // plant the seed
    srand(33);

    // init it with a random seed
    for (uint32_t i = 0; i < SIZE; i++) 
    {
        (*dataProducer)[i] = (double) rand() / RAND_MAX;
    }

    // =========================================================================


    // Init the MPI stuff
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    // read from the "NumberProducer" (emulated by the above loop)
    auto const& data = * dataProducer;

    // send the vector to the accumulator
    MPI_Send(data.data(), data.size(), MPI_DOUBLE, 1, baseTag_ + 98, MPI_COMM_WORLD);


    // Uncomment this for waiting on another thread
    //auto task = [holder](uint32_t baseTag_){
    //    // Probing for incoming buffers
    //    LOG("[NumberOffloader::acquire]:  Waiting for the result of the Accumulator", 1);
    //    int flag = false;
    //    while (not flag)
    //    {
    //        MPI_Iprobe(1, baseTag_ + 101, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
    //        std::this_thread::sleep_for(std::chrono::microseconds(1));
    //    }
    //};
    //std::thread thr(task, baseTag_);
    //thr.detach();

    int count = 0;
    int flag = false;
    while (not flag)
    {
        MPI_Iprobe(1, baseTag_ + 101, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
        std::this_thread::sleep_for(std::chrono::microseconds(SLEEP_TIME));
        count++;
    }
    
    int len = 0;
    MPI_Status status;

    // read the length of the incoming package
    MPI_Probe(1, baseTag_ + 101, MPI_COMM_WORLD, &status);
    MPI_Get_count(&status, MPI_DOUBLE, &len);

    // Create the result vector (will be filled with Server's output)
    auto result = std::make_unique<std::vector<double>>(len);

    // recive the result from the accumulator
    MPI_Recv(result->data(), 1, MPI_DOUBLE, 1, baseTag_ + 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void work_rank_1(int SLEEP_TIME, int SIZE)
{
    int baseTag_ = 40;

    // Init the MPI stuff
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int len; // length of the data package

    MPI_Status status;

    // read the length of the buffer
    MPI_Probe(0, baseTag_ + 98, MPI_COMM_WORLD, &status);
    MPI_Get_count(&status, MPI_DOUBLE, &len);

    bool useGPU    = true; // Do the computation on the Host or on the Device?
    bool GPUDirect = true; // use GPUDirect RDMA?

    double *input;
    if (!GPUDirect)
    {
        input = (double*)malloc(len * sizeof(double));   // host buffer
        MPI_Recv(&input[0], len, MPI_DOUBLE, 0, baseTag_ + 98, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // this is the variable on the host where the result will be stored
    double *sum_h = (double*)malloc(sizeof(double));
    if (useGPU)
    {
        // do the computation in a CUDA kernel
        // The array to be summed will be stored here on the GPU
        double *buf_d;
        cudaCheck( cudaMalloc((void **) &buf_d, len * sizeof(double)) );

        // and this will be its sum also on the GPU
        double *sum_d;
        cudaCheck( cudaMalloc((void **) &sum_d, sizeof(double)) );

        if (GPUDirect){
            MPI_Recv(buf_d, len, MPI_DOUBLE, 0, baseTag_ + 98, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else{
            // Move the host array to the GPU
            cudaMemcpy(buf_d, input, len * sizeof(double), cudaMemcpyHostToDevice);            
        }

        // Pass sum_h to the wrapper, so it can change its value to the actual
        // sum (which is what we want)
        cudaWrapper(buf_d, sum_d, len);

        if (GPUDirect)
        {
            // back to the client host
            MPI_Send(sum_d, 1, MPI_DOUBLE, 0, baseTag_ + 101, MPI_COMM_WORLD);
        }
        else
        {
            cudaCheck( cudaMemcpy(sum_h, sum_d, sizeof(double), cudaMemcpyDeviceToHost) );    
            
            // send the sum back to the offloader
            MPI_Send(sum_h, 1, MPI_DOUBLE, 0, baseTag_ + 101, MPI_COMM_WORLD);
        }
        
        //cudaFree(buf_d);
        //cudaFree(sum_d);
    }
    else
    {
        // sum all the input elements on the CPU
        sum_h[0] = 0;
        for (int i = 0; i < len; i++)
            sum_h[0] += input[i];

        MPI_Send(sum_h, 1, MPI_DOUBLE, 0, baseTag_ + 101, MPI_COMM_WORLD);
    }
}


int main(int argc, const char * argv[])
{

    MPI_Init(NULL, NULL);

    // Init the MPI stuff
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    float t0;
    float t1;
    float TIME = 0;

    int SLEEP_TIME; // us
    int SIZE;;      // number of doubles
    
    int min_sleep_time = 0;
    int max_sleep_time = 200; 
    int step_sleep_time = 20; 

    int min_size = 20;
    int max_size = 100;
    int step_size = 10;


    // print the first line of the table (sleep times)
    if (rank==0){
        std::cout << -1 << " ";
        for (SLEEP_TIME = min_sleep_time; SLEEP_TIME < max_sleep_time; SLEEP_TIME += step_sleep_time){
            std::cout << SLEEP_TIME << " ";
        }
        std::cout << "\n";
    }

    // print the first column of the table (sizes) and fill the table
    for (SIZE = min_size; SIZE < max_size; SIZE += step_size) // measured in NDoubles
    {
        float size_MB = SIZE * 8. / 1024 / 1024;
        if (rank==0) std::cout << SIZE << " ";
        for (SLEEP_TIME = min_sleep_time; SLEEP_TIME < max_sleep_time; SLEEP_TIME += step_sleep_time)
        {
            int nIter = 10;
            for (int i = 0; i < nIter; i++)
            {
                t0 = MPI_Wtime();
                if (rank == 0) {work_rank_0(SLEEP_TIME, SIZE);}
                if (rank == 1) {work_rank_1(SLEEP_TIME, SIZE);}
                t1 = MPI_Wtime();
                TIME += t1 - t0;
            }
            TIME = TIME / nIter;

            if (rank==0) std::cout << TIME << " ";
            TIME = 0;
        }
        if (rank==0) printf("\n");
    }
    
    MPI_Finalize();
}


