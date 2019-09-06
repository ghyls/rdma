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



long work_rank_0()
{
    int baseTag_ = 40;


    // Create the data, initialized to zeros
    //auto dataProducer = std::make_unique<std::vector<double>>(SIZE);

    // plant the seed
    //srand(33);

    // init it with a random seed
    //for (uint32_t i = 0; i < SIZE; i++) 
    //{
    //    (*dataProducer)[i] = (double) rand() / RAND_MAX;
    //}

    // =========================================================================


    // Init the MPI stuff
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    // read from the NumberProducer
    //auto const& data = * dataProducer;

    // send the vector to the accumulator (rank of the sender is 0)
    //std::cout << "[NumberOffloader::acquire]:  sending the data to the Accumulator" << std::endl;    
    //MPI_Send(data.data(), data.size(), MPI_DOUBLE, 1, baseTag_ + 98, MPI_COMM_WORLD);
    //std::cout << "[NumberOffloader::acquire]:  data sent!" << std::endl;


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

    struct timespec req, rem;   // nanosleep
    req.tv_sec = 0;
    req.tv_nsec = 0;


    //std::cout <<"entering the main loop" << std::endl;

    int iter = 100; 
    
    struct timespec t_0, t_1, elapsed;  
    clock_gettime(CLOCK_MONOTONIC, &t_0);
    //double t1 = MPI_Wtime();



    int count;
    int flag = 0;
    for (int i = 0; i < iter; i++)
    {
        // what you want to measure here
        
        MPI_Iprobe(1, baseTag_ + 101, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
        std::this_thread::sleep_for(std::chrono::nanoseconds(0));
        //nanosleep(&req, &rem);

        //std::cout << "hi" << std::endl;
        //count = 0; flag = 0;
        //while (not flag)
        //{
        //    count++;
        //    if (count==100) {flag=1;}
        //    //std::this_thread::sleep_for(std::chrono::microseconds(1));
        //}
        //count = 0; flag = 0;
        //while (true)
        //{
        //    count++;
        //    if (count==100) {flag=1;}
        //    if (flag) {break;}
        //    //std::this_thread::sleep_for(std::chrono::microseconds(1));
        //}

    }
    //double t2 = MPI_Wtime();
    clock_gettime(CLOCK_MONOTONIC, &t_1);
    elapsed.tv_sec = t_1.tv_sec - t_0.tv_sec; 
    elapsed.tv_nsec = t_1.tv_nsec - t_0.tv_nsec; 

    long total_elapsed = 1e9 * elapsed.tv_sec + elapsed.tv_nsec;


    //std::cout << total_elapsed/iter << std::endl;
    //std::cout << t2-t1 << std::endl;


    return total_elapsed / iter;

    //std::cout << "[NumberOffloader::produce]:  starting" << std::endl;

    //int len = 0;
    //MPI_Status status;

    //MPI_Probe(1, baseTag_ + 101, MPI_COMM_WORLD, &status);
    //MPI_Get_count(&status, MPI_DOUBLE, &len);
    //std::cout << "[NumberOffloader::produce]:  found MPI_Send, pkg_length = " + std::to_string(len) << std::endl;

    // Create the result vector (will be filled with Server's output)
    //auto result = std::make_unique<std::vector<double>>(len);

    // recive the result from the accumulator
    //MPI_Recv(result->data(), 1, MPI_DOUBLE, 1, baseTag_ + 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //std::cout << "[NumberOffloader::produce]:  result received!" << std::endl;
}

void work_rank_1()
{
/*
    int baseTag_ = 40;

    // Init the MPI stuff
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    // Only rank 1 is supposed to be here

    int len; // length of the data package

    MPI_Status status;

    //std::cout << "[NumberAccS::produce]:  Probing the incoming buffer" << std::endl;
    MPI_Probe(0, baseTag_ + 98, MPI_COMM_WORLD, &status);

    //std::cout << "[NumberAccS::produce]:  reading the length of the buffer" << std::endl;
    MPI_Get_count(&status, MPI_DOUBLE, &len);

    bool useGPU    = true;
    bool GPUDirect = true;

    double *input;
    if (!GPUDirect)
    {
        //std::cout << "[NumberAccS::produce]:  H -> H -> D." << std::endl;   

        input = (double*)malloc(len * sizeof(double));   // host buffer
        // Recv the actual buffer
        MPI_Recv(&input[0], len, MPI_DOUBLE, 0, baseTag_ + 98, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //std::cout << "[NumberAccS::produce]:  data Received" << std::endl;   
    }

    // this is the variable on the host where the result will be stored
    double *sum_h = (double*)malloc(sizeof(double));   // host buffer
    if (useGPU)
    {
        // do the computation in a CUDA kernel
        //std::cout << "[NumberAccS::produce]:  The sum will be computed on the GPU" << std::endl;

        // The array to be summed will be stored here on the GPU
        double *buf_d;
        cudaCheck( cudaMalloc((void **) &buf_d, len * sizeof(double)) );

        // and this will be its sum also on the GPU
        double *sum_d;
        cudaCheck( cudaMalloc((void **) &sum_d, sizeof(double)) );

        if (GPUDirect){
            //std::cout << "[NumberAccS::produce]:  H -> GPU (RDMA)." << std::endl;
            //std::cout << "[NumberAccS::produce]:  Receiving the package on the GPU" << std::endl;
            MPI_Recv(buf_d, len, MPI_DOUBLE, 0, baseTag_ + 98, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //std::cout << "[NumberAccS::produce]:  Received the package on the GPU" << std::endl;
        }
        else{
            // Move the host array to the GPU
            //std::cout << "[NumberAccS::produce]: Copying the package H -> D" << std::endl;
            cudaMemcpy(buf_d, input, len * sizeof(double), cudaMemcpyHostToDevice);            
        }

        // Pass sum_h to the wrapper, so it can change its value to the actual
        // sum (which is what we want)
        cudaWrapper(buf_d, sum_d, len);

        if (GPUDirect)
        {
            //std::cout << "[NumberAccS::produce]:  Sending the result (DtoH)" << std::endl;
            //std::this_thread::sleep_for(std::chrono::seconds(10));
            MPI_Send(sum_d, 1, MPI_DOUBLE, 0, baseTag_ + 101, MPI_COMM_WORLD);
            
            //std::cout << "[NumberAccS::produce]:  result Sent (DtoH)" << std::endl;
        }
        else
        {
            cudaCheck( cudaMemcpy(sum_h, sum_d, sizeof(double), cudaMemcpyDeviceToHost) );    
            //std::cout << "[NumberAccS::produce]:Result: " << sum_h[0] << std::endl;
            
            // send the sum back to the offloader
            MPI_Send(sum_h, 1, MPI_DOUBLE, 0, baseTag_ + 101, MPI_COMM_WORLD);
            //std::cout << "[NumberAccS::produce]:  result Sent (HtoH)" << std::endl;
        }
        
        //cudaFree(buf_d);
        //cudaFree(sum_d);
        // get back the result from the GPU
    }
    else
    {
        //std::cout << "[NumberAccS::produce]:  The computation is about to be done using only the CPU." << std::endl;
        // sum all the input elements 
        sum_h[0] = 0;
        for (int i = 0; i < len; i++)
            sum_h[0] += input[i];

        MPI_Send(sum_h, 1, MPI_DOUBLE, 0, baseTag_ + 101, MPI_COMM_WORLD);
        //std::cout << "[NumberAccS::produce]:  result Sent (HtoH)" << std::endl;
    }
*/
}


int main()
{

    int provided;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);

    // Init the MPI stuff
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int SIZE = 1000;

    float t0;
    float t1;
    float TIME;



    for (int i = 0; i < 10000; i++)
    {
        if (rank == 0) {
            long time_ns = work_rank_0();

            std::cout << time_ns << std::endl;
            
            }
        if (rank == 1) {work_rank_1();}
    }

    MPI_Finalize();
}


