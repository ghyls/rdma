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
    auto const& data = * dataProducer;

    // send the vector to the accumulator (rank of the sender is 0)
    //std::cout << "[NumberOffloader::acquire]:  sending the data to the Accumulator" << std::endl;    
    MPI_Send(data.data(), data.size(), MPI_DOUBLE, 1, baseTag_ + 98, MPI_COMM_WORLD);
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
        
    int flag = false;
    while (not flag)
    {
        MPI_Iprobe(1, baseTag_ + 101, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
        std::this_thread::sleep_for(std::chrono::microseconds(SLEEP_TIME));
    }
    
    //std::cout << "[NumberOffloader::produce]:  starting" << std::endl;

    int len = 0;
    MPI_Status status;

    MPI_Probe(1, baseTag_ + 101, MPI_COMM_WORLD, &status);
    MPI_Get_count(&status, MPI_DOUBLE, &len);
    //std::cout << "[NumberOffloader::produce]:  found MPI_Send, pkg_length = " + std::to_string(len) << std::endl;

    // Create the result vector (will be filled with Server's output)
    auto result = std::make_unique<std::vector<double>>(len);

    // recive the result from the accumulator
    MPI_Recv(result->data(), 1, MPI_DOUBLE, 1, baseTag_ + 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //std::cout << "[NumberOffloader::produce]:  result received!" << std::endl;
}

void work_rank_1(int SLEEP_TIME, int SIZE)
{
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
}


int main(int argc, const char * argv[])
{

    MPI_Init(NULL, NULL);

    // Init the MPI stuff
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int SIZE = 1000;
    int SLEEP_TIME = 1;


    float t0;
    float t1;
    float TIME = 0;

    // = = = = = = = = = = = = = =
    // int SLEEP_TIME = 1; // us
    // int SIZE = 1e4;     // doubles
    // = = = = = = = = = = = = = =

    //work(rank, SLEEP_TIME, SIZE);
    
    int min_sleep_time = 0; //10; //0;
    int max_sleep_time = 200; //1000000; //200;
    int step_sleep_time = 4; //2; //20;

    int min_size = argc == 5 ? atoi(argv[1]) : 10;
    int max_size = argc == 5 ? atoi(argv[2]) : 1000;
    int step_size = argc == 5 ? atoi(argv[3]) : 10;
    int command_line = argc == 5 ? atoi(argv[4]) : 1;

    //if (rank == 0) {work_rank_0(SLEEP_TIME, SIZE);}
    //if (rank == 1) {work_rank_1(SLEEP_TIME, SIZE);}

    if (command_line)
    {
        if (rank==0){
            std::cout << -1 << " ";
            for (SLEEP_TIME = min_sleep_time; SLEEP_TIME < max_sleep_time; SLEEP_TIME += step_sleep_time){
                std::cout << SLEEP_TIME << " ";
            }
            std::cout << "\n";
        }
    }
    for (SIZE = min_size; SIZE < max_size; SIZE += step_size)
    {
        
        float size_MB = SIZE * 4. / 1024 / 1024;
        if (rank==0) std::cout << SIZE << " ";
        for (SLEEP_TIME = min_sleep_time; SLEEP_TIME < max_sleep_time; SLEEP_TIME += step_sleep_time)
        {
            for (int i = 0; i < 10; i++)
            {
                t0 = MPI_Wtime();
                if (rank == 0) {work_rank_0(SLEEP_TIME, SIZE);}
                if (rank == 1) {work_rank_1(SLEEP_TIME, SIZE);}
                t1 = MPI_Wtime();
                TIME += t1 - t0;
            }
            TIME = TIME / 10;

            if (rank==0) std::cout << TIME << " ";
        }
        if (rank==0) printf("\n");
    }
    
    
    
    MPI_Finalize();
}


