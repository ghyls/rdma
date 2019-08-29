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



long double work_rank_0(int SLEEP_TIME, int SIZE)
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
    


    int flag = false;
    int reps = 10;
    struct timespec t0_thread, t1_thread, elapsed_thread;  
    struct timespec t0_real, t1_real, elapsed_real;  
    
    struct timespec req, rem;   // nanosleep
    req.tv_sec = 0;
    req.tv_nsec = SLEEP_TIME;



    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t0_thread);
    clock_gettime(CLOCK_MONOTONIC, &t0_real);

    for (int i = 0; i < reps; i++)
    {
        MPI_Iprobe(1, baseTag_ + 101, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
        //std::this_thread::sleep_for(std::chrono::nanoseconds(SLEEP_TIME));
        nanosleep(&req, &rem);
    }
    //MPI_Probe(1, baseTag_ + 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1_thread);
    clock_gettime(CLOCK_MONOTONIC, &t1_real);

    elapsed_thread.tv_sec = t1_thread.tv_sec - t0_thread.tv_sec; 
    elapsed_thread.tv_nsec = t1_thread.tv_nsec - t0_thread.tv_nsec; 

    elapsed_real.tv_sec = t1_real.tv_sec - t0_real.tv_sec; 
    elapsed_real.tv_nsec = t1_real.tv_nsec - t0_real.tv_nsec; 



    long double total_elapsed_thread = ((long double) 1e9 * elapsed_thread.tv_sec + elapsed_thread.tv_nsec);
    long double total_elapsed_real = ((long double) 1e9 * elapsed_real.tv_sec + elapsed_real.tv_nsec);
    
    //std::cout << "thread: " << total_elapsed_thread << std::endl;
    //std::cout << "real: " << total_elapsed_real << std::endl;
    //std::cout << "ratio: " << total_elapsed_thread / total_elapsed_real << std::endl;

    //std::cout << SLEEP_TIME << "    " <<  total_elapsed_thread / total_elapsed_real << std::endl;
    return total_elapsed_thread / total_elapsed_real;


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

void work_rank_1(int SLEEP_TIME, int SIZE)
{
    int baseTag_ = 40;
//    // Init the MPI stuff
//    int rank, size;
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//    MPI_Comm_size(MPI_COMM_WORLD, &size);
//
//    double *sum_h = (double*)malloc(sizeof(double));   // host buffer
//    
//    MPI_Send(sum_h, 1, MPI_DOUBLE, 0, baseTag_ + 101, MPI_COMM_WORLD);

/*

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

    //MPI_Init(NULL, NULL);
    int provided;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
    

    // Init the MPI stuff
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int SIZE = 1000;
    //MPI_Barrier(MPI_COMM_WORLD);

    float t0_thread;
    float t1_thread;
    float TIME;

    // = = = = = = = = = = = = = =
    // int SLEEP_TIME = 1; // us
    // int SIZE = 1e4;     // doubles
    // = = = = = = = = = = = = = =

    //work(rank, SLEEP_TIME, SIZE);
    

    //int SLEEP_TIME = atoi(argv[1]);

    int min_size = 10;
    int max_size = 1000;
    int step_size = 10;
    int command_line = 1;


    //if (rank==1) {work_rank_1(0,SIZE);}
    if (rank == 0)
    {
        std::cout << 0 << "  \t";   // take outside the loop the fuesr step
        for (int i = 0; i < 10; i++)
        {
            float ratio = work_rank_0(0, SIZE);
            std::cout << "\t" << ratio;
        }
        printf("\n");
        for (int SLEEP_TIME = 1; SLEEP_TIME < 1e9; SLEEP_TIME *= 2)
        {
            std::cout << SLEEP_TIME << "  \t";
            for (int i = 0; i < 10; i++)
            {
                float ratio = work_rank_0(SLEEP_TIME, SIZE);

                std::cout << "\t" << ratio;
            }
            printf("\n");
        }
    }

    //if (rank == 1) {work_rank_1(SLEEP_TIME, SIZE);}


    MPI_Finalize();
}


