/*

    This script measures the CPU consumption inside the Iprobe loop, as function
    of the time we wait between checks

    Outputs two files. One of them contains "Thread time per iter vs SLEEP_TIME"
    and the other one "Real time per iter vs SLEEP_TIME"

    Compilation:

        nvcc -I/path/to/ompi/include -L/path/to/ompi/lib -lmpi \
             fullTest.cc cudaWrapper.cu -o main

    Runtime:

        mpirun -np 2 --hostfile hosts -x UCX_MEMTYPE_CACHE=n -x UCX_TLS=all
                    --mca pml ucx --mca btl ^openib -x UCX_RNDV_SEND_NBR_THRESH=0K
                    -x UCX_LOG_LEVEL=func main
                    
        where hosts is a file containing, for example,
        
            patatrack01.cern.ch slots=1
            felk40 slots=1
*/


#include <memory>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <mpi.h>
#include <chrono>
#include <thread>
#include "header.h"
#include "cudaCheck.h"
#include <cuda_runtime.h>
#include <vector>



std::vector<long double> work_rank_0(int SLEEP_TIME, int SIZE)
{
    int baseTag_ = 40;


    // Init the MPI stuff
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    int flag = false;   // will rise when a package is ready to be received
    int reps = 20;      // more statistics

    // nanosleep
    struct timespec t0_thread, t1_thread, elapsed_thread;  
    struct timespec t0_real, t1_real, elapsed_real;  
    struct timespec req, rem;

    req.tv_sec = 0;
    req.tv_nsec = SLEEP_TIME; // sleep time


    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t0_thread);
    clock_gettime(CLOCK_MONOTONIC, &t0_real);

    // here what we want to measure >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    for (int i = 0; i < reps; i++)
    {
        MPI_Iprobe(1, baseTag_ + 101, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
        std::this_thread::sleep_for(std::chrono::nanoseconds(SLEEP_TIME));
        //nanosleep(&req, &rem);
    }
    //MPI_Probe(1, baseTag_ + 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1_thread);
    clock_gettime(CLOCK_MONOTONIC, &t1_real);

    elapsed_thread.tv_sec = t1_thread.tv_sec - t0_thread.tv_sec; 
    elapsed_thread.tv_nsec = t1_thread.tv_nsec - t0_thread.tv_nsec; 

    elapsed_real.tv_sec = t1_real.tv_sec - t0_real.tv_sec; 
    elapsed_real.tv_nsec = t1_real.tv_nsec - t0_real.tv_nsec; 


    long double total_elapsed_thread = ((long double) 1e9 * elapsed_thread.tv_sec + elapsed_thread.tv_nsec);
    long double total_elapsed_real = ((long double) 1e9 * elapsed_real.tv_sec + elapsed_real.tv_nsec);
    
    
    std::vector<long double> times = std::vector<long double>(2);
    times[0] = total_elapsed_thread / reps;
    times[1] = total_elapsed_real / reps;

    return times;
}

void work_rank_1(int SLEEP_TIME, int SIZE)
{
    // the CPU consumption is a function of the time spending sleeping, and the
    // whole thing runs in the same process. Therefore, the measurement is
    // independent of what does the 2nd rank do.d
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


    int min_size = 10;
    int max_size = 1000;
    int step_size = 10;
    int command_line = 1;


    if (rank == 0)
    {

        std::ofstream f;
        std::ofstream g;
        f.open ("1D_thr.txt");
        g.open ("1D_rea.txt");
        printf("\n");

        f << "# thread\n";  // thread elapsed time as function of SLEEP_TIME
        g << "# real  \n";  // real elapsed time as function of SLEEP_TIME

        for (int SLEEP_TIME = 1; SLEEP_TIME < 1e6; SLEEP_TIME *= 2)
        {
            f << SLEEP_TIME << "  \t";
            g << SLEEP_TIME << "  \t";
            
            std::cout << "sleept_time: " << SLEEP_TIME << std::endl;

            for (int i = 0; i < 10; i++)
            {
                std::vector<long double> result = work_rank_0(SLEEP_TIME, SIZE);
                float ratio = result[1];

                f << "\t" << result[0];
                g << "\t" << result[1];
            }
            f << "\n";
            g << "\n";
        }
        f.close();
        g.close();
    }

    MPI_Finalize();
}


