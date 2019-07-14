#include <iostream>
#include "mpi.h"

//  A   ->  Felk (master)
//  B   ->  Patatrack

int main(int argc, char* argv[])
{

    int MAX_SIZE =    1e7;      // max memory, allocated for sending / receiving
    int NUM_ROUNDS =  100;      // number of operations per sincreasing size

    int        rank, size;
    double     t_0, t_1;

    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    if (size != 2) {
        if (rank == 0) printf("This benchmark should be run on exactly two processes");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0)
    {
        printf("# MPI ping-pong test, %d rounds.\n\n", NUM_ROUNDS);
    }
    if (rank == 0)
        std::cout << "#size (MB)" << "\ttime Send AB (s)" << "\ttime BA (s)"
                                        << "\ttime A->B->A (s)" << std::endl;
    
    for (int numberOfInts = 25000000; numberOfInts < 250000000; numberOfInts += 2250000)
    //for (int i = 0; i < 100; i++)
    {
        //int numberOfInts = 50000;
        //int* buff = new int[numberOfInts];
        
        int sizeOfBuff = numberOfInts * sizeof(int);
        int* buff = (int *) malloc(sizeOfBuff);

        //std::cout << 1 % 2 << std::endl;

        float t_AtoB = 0, t_BtoA = 0;
        float t_SendAB = 0, t_SendBA = 0;

        // Perform NUM_ROUNDS transactions
        MPI_Barrier(MPI_COMM_WORLD);        // wait for everyone
        float t_beg = MPI_Wtime();          // start transaction  
        

        int partner_rank = (rank + 1) % 2;
        for (int round = 0; round < NUM_ROUNDS; round++)
        // round -> one direction single transfer
        {
            //MPI_Barrier(MPI_COMM_WORLD);    // wait for everyone
            t_0 = MPI_Wtime();
            //std::chrono::high_resolution_clock::time_point t_0 = high_resolution_clock::now();
            
            if (rank == round % 2) // 0, 1, 0, 1, ... sender, A
            {
                MPI_Ssend(buff, numberOfInts, MPI_INT, partner_rank, 0, MPI_COMM_WORLD);
                //std::cout << "sent" << std::endl;   
            }
            else {
                MPI_Recv(buff, numberOfInts, MPI_INT, partner_rank, 0, MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE);
                //std::cout << "received" << std::endl;
            }
            // MPI_Barrier(MPI_COMM_WORLD);
            t_1 = MPI_Wtime();

            
            if (rank == 0)  // felk
            {
                if ((round % 2)) { // 1, 3, 5...
                    //t_AtoB += t_1 - t_0;
                    t_SendBA += t_1 - t_0;
                    }
                else { // 0, 2, 4...
                    //t_BtoA += t_1 - t_0;
                    t_SendAB += t_1 - t_0;
                    }
            }
            else if (rank == 1) // patatrack
            {
                if ((round % 2)) { // 1, 3, 5...
                    //t_AtoB += t_1 - t_0;
                    t_SendBA += t_1 - t_0;
                    }
                else { // 0, 2, 4...
                    //t_BtoA += t_1 - t_0;
                    t_SendAB += t_1 - t_0;
                    }
            }
            
            
            //if (rank == 0) std::cout << t_AtoB << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        float t_end = MPI_Wtime();           // stop transaction

        t_SendAB = t_SendAB / (NUM_ROUNDS / 2);
        t_SendBA = t_SendBA / (NUM_ROUNDS / 2);
        float t_AtoBtoA = (t_end - t_beg) / (NUM_ROUNDS/2);
        
        if (rank == 1)
        {
            MPI_Send(&t_SendAB, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(&t_SendBA, 1, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
            MPI_Send(&t_AtoBtoA, 1, MPI_FLOAT, 0, 2, MPI_COMM_WORLD);
        }

        if (rank == 0)
        {
            float t_SendAB_Aux;
            float t_SendBA_Aux;
            float t_AtoBtoA_Aux;

            MPI_Recv(&t_SendAB_Aux, 1, MPI_FLOAT, 1, 0, MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE);
            MPI_Recv(&t_SendBA_Aux, 1, MPI_FLOAT, 1, 1, MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE);
            MPI_Recv(&t_AtoBtoA_Aux, 1, MPI_FLOAT, 1, 2, MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE);

            t_SendAB = (t_SendAB + t_SendAB_Aux) / 2;
            t_SendBA = (t_SendBA + t_SendBA_Aux) / 2;
            t_AtoBtoA = (t_AtoBtoA + t_AtoBtoA_Aux) / 2;
            float deliveredSize = sizeOfBuff * 1e-6; // in MB
            
            //std::cout << "time per transmission: " << t_AtoBtoA << " s" << std::endl;
            //std::cout << deliveredSize * 1e-6 << " Mb" << std::endl;
            //std::cout << "speed " << deliveredSize * 1e-6 / t_AtoBtoA <<  " Mb/s " << std::endl;
            //std::cout << deliveredSize << "\t" << t_AtoB << "\t" << 
            //                        t_BtoA << "\t" << t_AtoBtoA << std::endl;
            printf("%.9f \t %.9f \t %.9f \t %.9f \n", deliveredSize, t_SendAB, t_SendBA, t_AtoBtoA);
        }
    }
    
	MPI_Finalize();
	
    return 0;
}

