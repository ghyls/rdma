/*

This script runs on the server. It receives the data from the client and runs
with it a simple kernel (defined in cudaWrapper.cu). Then it sends the result
back to the NumberOffloader (which runs on the client)

It can receive the result directly on the GPU, follow the path H → H → D or just
receive the package on the host and compute the result there (see the flags in
the code).

*/


// system include files
#include <memory>
#include <iostream>
#include <mpi.h>
#include <cuda.h>
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "header.h"
#include <chrono>
#include <thread>

// user include files
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/StreamID.h"



// class declaration

class NumberAccS : public edm::stream::EDProducer<> {
public:
    explicit NumberAccS(const edm::ParameterSet& config);
    //~NumberAccS() = default;
    ~NumberAccS();

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
    virtual void beginStream(edm::StreamID) override;
    virtual void produce(edm::Event& event, const edm::EventSetup& setup) override;
    virtual void endStream() override;
    
    const uint32_t baseTag_;
};



NumberAccS::NumberAccS(const edm::ParameterSet& config) :
    baseTag_(config.getParameter<uint32_t>("baseTag"))
{
    LOG("[NumberAccS::NumberAccS]:  Constructor called.", 1);
    produces<std::vector<double>>();
}
NumberAccS::~NumberAccS()
{
    LOG("[NumberAccS::~NumberAccS]:  Destructor called.", 1);
}

void
NumberAccS::produce(edm::Event& event, const edm::EventSetup& setup)
{
    // Init the MPI stuff
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    LOG("[NumberOffloader::produce]:  RANK: " + std::to_string(rank) + 
                                        "; SIZE: " + std::to_string(size), 1);


    int len; // length of the data package

    MPI_Status status;

    LOG("[NumberAccS::produce]:  Probing the incoming buffer", 1);
    MPI_Probe(0, baseTag_ + 98, MPI_COMM_WORLD, &status);

    LOG("[NumberAccS::produce]:  reading the length of the buffer", 1);
    MPI_Get_count(&status, MPI_DOUBLE, &len);

    bool useGPU    = true;
    bool GPUDirect = true;

    double *input;
    if (!GPUDirect)
    {
        LOG("[NumberAccS::produce]:  H -> H -> D.", 0);   

        input = (double*)malloc(len * sizeof(double));   // host buffer

        MPI_Recv(&input[0], len, MPI_DOUBLE, 0, baseTag_ + 98, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        LOG("[NumberAccS::produce]:  data Received", 1);   
    }

    // this is the variable on the host where the result will be stored
    double *sum_h = (double*)malloc(sizeof(double));   // host buffer
    if (useGPU)
    {
        // do the computation in a CUDA kernel
        LOG("[NumberAccS::produce]:  The sum will be computed on the GPU", 1);

        // The array to be summed will be stored here on the GPU
        double *buf_d;
        cudaCheck( cudaMalloc((void **) &buf_d, len * sizeof(double)) );

        // and this will be its sum also on the GPU
        double *sum_d;
        cudaCheck( cudaMalloc((void **) &sum_d, sizeof(double)) );

        if (GPUDirect){
            LOG("[NumberAccS::produce]:  H -> GPU (RDMA).", 0);
            LOG("[NumberAccS::produce]:  Receiving the package on the GPU", 1);
            MPI_Recv(buf_d, len, MPI_DOUBLE, 0, baseTag_ + 98, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            LOG("[NumberAccS::produce]:  Received the package on the GPU", 1);
        }
        else{
            // Move the host array to the GPU
            LOG("[NumberAccS::produce]: Copying the package H -> D", 1);
            cudaMemcpy(buf_d, input, len * sizeof(double), cudaMemcpyHostToDevice);            
        }

        // Pass sum_h to the wrapper, so it can change its value to the actual
        // sum (which is what we want)
        cudaWrapper(buf_d, sum_d, len);

        if (GPUDirect)
        {
            LOG("[NumberAccS::produce]:  Sending the result (DtoH)", 1);
            MPI_Ssend(sum_d, 1, MPI_DOUBLE, 0, baseTag_ + 101, MPI_COMM_WORLD);
            LOG("[NumberAccS::produce]:  result Sent (DtoH)", 1);
        }
        else
        {
            cudaCheck( cudaMemcpy(sum_h, sum_d, sizeof(double), cudaMemcpyDeviceToHost) );    
            LOG("[NumberAccS::produce]:   Result: " + std::to_string(sum_h[0]), 2);
            
            // send the sum back to the offloader
            MPI_Ssend(sum_h, 1, MPI_DOUBLE, 0, baseTag_ + 101, MPI_COMM_WORLD);
            LOG("[NumberAccS::produce]:  result Sent (HtoH)", 1);
        }

        cudaFree(buf_d);
        cudaFree(sum_d);
    }
    else
    {
        LOG("[NumberAccS::produce]:  The computation is about to be done using "
                                    "only the CPU.", 0);
        // sum all the input elements 
        sum_h[0] = 0;
        for (int i = 0; i < len; i++)
            sum_h[0] += input[i];

        MPI_Ssend(sum_h, 1, MPI_DOUBLE, 0, baseTag_ + 101, MPI_COMM_WORLD);
        LOG("[NumberAccS::produce]:  result Sent (HtoH)", 1);
    }

}

void
NumberAccS::beginStream(edm::StreamID){
}

void
NumberAccS::endStream() {
}

void
NumberAccS::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // NumberAccS
  edm::ParameterSetDescription desc;
  desc.add<unsigned int>("baseTag", 32);  
  descriptions.add("numberAccS", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(NumberAccS);

