// system include files
#include <memory>
#include <iostream>
#include <mpi.h>

// user include files
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/StreamID.h"


extern void LOG(std::string message, int t);

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
    
};


NumberAccS::NumberAccS(const edm::ParameterSet& config)
{
    LOG("[NumberAccS::NumberAccS]:  initializing MPI enviroment...", 0);
    MPI_Init(NULL, NULL);
    produces<std::vector<double>>(); // only consisting on one element!
    LOG("[NumberAccS::NumberAccS]:  done", 0);
}
NumberAccS::~NumberAccS()
{
    LOG("[NumberAccS::~NumberAccS]:  destruting...", 0);
    MPI_Finalize();
}

void
NumberAccS::produce(edm::Event& event, const edm::EventSetup& setup)
{
    // Init the MPI stuff
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    LOG("[NumberAccS::produce]:  RANK: " + std::to_string(rank), 1);
    LOG("[NumberAccS::produce]:  SIZE: " + std::to_string(size), 1);
    // Only rank 1 is supposed to be here
    if (rank == 1)
    {
        int len;

        // Recv the len of the buffer
        MPI_Recv(&len, 1, MPI_INT, 0, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        LOG("[NumberAccS::produce]:  len Received", 0);

        // Allocate the memory
        auto input = std::make_unique<double[]>(len);

        // Recv the actual buffer
        MPI_Recv(&input[0], len, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        LOG("[NumberAccS::produce]:  data Received", 0);

        // sum all the input elements 
        double sum = 0;
        for (int i = 0; i < len; i++)
            sum += input[i];

        // send the sum back to the offloader
        MPI_Ssend(&sum, 1, MPI_DOUBLE, 0, 101, MPI_COMM_WORLD);
        LOG("[NumberAccS::produce]:  result Sent", 0);
    }
}

void
NumberAccS::beginStream(edm::StreamID)
{
}

void
NumberAccS::endStream() {
}

void
NumberAccS::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // NumberAccS
  edm::ParameterSetDescription desc;
  descriptions.add("numberAccS", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(NumberAccS);

