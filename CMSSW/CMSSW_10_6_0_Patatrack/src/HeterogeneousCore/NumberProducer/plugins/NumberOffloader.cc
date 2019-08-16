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

class NumberOffloader : public edm::stream::EDProducer<> {
public:
    explicit NumberOffloader(const edm::ParameterSet& config);
    //~NumberOffloader() = default;
    ~NumberOffloader();

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
    virtual void beginStream(edm::StreamID) override;
    virtual void produce(edm::Event& event, const edm::EventSetup& setup) override;
    virtual void endStream() override;

    const edm::EDGetTokenT<std::vector<double>> data_;
};

NumberOffloader::NumberOffloader(const edm::ParameterSet& config) :
    data_(consumes<std::vector<double>>(config.getParameter<edm::InputTag>("data")))
{
    produces<std::vector<double>>();
    LOG("[NumberOffloader::NumberOffloader]:  Constructor called.", 1);
}
NumberOffloader::~NumberOffloader()
{
    LOG("[NumberOffloader::~NumberOffloader]:  Destructor called.", 1);
}

void
NumberOffloader::produce(edm::Event& event, const edm::EventSetup& setup)
{
    // Init the MPI stuff
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    LOG("[NumberOffloader::produce]:  RANK: " + std::to_string(rank) + 
                                        "; SIZE: " + std::to_string(size), 1);


    // read from the NumberProducer
    edm::Handle<std::vector<double>> handle;
    event.getByToken(data_, handle);
    auto const& data = * handle;

    // store the length of the package
    int len = data.size();

    // Create the result vector (will be filled with Server's output)
    auto result = std::make_unique<std::vector<double>>(1);

    // send the vector to the accumulator (rank of the sender is 0)
    LOG("[NumberOffloader::produce]:  sending the data too the Accumulator", 1);    
    MPI_Ssend(data.data(), len, MPI_DOUBLE, 1, 98, MPI_COMM_WORLD);
    LOG("[NumberOffloader::produce]:  data sent!", 1);

    // recive the result from the accumulator
    LOG("[NumberOffloader::produce]:  Waiting for the result of the Accumulator", 1);
    MPI_Recv(result->data(), 1, MPI_DOUBLE, 1, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    LOG("[NumberOffloader::produce]:  result received!", 1);

    event.put(std::move(result));    
    
    //(*result)[0] = 33;
    //event.put(std::move(result));
}

void
NumberOffloader::beginStream(edm::StreamID)
{
}

void
NumberOffloader::endStream() {
}

void
NumberOffloader::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // NumberOffloader
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("data", { "numberProducer" });
  descriptions.add("numberOffloader", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(NumberOffloader);

