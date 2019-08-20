// system include files
#include <memory>
#include <iostream>
#include <mpi.h>
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


extern void LOG(std::string message, int t);

// class declaration

class NumberOffloader : public edm::stream::EDProducer<edm::ExternalWork> {
public:
    explicit NumberOffloader(const edm::ParameterSet& config);
    //~NumberOffloader() = default;
    ~NumberOffloader();

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
    void beginStream(edm::StreamID) override;
    void acquire(edm::Event const& event, edm::EventSetup const& setup, edm::WaitingTaskWithArenaHolder holder) override; 
    void produce(edm::Event& event, const edm::EventSetup& setup) override;
    void endStream() override;

    const edm::EDGetTokenT<std::vector<double>> data_;
    const uint32_t baseTag_;
};

NumberOffloader::NumberOffloader(const edm::ParameterSet& config) :
    data_(consumes<std::vector<double>>(config.getParameter<edm::InputTag>("data"))),
    baseTag_(config.getParameter<uint32_t>("baseTag"))
{
    produces<std::vector<double>>();
    LOG("[NumberOffloader::NumberOffloader]:  Constructor called.", 1);
}
NumberOffloader::~NumberOffloader()
{
    LOG("[NumberOffloader::~NumberOffloader]:  Destructor called.", 1);
}


void
NumberOffloader::acquire(edm::Event const& event, edm::EventSetup const& setup, edm::WaitingTaskWithArenaHolder holder)
{
    // Init the MPI stuff
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    LOG("[NumberOffloader::acquire]:  RANK: " + std::to_string(rank) + 
                                        "; SIZE: " + std::to_string(size), 1);

    // read from the NumberProducer
    edm::Handle<std::vector<double>> handle;
    event.getByToken(data_, handle);
    auto const& data = * handle;

    // send the vector to the accumulator (rank of the sender is 0)
    LOG("[NumberOffloader::acquire]:  sending the data to the Accumulator", 1);    
    MPI_Send(data.data(), data.size(), MPI_DOUBLE, 1, baseTag_ + 98, MPI_COMM_WORLD);
    LOG("[NumberOffloader::acquire]:  data sent!", 1);


    auto task = [holder](uint32_t baseTag_){
        // Probing for incoming buffers
        LOG("[NumberOffloader::acquire]:  Waiting for the result of the Accumulator", 1);
        int flag = false;
        while (not flag)
        {
            MPI_Iprobe(1, baseTag_ + 101, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
    };
    std::thread thr(task, baseTag_);
    thr.detach();
        
    //int flag = false;
    //while (not flag)
    //{
    //    MPI_Iprobe(1, baseTag_ + 101, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
    //    std::this_thread::sleep_for(std::chrono::microseconds(1));
    //}
    
    LOG("[NumberOffloader::acquire]:  main thread finished", 1);
}


void
NumberOffloader::produce(edm::Event& event, const edm::EventSetup& setup)
{
    LOG("[NumberOffloader::produce]:  starting", 1);

    int len = 0;
    MPI_Status status;

    MPI_Probe(1, baseTag_ + 101, MPI_COMM_WORLD, &status);
    MPI_Get_count(&status, MPI_DOUBLE, &len);
    LOG("[NumberOffloader::produce]:  found MPI_Send, pkg_length = " + std::to_string(len), 1);

    // Create the result vector (will be filled with Server's output)
    auto result = std::make_unique<std::vector<double>>(len);

    // recive the result from the accumulator
    MPI_Recv(result->data(), 1, MPI_DOUBLE, 1, baseTag_ + 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    LOG("[NumberOffloader::produce]:  result received!", 1);

    event.put(std::move(result));    
    
    //(*result)[0] = 33;
    //event.put(std::move(result));
}

void
NumberOffloader::beginStream(edm::StreamID){
}

void
NumberOffloader::endStream() {
}

void
NumberOffloader::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // NumberOffloader
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("data", { "numberProducer" });
  desc.add<unsigned int>("baseTag", 32);  
  descriptions.add("numberOffloader", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(NumberOffloader);

