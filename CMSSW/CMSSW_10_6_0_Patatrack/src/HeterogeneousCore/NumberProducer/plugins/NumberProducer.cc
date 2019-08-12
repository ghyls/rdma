// system include files
#include <memory>
#include <iostream>

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

class NumberProducer : public edm::stream::EDProducer<> {
public:
    explicit NumberProducer(const edm::ParameterSet& config);
    ~NumberProducer() = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
    virtual void beginStream(edm::StreamID) override;
    virtual void produce(edm::Event& event, const edm::EventSetup& setup) override;
    virtual void endStream() override;

    const uint32_t size_;
    const uint32_t seed_;

};

NumberProducer::NumberProducer(const edm::ParameterSet& config) :
   size_(config.getParameter<uint32_t>("size")),
   seed_(config.getParameter<uint32_t>("seed"))
{
   //register your products
   produces<std::vector<double>>();

   //now do what ever other initialization is needed
}


void
NumberProducer::produce(edm::Event& event, const edm::EventSetup& setup)
{
    // Create the data, initialized to zeros
    auto data = std::make_unique<std::vector<double>>(size_);

    // plant the seed
    srand(seed_);

    // init it with a random seed
    for (uint32_t i = 0; i < size_; i++) 
    {
        (*data)[i] = (double) rand() / RAND_MAX;
    }

    event.put(std::move(data));
}

void
NumberProducer::beginStream(edm::StreamID)
{
}

void
NumberProducer::endStream() {
}

void
NumberProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // numberProducer
  edm::ParameterSetDescription desc;
  desc.add<unsigned int>("size", 32);
  desc.add<unsigned int>("seed", 32);
  descriptions.add("numberProducer", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(NumberProducer);
