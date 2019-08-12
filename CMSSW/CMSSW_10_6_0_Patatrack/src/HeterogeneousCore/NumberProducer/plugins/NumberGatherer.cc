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

class NumberGatherer : public edm::stream::EDProducer<> {
public:
    explicit NumberGatherer(const edm::ParameterSet& config);
    ~NumberGatherer() = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
    virtual void beginStream(edm::StreamID) override;
    virtual void produce(edm::Event& event, const edm::EventSetup& setup) override;
    virtual void endStream() override;

    const edm::EDGetTokenT<std::vector<double>> data_;
};

NumberGatherer::NumberGatherer(const edm::ParameterSet& config) :
    data_(consumes<std::vector<double>>(config.getParameter<edm::InputTag>("data")))
{
   
   produces<std::vector<double>>(); // only consisting on one element!

}


void
NumberGatherer::produce(edm::Event& event, const edm::EventSetup& setup)
{
    // read from the NumberProducer
    edm::Handle<std::vector<double>> handle;
    event.getByToken(data_, handle);
    auto const& data = * handle;

    // sum all the elements in data
    double sum = 0;
    for (double item: data)
        sum += item;

    // Create the output vector
    auto result = std::make_unique<std::vector<double>>(1); // only one element!

    // fill it with sum
    (*result)[0] = (double) sum;
    
    event.put(std::move(result));
}

void
NumberGatherer::beginStream(edm::StreamID)
{
}

void
NumberGatherer::endStream() {
}

void
NumberGatherer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // NumberGatherer
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("data", { "numberProducer" });
  descriptions.add("numberGatherer", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(NumberGatherer);

