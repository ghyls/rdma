// system include files
#include <iostream>
#include <vector>
#include <string>

// user include files
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/StreamID.h"


// class declaration
class NumberLogger : public edm::global::EDAnalyzer<> {
public:
    explicit NumberLogger(const edm::ParameterSet& config);
    ~NumberLogger() = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
    virtual void analyze(edm::StreamID sid, const edm::Event& event, const edm::EventSetup& setup) const override;

    const edm::EDGetTokenT<std::vector<double>> data_;

};


void LOG(std::string message, int t)
{
    /*
    t  ==  0 -> info  (white)
    t  ==  1 -> debug (blue)
    t  == -1 -> error (red)
    */

    bool info  = 1;
    bool debug = 1;
    bool error = 1;

    switch (t)
    {
        case -1: if (error) {std::cout << "\033[1;31m" << message << "\033[0m" << std::endl;} ; break;
        case  0: if (info)  {std::cout << "\033[0;32m" << message << "\033[0m" << std::endl;} ; break;
        case  1: if (debug) {std::cout << "\033[1;34m" << message << "\033[0m" << std::endl;} ; break;

        default:
            break;
    }
}


NumberLogger::NumberLogger(const edm::ParameterSet& config) :
    data_(consumes<std::vector<double>>(config.getParameter<edm::InputTag>("data")))
{
    LOG("[NumberLogger::NumberLogger]:  Constructor called", 0);
    // now do what ever other initialization is needed
}


void
NumberLogger::analyze(edm::StreamID sid, const edm::Event& event, const edm::EventSetup& setup) const
{
    edm::Handle<std::vector<double>> handle;
    event.getByToken(data_, handle);
    auto const& data = * handle;

    for (double item: data)
        std::cout << item << " ";
    printf("\n");
}

void
NumberLogger::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // numberLogger
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("data", { "numberProducer", "numberAccumulator" });
  descriptions.add("numberLogger", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(NumberLogger);
