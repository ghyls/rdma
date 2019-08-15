import FWCore.ParameterSet.Config as cms


process = cms.Process("TEST")

process.MPIService  = cms.Service("MPIService")
#process.Tracer = cms.Service("Tracer")


# All the objects in the chain 
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# The data is produced from scratch on the producer.
process.source = cms.Source("EmptySource")

# We only load the accumulator here
process.load('HeterogeneousCore.NumberProducer.numberAccS_cfi')
#process.numberLogger.data = 'numberGatherer'

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# Tasks and paths 
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
process.path1 = cms.Path( process.numberAccS)
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# Extra options
process.options = cms.untracked.PSet(
    numberOfStreams = cms.untracked.uint32(0),
    numberOfThreads = cms.untracked.uint32(1),
    wantSummary = cms.untracked.bool(True)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)
