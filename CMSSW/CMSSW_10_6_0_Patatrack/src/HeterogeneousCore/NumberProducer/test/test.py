import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
#process.Tracer = cms.Service("Tracer"


# All the objects in the chain 
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# The data is produced from scratch on the producer.
process.source = cms.Source("EmptySource")

# Edit variables called from NumberProducer::NumberProducer
process.load('HeterogeneousCore.NumberProducer.numberProducer_cfi')
process.numberProducer.size = 4
process.numberProducer.seed = 65

# There is no need to do anything on the Gatherer
process.load('HeterogeneousCore.NumberProducer.numberGatherer_cfi')
process.numberGatherer.data = 'numberProducer'

# Edit variables called from NumberLogger::NumberLogger
process.load('HeterogeneousCore.NumberProducer.numberLogger_cfi')
process.numberLogger.data = 'numberGatherer'

# We can create another logger
process.otherLogger = process.numberLogger.clone()
process.otherLogger.data = 'numberProducer'
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# Tasks and paths 
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# create the first task
process.task = cms.Task( process.numberProducer )

# Append elements to the task. 
# Note that process.* do not define a commutative group under "+"!
process.path1 = cms.Path(  process.numberGatherer + process.numberLogger, process.task )
#process.path2 = cms.Path( process.otherLogger, process.task )
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# Extra options
process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32(0),
    numberOfStreams = cms.untracked.uint32(1),
    wantSummary = cms.untracked.bool(True)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)
