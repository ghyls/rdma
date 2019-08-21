import FWCore.ParameterSet.Config as cms


process = cms.Process("TEST")
process.MPIService  = cms.Service("MPIService")

#process.Tracer = cms.Service("Tracer")


# All the objects in the chain 
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# The data is produced from scratch on the producer.
process.source = cms.Source("EmptySource")

# Edit variables called from NumberProducer::NumberProducer
process.load('HeterogeneousCore.NumberProducer.numberProducer_cfi')
process.numberProducer.size = 10
process.numberProducer.seed = 65

# There is no need to do anything on the Accumulator
process.load('HeterogeneousCore.NumberProducer.numberAccumulator_cfi')
process.numberAccumulator.data = 'numberProducer'

# Edit variables called from numberOffloader::numberOffloader
process.load('HeterogeneousCore.NumberProducer.numberOffloader_cfi')
process.numberOffloader.data = 'numberProducer'
process.numberOffloader.baseTag = 42


# Edit variables called from NumberLogger::NumberLogger
process.load('HeterogeneousCore.NumberProducer.numberLogger_cfi')
process.numberLogger.data = 'numberOffloader'

# We can create another logger
process.SerialNumberLogger = process.numberLogger.clone()
process.SerialNumberLogger.data = 'numberAccumulator'
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# Tasks and paths 
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# create the first task
process.task = cms.Task( process.numberProducer )

# Append elements to the task. 
# Note that process.* do not define a commutative group under "+"!

## Serial Path:
#process.path1 = cms.Path( process.numberAccumulator + process.SerialNumberLogger, 
#                            process.task )

# Parallel Path:
process.path2 = cms.Path( process.numberOffloader + 
                           process.numberLogger, process.task )
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
