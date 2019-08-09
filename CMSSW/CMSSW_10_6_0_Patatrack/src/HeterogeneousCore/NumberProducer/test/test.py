import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

#process.Tracer = cms.Service("Tracer")

process.source = cms.Source("EmptySource")

process.load('HeterogeneousCore.NumberProducer.numberProducer_cfi')
process.numberProducer.size = 4

process.load('HeterogeneousCore.NumberProducer.numberLogger_cfi')
process.numberLogger.data = 'numberProducer'

process.otherLogger = process.numberLogger.clone()
process.otherLogger.data = 'numberProducer'
 
process.task = cms.Task( process.numberProducer )

process.path1 = cms.Path( process.numberLogger, process.task )
process.path2 = cms.Path( process.otherLogger, process.task )

process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32(2),
    numberOfStreams = cms.untracked.uint32(1),
    wantSummary = cms.untracked.bool(True)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)
