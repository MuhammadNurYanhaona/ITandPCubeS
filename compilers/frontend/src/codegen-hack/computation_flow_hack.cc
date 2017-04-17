#include "../../../common-libs/utils/list.h"
#include "../semantics/computation_flow.h"
#include "../semantics/task_space.h"
#include "../static-analysis/sync_stat.h"
#include <fstream>

List<const char*> *FlowStage::getAllOutgoingDependencyNamesAtNestingLevel(int nestingLevel) { return NULL; }
void FlowStage::generateInvocationCode(std::ofstream &stream, int indentation, Space *containerSpace) {}

void StageInstanciation::translateCode(std::ofstream &stream) {}
void StageInstanciation::generateInvocationCode(std::ofstream &stream, int indentation, Space *containerSpace) {}

void CompositeStage::generateInvocationCode(std::ofstream &stream, int indentation, Space *containerSpace) {}
List<const char*> *CompositeStage::getAllOutgoingDependencyNamesAtNestingLevel(int nestingLevel) { return NULL; }
void CompositeStage::declareSynchronizationCounters(std::ofstream &stream, int indentation, int nestingLevel) {}
void CompositeStage::generateDataReceivesForGroup(std::ofstream &stream, 
		int indentation, List<SyncRequirement*> *commDependencies) {}
void CompositeStage::genSimplifiedWaitingForReactivationCode(std::ofstream &stream, 
		int indentation, List<SyncRequirement*> *syncRequirements) {}
void CompositeStage::genSimplifiedSignalsForGroupTransitionsCode(std::ofstream &stream, int indentation,
		List<SyncRequirement*> *syncRequirements) {}
void CompositeStage::generateDataSendsForGroup(std::ofstream &stream, int indentation,
		List<SyncRequirement*> *commRequirements) {}

void RepeatControlBlock::generateInvocationCode(std::ofstream &stream, int indentation, Space *containerSpace) {}
void ConditionalExecutionBlock::generateInvocationCode(std::ofstream &stream, int indentation, Space *containerSpace) {}

void LpsTransitionBlock::genReductionResultPreprocessingCode(std::ofstream &stream, int indentation) {}
void LpsTransitionBlock::generateInvocationCode(std::ofstream &stream, int indentation, Space *containerSpace) {}

void EpochBoundaryBlock::genCodeForScalarVarEpochUpdates(std::ofstream &stream, 
		int indentation,
		List<const char*> *scalarVarList) {}
void EpochBoundaryBlock::genCodeForArrayVarEpochUpdates(std::ofstream &stream,
		const char *affectedLpsName,
		int indentation,
		List<const char*> *arrayVarList) {}
void EpochBoundaryBlock::genLpuTraversalLoopBegin(std::ofstream &stream, const char *lpsName, int indentation) {}
void EpochBoundaryBlock::genLpuTraversalLoopEnd(std::ofstream &stream, const char *lpsName, int indentation) {}
void EpochBoundaryBlock::generateInvocationCode(std::ofstream &stream, int indentation, Space *containerSpace) {}

void ReductionBoundaryBlock::generateInvocationCode(std::ofstream &stream, int indentation, Space *containerSpace) {}
