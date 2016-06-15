#include "batch_lpu_mgmt.h"
#include "lpu_management.h"
#include "structure.h"
#include "../memory-management/part_tracking.h"
#include "../memory-management/part_generation.h"
#include "../memory-management/part_management.h"
#include "../communication/communicator.h"
#include "../utils/list.h"
#include "../utils/hashtable.h"
#include "../gpu-offloader/gpu_code_executor.h"

#include <fstream>
#include <vector>

using namespace std;

//------------------------------------------------------- Batch PPU State -----------------------------------------------------------/

BatchPpuState::BatchPpuState(int lpsCount, List<ThreadState*> *ppuStateList, std::vector<int> *groupLeaderPpuCounts) {
	this->lpsCount = lpsCount;
	ppuStates = new vector<ThreadState*>;
	ppuStates->reserve(ppuStateList->NumElements());
	for (int i = 0; i < ppuStateList->NumElements(); i++) {
		ppuStates->push_back(ppuStateList->Nth(i));
	}
	this->groupLeaderPpuCounts = groupLeaderPpuCounts;
	this->loggingEnabled = false;
	this->logFile = NULL;
	initializeLpuVectors();
	gpuCodeExecutors = NULL;
}

BatchPpuState::~BatchPpuState() {
	delete ppuStates;
	while (lpuVectorsForLPSes->NumElements() > 0) {
		vector<LPU*> *lpuVector = lpuVectorsForLPSes->Nth(0);
		lpuVectorsForLPSes->RemoveAt(0);
		delete lpuVector;
	}
	delete lpuVectorsForLPSes;
}

vector<LPU*> *BatchPpuState::getNextLpus(int lpsId, int containerLpsId, std::vector<int> *currentLpuIds) {
	
	vector<LPU*> *lpuVector = lpuVectorsForLPSes->Nth(lpsId);
	int activePpuCounts = groupLeaderPpuCounts->at(lpsId);
	int activePpuGap = ppuStates->size() / activePpuCounts;
	
	bool emptyBatch = true;
	for (unsigned int i = 0; i < ppuStates->size(); i++) {
		ThreadState *threadState = ppuStates->at(i);
		int lpuGenerationStatus = ppuLpuGenerationStatus->at(i);
		if (lpuGenerationStatus == LPU_GEN_STATE_CONTRIB_ACTIVE 
				|| lpuGenerationStatus == LPU_GEN_STATE_NON_CONTRIB_ACTIVE) {
			
			int groupLeaderLpuIndex = i / activePpuGap; 
			int currentLpuId = currentLpuIds->at(groupLeaderLpuIndex);
			LPU *lpu = threadState->getNextLpu(lpsId, containerLpsId, currentLpuId);
			if (lpu != NULL) {
				if (lpuGenerationStatus == LPU_GEN_STATE_CONTRIB_ACTIVE) {
					lpuVector->at(i) = lpu;	
					emptyBatch = false;
				}
			} else {
				if (lpuGenerationStatus == LPU_GEN_STATE_CONTRIB_ACTIVE) {
					ppuLpuGenerationStatus->at(i) = LPU_GEN_STATE_CONTRIB_DEPLATED; 
					lpuVector->at(i) = NULL;	
				} else {
					ppuLpuGenerationStatus->at(i) = LPU_GEN_STATE_NON_CONTRIB_DEPLATED;
				}
			}
		} else if (lpuGenerationStatus == LPU_GEN_STATE_CONTRIB_DEPLATED) {
			ppuLpuGenerationStatus->at(i) = LPU_GEN_STATE_CONTRIB_DEPLATED; 
                        lpuVector->at(i) = NULL;
		}
	}
	if (emptyBatch) return NULL;
	return lpuVector;			
}

void BatchPpuState::initLpuIdVectorsForLPSTraversal(int lpsId, std::vector<int> *lpuIdVector) {
	
	lpuIdVector->clear();
	ppuLpuGenerationStatus->clear();

	int activePpuCounts = groupLeaderPpuCounts->at(lpsId);
	int activePpuGap = ppuStates->size() / activePpuCounts;

	for (unsigned int i = 0; i < ppuStates->size(); i++) {
		if (i % activePpuGap == 0) {
			lpuIdVector->push_back(INVALID_ID);
			ppuLpuGenerationStatus->push_back(LPU_GEN_STATE_CONTRIB_ACTIVE);	
		} else {
			ppuLpuGenerationStatus->push_back(LPU_GEN_STATE_NON_CONTRIB_ACTIVE);
		}
	}
}

bool BatchPpuState::hasValidPpus(int lpsId) {
	for (unsigned int i = 0; i < ppuStates->size(); i++) {
		ThreadState *threadState = ppuStates->at(i);
		if (threadState->isValidPpu(lpsId)) return true; 
	}
	return false;
}

void BatchPpuState::removeIterationBound(int lpsId) {
	for (unsigned int i = 0; i < ppuStates->size(); i++) {
		ppuStates->at(i)->removeIterationBound(lpsId);
	}	
}

void BatchPpuState::covertLpuVectorToList(List<LPU*> *destinationList, std::vector<LPU*> *lpuVector) {
	destinationList->clear();
	for (unsigned int i = 0; i < lpuVector->size(); i++) {
		if (lpuVector->at(i) != NULL) {
			destinationList->Append(lpuVector->at(i));
		}
	}
}

void BatchPpuState::enableLogging(std::ofstream *logFile) {
	loggingEnabled = true;
	this->logFile = logFile;
}

void BatchPpuState::extractLpuIdsFromLpuVector(std::vector<int> *idVector, std::vector<LPU*> *lpuVector) {
	idVector->clear();
	for (unsigned int i = 0; i < lpuVector->size(); i++) {
		LPU *lpu = lpuVector->at(i);
		if (lpu != NULL) {
			idVector->push_back(lpu->id);
		} else {
			idVector->push_back(INVALID_ID);
		}
	}
}

void BatchPpuState::initializeLpuVectors() {
	lpuVectorsForLPSes = new List<vector<LPU*>*>;
	for (int i = 0; i < lpsCount; i++) {
		vector<LPU*> *lpuVector = new vector<LPU*>;
		int participantsCount = groupLeaderPpuCounts->at(i);
		for (int j = 0; j < participantsCount; j++) {
			lpuVector->push_back(NULL);
		}
		lpuVectorsForLPSes->Append(lpuVector);
	}
}

