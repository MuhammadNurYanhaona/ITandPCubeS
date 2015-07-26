#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <pthread.h>
#include <cstdlib>

#include "lpu_management.h"
#include "structure.h"
#include "../utils/list.h"
#include "../utils/hashtable.h"
#include "../utils/interval_utils.h"
#include "../memory-management/allocation.h"
#include "../memory-management/part_tracking.h"
#include "../memory-management/part_generation.h"

/*************************************************  LPU Counter  ****************************************************/

LpuCounter::LpuCounter() {
	lpsDimensions = 0;
	lpuCounts = NULL;
	lpusUnderDimensions = NULL;
	currentLpuId = NULL;
	currentRange = NULL;
	currentLinearLpuId = INVALID_ID;
}

LpuCounter::LpuCounter(int lpsDimensions) {
	
	this->lpsDimensions = lpsDimensions;
	
	lpuCounts = new int[lpsDimensions];
	lpusUnderDimensions = new int[lpsDimensions];
	currentLpuId = new int[lpsDimensions];
	currentLinearLpuId = INVALID_ID;

	currentRange = new LpuIdRange;
	currentRange->startId = INVALID_ID;
	currentRange->endId = INVALID_ID;
}

void LpuCounter::setLpuCounts(int lpuCounts[]) {
	for (int i = 0; i < lpsDimensions; i++) {		
		this->lpuCounts[i] = lpuCounts[i];
	}
	int lpusUnderDim = 1;
	lpusUnderDimensions[lpsDimensions - 1] = 1;
	for (int j = lpsDimensions - 2; j >= 0; j--) {
		lpusUnderDim *= lpuCounts[j + 1];
		lpusUnderDimensions[j] = lpusUnderDim;
	} 
}

void LpuCounter::setCurrentRange(PPU_Ids ppuIds) {	
	int totalLpus = 1;
	for (int i = 0; i < lpsDimensions; i++) {
		totalLpus *= lpuCounts[i];
	}

	int ppuCount = ppuIds.ppuCount;
	int groupId = ppuIds.groupId;
	if (ppuCount > totalLpus) {
		if (groupId < totalLpus) {
			currentRange->startId = groupId;
			currentRange->endId = groupId;
		} else {
			currentRange->startId = INVALID_ID;
			currentRange->endId = INVALID_ID;
		}
	} else {
		int lpusPerPpu = totalLpus / ppuCount;
		int extraLpus = totalLpus % ppuCount;
		currentRange->startId = lpusPerPpu * groupId;
		currentRange->endId = currentRange->startId + lpusPerPpu - 1;
		if (groupId == ppuCount - 1) {
			currentRange->endId = currentRange->endId + extraLpus; 
		}
	}
}

int *LpuCounter::setCurrentCompositeLpuId(int linearId) {
	int remaining = linearId;
	for (int i = 0; i < lpsDimensions - 1; i++) {
		currentLpuId[i] = remaining / lpusUnderDimensions[i];
		remaining = remaining % lpusUnderDimensions[i];
	}
	currentLpuId[lpsDimensions - 1] = remaining;
	currentLinearLpuId = linearId;
	return currentLpuId;
}

int *LpuCounter::copyCompositeLpuId() {
	int *copy = new int[lpsDimensions];
	for (int i = 0; i < lpsDimensions; i++) {
		copy[i] = currentLpuId[i];
	}
	return copy;
} 

int LpuCounter::getNextLpuId(int previousLpuId) {
	if (previousLpuId == INVALID_ID) {
		return currentRange->startId;
	} else {
		int candidateNext = previousLpuId + 1;
		if (candidateNext > currentRange->endId) return INVALID_ID;
		else return candidateNext;
	}
}

void LpuCounter::resetCounter() {
	currentRange->startId = INVALID_ID;
	currentRange->endId = INVALID_ID;
	currentLinearLpuId = INVALID_ID;
}

void LpuCounter::logLpuRange(std::ofstream &log, int indent) {
	for (int i = 0; i < indent; i++) log << '\t';
	log << "LPU ID Range: ";
	log << currentRange->startId;
	log << " -- " << currentRange->endId;
	log << std::endl;	
}

void LpuCounter::logLpuCount(std::ofstream &log, int indent) {
	for (int i = 0; i < indent; i++) log << '\t';
	log << "LPU Count: ";
	for (int i = 0; i < lpsDimensions; i++) {
		if (i > 0) log << " * ";		
		log << lpuCounts[i];
	}
	log << std::endl;
}

void LpuCounter::logCompositeLpuId(std::ofstream &log, int indent) {
	for (int i = 0; i < indent; i++) log << '\t';
	log << "Composite ID: ";
	for (int i = 0; i < lpsDimensions; i++) {
		if (i > 0) log << " * ";		
		log << currentLpuId[i];
	}
	log << std::endl;
}

/********************************************** Mock Lpu Counter  ***************************************************/

MockLpuCounter::MockLpuCounter(PPU_Ids ppuIds) : LpuCounter() {
	active = (ppuIds.groupId == 0); 
}

int MockLpuCounter::getNextLpuId(int previousLpuId) {
	if (!active) return INVALID_ID;
	if (previousLpuId == INVALID_ID) return 0;
	return INVALID_ID;
}

int *MockLpuCounter::setCurrentCompositeLpuId(int linearId) { 
	currentLinearLpuId = linearId;
	return &currentLinearLpuId; 
}

int *MockLpuCounter::copyCompositeLpuId() {
	int *copy = new int;
	copy[0] = currentLinearLpuId;
	return copy;
}

void MockLpuCounter::logLpuCount(std::ofstream &log, int indent) {
	for (int i = 0; i < indent; i++) log << '\t';
	log << "Mock LPU Count: 1\n";
}

void MockLpuCounter::logCompositeLpuId(std::ofstream &log, int indent) {
	for (int i = 0; i < indent; i++) log << '\t';
	log << "Mock Composite ID: 0\n";
}


/**************************************************  LPS State  *****************************************************/

LpsState::LpsState(int lpsDimensions, PPU_Ids ppuIds) {
	if (lpsDimensions == 0) {
		counter = new MockLpuCounter(ppuIds);
	} else {
		counter = new LpuCounter(lpsDimensions);
	}
	iterationBound = false;
	lpu = NULL;
}

LPU *LpsState::getCurrentLpu(bool allowInvalid) {
	if (allowInvalid) return lpu;
	if (lpu != NULL && lpu->isValid()) {
		return lpu;
	} 
	return NULL; 
}


/************************************************  Thread State  ****************************************************/

ThreadState::ThreadState(int lpsCount, int *lpsDimensions, int *partitionArgs, ThreadIds *threadIds) {
	this->lpsCount = lpsCount;
	lpsStates = new LpsState*[lpsCount];
	for (int i = 0; i < lpsCount; i++) {
		lpsStates[i] = new LpsState(lpsDimensions[i], threadIds->ppuIds[i]);
	}
	lpsParentIndexMap = NULL;
	this->partitionArgs = partitionArgs;
	this->threadIds = threadIds;
	this->lpuIdChain = new List<int*>;
	this->taskData = NULL;
	this->partConfigMap = NULL;
	this->partIteratorMap = NULL;
	this->loggingEnabled = false;
}

PartIterator *ThreadState::getIterator(int lpsId, const char *varName) {
	if (partIteratorMap == NULL) {
		std::cout << "Data-part iterator map has not been set in thread-state\n";
		std::exit(EXIT_FAILURE);
	}
	std::ostringstream key;
	key << "Space_" << lpsId << "_Var_" << varName;
	PartIterator *iterator = partIteratorMap->Lookup(key.str().c_str());
	if (iterator == NULL) {
		std::cout << "Part iterator has not been found for Space #" << lpsId;
		std::cout << " variable '" << varName << "\n";
		std::exit(EXIT_FAILURE);
	}
	return iterator;
}

LPU *ThreadState::getNextLpu(int lpsId, int containerLpsId, int currentLpuId) {
	
	// this is not the first call to get next LPU when current Id is valid
	if (currentLpuId != INVALID_ID) {
		LpsState *state = lpsStates[lpsId];
		LpuCounter *counter = state->getCounter();
		int nextLpuId = counter->getNextLpuId(currentLpuId);
		
		// if next LPU is valid then just compute it, update LPS state and return the LPU to the caller
		if (nextLpuId != INVALID_ID) {
			counter->setCurrentCompositeLpuId(nextLpuId);
			LPU *lpu = computeNextLpu(lpsId);
		
			/*---------------------- Disabled	
			// log LPU execution
			if(loggingEnabled) {
				for (int i = 0; i < lpsId; i++) threadLog << '\t';
				threadLog << "Next LPU: " << nextLpuId << std::endl;
				counter->logCompositeLpuId(threadLog, lpsId);
				lpu->print(threadLog, lpsId + 1);
			}
			-------------------------------*/
		
			// set the LPU Id so that recursion can advance to the next LPU in next call 
			lpu->setId(nextLpuId);	
			return lpu;

		// otherwise there is a possible need for recursively going up in the LPS hierarchy and 
		// recompute some parent LPUs before we can decide if anymore LPUs are there in the current LPS
		// to execute	
		} else {
			// reset state regardless of subsequent changes			
			counter->resetCounter();
			state->invalidateCurrentLpu();
			
			// if parent is checkpointed then there is no further scope for recursion and we should
			// declare that no new LPUs to be executed
			int parentLpsId = lpsParentIndexMap[lpsId];
			LpsState *parentState = lpsStates[parentLpsId];
			if (parentState->isIterationBound()) {
				return NULL;
			// otherwise check if parent LPS'es LPU can be updated to resume iteration of LPUs in
			// current LPS
			} else {
				while (nextLpuId == INVALID_ID) {			
					LpuCounter *parentCounter = parentState->getCounter();
					int parentLpuId = parentCounter->getCurrentLpuId();

					// recursively call the same routine in the parent LPS to update the 
					// parent LPU if possible
					LPU *parentLpu = getNextLpu(parentLpsId, containerLpsId, parentLpuId);
				
					// If parent LPU is NULL then it means all parent LPUs have been executed 
					// too. So there is nothing more to do in current LPS either 
					if (parentLpu == NULL) return NULL;
				
					// Otherwise, counters for current LPS should be reset, the range of LPUs 
					// that current thread needs to execute should also be renewed
					int *newLpuCounts = computeLpuCounts(lpsId);
					counter->setLpuCounts(newLpuCounts);
					delete[] newLpuCounts;
					counter->setCurrentRange(threadIds->ppuIds[lpsId]);
	
					/*---------------------- Disabled	
					// log counter update
					if (loggingEnabled) {
						counter->logLpuCount(threadLog, lpsId);
						counter->logLpuRange(threadLog, lpsId);
					}
					-------------------------------*/
					
					// retrieve next LPU Id from the updated counter
					nextLpuId = counter->getNextLpuId(INVALID_ID);
				}

				// finally, compute next LPU to execute, save state, and return the LPU
				counter->setCurrentCompositeLpuId(nextLpuId);
				LPU *lpu = computeNextLpu(lpsId);
				
				/*---------------------- Disabled	
				// log LPU execution
				if (loggingEnabled) {
					for (int i = 0; i < lpsId; i++) threadLog << '\t';
					threadLog << "Return LPU: " << nextLpuId << std::endl;
					counter->logCompositeLpuId(threadLog, lpsId);
					lpu->print(threadLog, lpsId + 1);
				}
				-------------------------------*/
	
				// set the LPU Id so that recursion can advance to the next LPU in next call 
				lpu->setId(nextLpuId);	
				return lpu;	
			}
		}
	}

	// this is the first call to the routine if current ID is invalid  
	LpsState *containerState = lpsStates[containerLpsId];
	// checkpoint the container state to limit the span of recursive get-Next-LPU call below that LPS
	containerState->markAsIterationBound();
	
	// check if parent LPS is the same as the container LPS	
	int parentLpsId = lpsParentIndexMap[lpsId];
	LpsState *parentState = lpsStates[parentLpsId];
	// if they are not the same then do a recursive get-Next_LPU call on the parent to initiate parent's counter
	if (containerLpsId != parentLpsId && parentLpsId != INVALID_ID) {
		LPU *parentLpu = getNextLpu(parentLpsId, containerLpsId, INVALID_ID);
		if (parentLpu == NULL) return NULL;
	}

	// initiate LPU counter and range variables
	LpsState *state = lpsStates[lpsId];
	LpuCounter *counter = state->getCounter();
	int *newLpuCounts = computeLpuCounts(lpsId);
	counter->setLpuCounts(newLpuCounts);
	delete[] newLpuCounts;
	counter->setCurrentRange(threadIds->ppuIds[lpsId]);
			
	/*---------------------- Disabled	
	// log counter update
	if (loggingEnabled) {	
		counter->logLpuCount(threadLog, lpsId);
		counter->logLpuRange(threadLog, lpsId);
	}	
	-------------------------------*/
	
	// compute the next LPU for the current LPS using a recursive procedure
	int nextLpuId = counter->getNextLpuId(INVALID_ID);
	while (nextLpuId == INVALID_ID) {
	
		// if it is not possible to go recursively up to find the first valid LPU for current
		// LPS then return NULL	
		if (containerLpsId == parentLpsId || parentLpsId == INVALID_ID) return NULL;			
		
		LpuCounter *parentCounter = parentState->getCounter();
		int parentLpuId = parentCounter->getCurrentLpuId();

		// recursively call the same routine in the parent LPS to update the 
		// parent LPU if possible
		LPU *parentLpu = getNextLpu(parentLpsId, containerLpsId, parentLpuId);
	
		// If parent LPU is NULL then it means all parent LPUs have been executed 
		// too. So there is nothing more to do in current LPS either 
		if (parentLpu == NULL) return NULL;
	
		// Otherwise, counters for current LPS should be reset, the range of LPUs 
		// that current thread needs to execute should also be renewed
		int *newLpuCounts = computeLpuCounts(lpsId);
		counter->setLpuCounts(newLpuCounts);
		delete[] newLpuCounts;
		counter->setCurrentRange(threadIds->ppuIds[lpsId]);

		/*---------------------- Disabled	
		// log counter update
		if (loggingEnabled) {
			counter->logLpuCount(threadLog, lpsId);
			counter->logLpuRange(threadLog, lpsId);
		}
		-------------------------------*/
		
		// retrieve next LPU Id from the updated counter
		nextLpuId = counter->getNextLpuId(INVALID_ID);
	}
	counter->setCurrentCompositeLpuId(nextLpuId);
	LPU *lpu = computeNextLpu(lpsId);
	
	/*---------------------- Disabled	
	// log LPU execution
	if (loggingEnabled) {
		for (int i = 0; i < lpsId; i++) threadLog << '\t';
		threadLog << "Start LPU: " << nextLpuId << std::endl;
		counter->logCompositeLpuId(threadLog, lpsId);
		lpu->print(threadLog, lpsId + 1);
	}
	-------------------------------*/
	
	// set the LPU Id so that recursion can advance to the next LPU in next call 
	lpu->setId(nextLpuId);	
	return lpu;
}

int ThreadState::getNextLpuId(int lpsId, int containerLpsId, int currentLpuId) {
	LPU *lpu = getNextLpu(lpsId, containerLpsId, currentLpuId);
	if (lpu == NULL) return INVALID_ID;
	else return lpu->id;
}

int *ThreadState::getCurrentLpuId(int lpsId) {
	LpsState *state = lpsStates[lpsId];
	LpuCounter *counter = state->getCounter();
	return counter->getCompositeLpuId();
}

List<int*> *ThreadState::getLpuIdChain(int lpsId, int rootLpsId) {
	
	List<int*> *chain = new List<int*>;
	
	// determine the list of ancestor LPSes including the argument LPS whose LPU ids will be needed to
	// make sense of an LPU id in the latter.
	List<int> *relevantLpsIds = new List<int>;
	relevantLpsIds->Append(lpsId);
	int currentLpsId = lpsId;
	while ((currentLpsId = lpsParentIndexMap[currentLpsId]) != rootLpsId) {
		relevantLpsIds->Append(currentLpsId);
	}

	// enter LPU ids in the list from upper to lower LPS order to simplify future processing
	for (int i = relevantLpsIds->NumElements() - 1; i >= 0; i--) {
		currentLpsId = relevantLpsIds->Nth(i);
		LpsState *state = lpsStates[currentLpsId];
		LpuCounter *counter = state->getCounter();
		chain->Append(counter->copyCompositeLpuId());
	}

	return chain;
}

List<int*> *ThreadState::getLpuIdChainWithoutCopy(int lpsId, int rootLpsId) {
	// clear the current LPU Id chain
	while (lpuIdChain->NumElements() > 0) {
		lpuIdChain->RemoveAt(0);
	}
	// iterate over parent pointers and add LPU Ids in the chain
	int currentLpsId = lpsId;
	while (currentLpsId != rootLpsId) {
		LpsState *state = lpsStates[currentLpsId];
		LpuCounter *counter = state->getCounter();
		lpuIdChain->InsertAt(counter->getCompositeLpuId(), 0);
		currentLpsId = lpsParentIndexMap[currentLpsId];
	}
	return lpuIdChain;	
}

int *ThreadState::getLpuCounts(int lpsId) {
	LpsState *state = lpsStates[lpsId];
	LpuCounter *counter = state->getCounter();
	return counter->getLpuCounts();
}

void ThreadState::removeIterationBound(int lpsId) {
	LpsState *state = lpsStates[lpsId];
	state->removeIterationBound();
}

bool ThreadState::isValidPpu(int lpsId) {
	PPU_Ids ppu = threadIds->ppuIds[lpsId];
	return ppu.id != INVALID_ID;
}

void ThreadState::initiateLogFile(const char *fileNamePrefix) {
	std::ostringstream fileName;
	fileName << fileNamePrefix;
	fileName << "_" << threadIds->threadNo << ".log";
	threadLog.open(fileName.str().c_str(), std::ofstream::out | std::ofstream::app);
	if (!threadLog.is_open()) {
		std::cout << "Could not open log file for Thread-" << threadIds->threadNo << "\n";
	}
}

void ThreadState::logExecution(const char *stageName, int spaceId) {
	if (loggingEnabled) {
		for (int i = 0; i <= spaceId; i++) threadLog << '\t';
		threadLog << "Executed: " << stageName << std::endl;
	}
}

void ThreadState::logThreadAffinity() {
	if (loggingEnabled) {
		threadLog << "Thread Id: " << pthread_self() << std::endl;
		threadLog << "Thread CPU Id: " << sched_getcpu() << std::endl;
	} 	
}

LPU *ThreadState::getCurrentLpu(int lpsId, bool allowInvalid) {
	LpsState *state = lpsStates[lpsId];
	return state->getCurrentLpu(allowInvalid);
}

void ThreadState::closeLogFile() {
	if (threadLog.is_open()) threadLog.close();
}

void ThreadState::logIteratorStatistics() {
	if (loggingEnabled) {
		Iterator<PartIterator*> iterator = partIteratorMap->GetIterator();
		PartIterator *partIterator = NULL;
		while ((partIterator = iterator.GetNextValue()) != NULL) {
			partIterator->printStats(threadLog, 0);
		}
	}
}

/***********************************************  Segment State  ****************************************************/

SegmentState::SegmentState(int segmentId, int physicalId) {
                this->segmentId = segmentId;
                this->physicalId = physicalId;
                this->participantList = new List<ThreadState*>;
                this->partConfigMap = NULL;
}
