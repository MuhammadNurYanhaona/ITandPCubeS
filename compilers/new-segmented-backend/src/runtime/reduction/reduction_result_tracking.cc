#include "reduction_barrier.h"
#include "reduction_result_tracking.h"
#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/utils/binary_search.h"

#include <vector>
#include <cstdlib>

//------------------------------------------------ Reduction Result Container ----------------------------------------------/

ReductionResultContainer::ReductionResultContainer(int lpsIndex, int lpuDimIndex) {
	this->lpsIndex = lpsIndex;
	this->lpuDimIndex = lpuDimIndex;
}

int ReductionResultContainer::getLpsIndexOfNextContainer(vector<int> *lpuIdDimensions) {
	int currentIdDimensionality = lpuIdDimensions->at(lpsIndex);
	if (lpuDimIndex == currentIdDimensionality - 1) return lpsIndex + 1;
	else return lpsIndex;
}
        
int ReductionResultContainer::getlpuDimIndexOfNextContainer(vector<int> *lpuIdDimensions) {
	int currentIdDimensionality = lpuIdDimensions->at(lpsIndex);
        if (lpuDimIndex == currentIdDimensionality - 1) return 0;
	return lpuDimIndex + 1;
}

//------------------------------------------ Intermediate Reduction Result Container ---------------------------------------/

InterimReductionResultContainer::InterimReductionResultContainer(int lpsIndex, 
		int lpuDimIndex) 
		: ReductionResultContainer(lpsIndex, lpuDimIndex) {}

InterimReductionResultContainer::~InterimReductionResultContainer() {
	for (int i = 0; i < nextLevelContainers.size(); i++) {
		ReductionResultContainer *nextContainer = nextLevelContainers[i];
		delete nextContainer; 
	}
}

void InterimReductionResultContainer::initiateResultVarforLpu(List<int*> *lpuId, 
		int remainingPositions, 
		vector<int> *lpuIdDimensions) {
	
	int idPart = lpuId->Nth(lpsIndex)[lpuDimIndex];
	int index = binsearch::locateKey(idArray, idPart);
	if (index == KEY_NOT_FOUND) {
		int nextLpsIndex = getLpsIndexOfNextContainer(lpuIdDimensions);
		int nextLpuDimIndex = getlpuDimIndexOfNextContainer(lpuIdDimensions);
		ReductionResultContainer *nextContainer = NULL;
		
		if (remainingPositions > 1) {
			nextContainer = new InterimReductionResultContainer(nextLpsIndex, nextLpuDimIndex);
		} else {
			nextContainer = new TerminalReductionResultContainer(nextLpsIndex, nextLpuDimIndex);
		}
		
		int insertIndex = binsearch::locatePointOfInsert(idArray, idPart);
		idArray.insert(idArray.begin() + insertIndex, idPart);

		nextLevelContainers.insert(nextLevelContainers.begin() + insertIndex, nextContainer);
                nextContainer->initiateResultVarforLpu(lpuId, remainingPositions - 1, lpuIdDimensions);	
	} else {
		ReductionResultContainer *nextContainer = nextLevelContainers[index];
                nextContainer->initiateResultVarforLpu(lpuId, remainingPositions - 1, lpuIdDimensions);
	}
}

reduction::Result *InterimReductionResultContainer::retrieveResultForLpu(List<int*> *lpuId) {
	int idPart = lpuId->Nth(lpsIndex)[lpuDimIndex];
        int index = binsearch::locateKey(idArray, idPart);
	ReductionResultContainer *nextContainer = nextLevelContainers[index];
	return nextContainer->retrieveResultForLpu(lpuId);
}

//-------------------------------------------- Terminal Reduction Result Container -----------------------------------------/

TerminalReductionResultContainer::TerminalReductionResultContainer(int lpsIndex,
                int lpuDimIndex) 
                : ReductionResultContainer(lpsIndex, lpuDimIndex) {}

TerminalReductionResultContainer::~TerminalReductionResultContainer() {
	for (int i = 0; i < resultVariables.size(); i++) {
		reduction::Result *variable = resultVariables[i];
		delete variable;
	}
}

void TerminalReductionResultContainer::initiateResultVarforLpu(List<int*> *lpuId,
		int remainingPositions,
		vector<int> *lpuIdDimensions) {

	int idPart = lpuId->Nth(lpsIndex)[lpuDimIndex];
        int index = binsearch::locateKey(idArray, idPart);
        if (index != KEY_NOT_FOUND) return;

	int insertIndex = binsearch::locatePointOfInsert(idArray, idPart);
	idArray.insert(idArray.begin() + insertIndex, idPart);
	resultVariables.insert(resultVariables.begin() + insertIndex, new reduction::Result());
}

reduction::Result *TerminalReductionResultContainer::retrieveResultForLpu(List<int*> *lpuId) {
	int idPart = lpuId->Nth(lpsIndex)[lpuDimIndex];
        int index = binsearch::locateKey(idArray, idPart);
	return resultVariables[index];
}

//---------------------------------------------- Reduction Result Access Container -----------------------------------------/

ReductionResultAccessContainer::ReductionResultAccessContainer(vector<int> idDimensions) {
	idComponents = 0;
	for (int i = 0; i < idDimensions.size(); i++) {
		lpuIdDimensions.insert(lpuIdDimensions.end(), idDimensions[i]);
		idComponents += idDimensions[i];
	}
	if (idComponents == 1) {
		topLevelContainer = new TerminalReductionResultContainer(0, 0);
	} else {
		topLevelContainer = new InterimReductionResultContainer(0, 0);
	}
}

ReductionResultAccessContainer::~ReductionResultAccessContainer() {
	delete topLevelContainer;
}

void ReductionResultAccessContainer::initiateResultForLpu(List<int*> *lpuId) {
	topLevelContainer->initiateResultVarforLpu(lpuId, idComponents - 1, &lpuIdDimensions);
}

reduction::Result *ReductionResultAccessContainer::getResultForLpu(List<int*> *lpuId) {
	return topLevelContainer->retrieveResultForLpu(lpuId);
}
