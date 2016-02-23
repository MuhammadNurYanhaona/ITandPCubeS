#include "task_env_stat.h"
#include "data_access.h"
#include "../semantics/task_space.h"
#include "../utils/list.h"
#include "../utils/hashtable.h"

#include <deque>
#include <iostream>
#include <cstdlib>

using namespace std;

//--------------------------------------------------- Environmental Variable Statistics ---------------------------------------------/

EnvVarStat::EnvVarStat(VariableAccess *accessLog) {
	this->varName = accessLog->getName();
	this->lpsStats = new Hashtable<EnvVarAllocationStat*>;
	this->read = accessLog->isRead();
	this->updated = accessLog->isModified();
}

void EnvVarStat::initiateLpsAllocationStat(Space *lps) {
	const char *lpsName = lps->getName();
	if (lpsStats->Lookup(lpsName) == NULL) {
		lpsStats->Enter(lps->getName(), new EnvVarAllocationStat(lps));
	}
}

void EnvVarStat::flagReadOnLps(Space *lps) {
	const char *lpsName = lps->getName();
	EnvVarAllocationStat *lpsStat = lpsStats->Lookup(lpsName);
	lpsStat->setStaleFreshMarker(true);
}
        
void EnvVarStat::flagWriteOnLps(Space *lps) {
	Iterator<EnvVarAllocationStat*> iterator = lpsStats->GetIterator();
	EnvVarAllocationStat *lpsStat = NULL;
	while ((lpsStat = iterator.GetNextValue()) != NULL) {
		lpsStat->setStaleFreshMarker(lpsStat->getLps() == lps);
	}
}

bool EnvVarStat::hasStaleAllocations() {
	Iterator<EnvVarAllocationStat*> iterator = lpsStats->GetIterator();
	EnvVarAllocationStat *lpsStat = NULL;
	while ((lpsStat = iterator.GetNextValue()) != NULL) {
		if (!lpsStat->isFresh()) return true;
	}
	return false;
}

List<Space*> *EnvVarStat::getAllocatorLpsesForState(bool fresh) {
	List<Space*> *lpsList = new List<Space*>;
	Iterator<EnvVarAllocationStat*> iterator = lpsStats->GetIterator();
	EnvVarAllocationStat *lpsStat = NULL;
	while ((lpsStat = iterator.GetNextValue()) != NULL) {
		if (lpsStat->isFresh() == fresh) {
			lpsList->Append(lpsStat->getLps());
		}
	}
	return lpsList;
}

//----------------------------------------------------- Task Environment Statistics -------------------------------------------------/

TaskEnvStat::TaskEnvStat(List<VariableAccess*> *accessMap, Space *rootLps) {
	
	varStatMap = new Hashtable<EnvVarStat*>;
	for (int i = 0; i < accessMap->NumElements(); i++) {
		VariableAccess *accessLog = accessMap->Nth(i);
		const char *varName = accessLog->getName();
		EnvVarStat *varStat = new EnvVarStat(accessLog);
		varStatMap->Enter(varName, varStat);
	}

	deque<Space*> lpsQueue;
	lpsQueue.push_back(rootLps);
	while (!lpsQueue.empty()) {

                Space *lps = lpsQueue.front();
                lpsQueue.pop_front();

                List<Space*> *children = lps->getChildrenSpaces();
                for (int i = 0; i < children->NumElements(); i++) {
                        lpsQueue.push_back(children->Nth(i));
                }
                if (lps->getSubpartition() != NULL) lpsQueue.push_back(lps->getSubpartition());

		List<const char*> *varList = lps->getLocalDataStructureNames();
		for (int i = 0; i < varList->NumElements(); i++) {
			const char *varName = varList->Nth(i);
			EnvVarStat *varStat = varStatMap->Lookup(varName);
			if (varStat == NULL) continue;
			DataStructure *structure = lps->getLocalStructure(varName);
			LPSVarUsageStat *usageStat = structure->getUsageStat();
			if (usageStat->isAllocated()) {
				varStat->initiateLpsAllocationStat(lps);
			}
					
		}
	}
}
