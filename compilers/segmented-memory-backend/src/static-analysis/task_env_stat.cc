#include "task_env_stat.h"
#include "data_access.h"
#include "data_flow.h"
#include "../semantics/task_space.h"
#include "../utils/list.h"
#include "../utils/hashtable.h"

#include <deque>
#include <sstream>	
#include <iostream>
#include <cstdlib>
#include <string>

using namespace std;

//--------------------------------------------------- Environmental Variable Statistics ---------------------------------------------/

EnvVarStat::EnvVarStat(VariableAccess *accessLog) {
	this->varName = accessLog->getName();
	this->lpsStats = new Hashtable<EnvVarLpsStat*>;
	this->read = accessLog->isRead();
	this->updated = accessLog->isModified();
}

void EnvVarStat::initiateLpsUsageStat(Space *lps) {
	const char *lpsName = lps->getName();
	if (lpsStats->Lookup(lpsName) == NULL) {
		lpsStats->Enter(lps->getName(), new EnvVarLpsStat(lps));
	}
}

void EnvVarStat::flagReadOnLps(Space *lps) {
	const char *lpsName = lps->getName();
	EnvVarLpsStat *lpsStat = lpsStats->Lookup(lpsName);
	lpsStat->setStaleFreshMarker(true);
}
        
void EnvVarStat::flagWriteOnLps(Space *lps) {
	
	Iterator<EnvVarLpsStat*> iterator = lpsStats->GetIterator();
	EnvVarLpsStat *lpsStat = NULL;
	bool isSubpartition = lps->isSubpartitionSpace();

	while ((lpsStat = iterator.GetNextValue()) != NULL) {
		Space *currLps = lpsStat->getLps();
		bool stat = (currLps == lps) || (isSubpartition && currLps == lps->getParent());
		lpsStat->setStaleFreshMarker(stat);
	}
}

bool EnvVarStat::hasStaleLpses() {
	Iterator<EnvVarLpsStat*> iterator = lpsStats->GetIterator();
	EnvVarLpsStat *lpsStat = NULL;
	while ((lpsStat = iterator.GetNextValue()) != NULL) {
		if (!lpsStat->isFresh()) return true;
	}
	return false;
}

List<Space*> *EnvVarStat::getLpsesForState(bool fresh) {
	List<Space*> *lpsList = new List<Space*>;
	Iterator<EnvVarLpsStat*> iterator = lpsStats->GetIterator();
	EnvVarLpsStat *lpsStat = NULL;
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
			if (usageStat->isAccessed() || usageStat->isReduced()) {
				varStat->initiateLpsUsageStat(lps);
			}
					
		}
	}
}

List<FlowStage*> *TaskEnvStat::generateSyncStagesForStaleLpses() {
	
	Hashtable<Space*> *lpsTable = new Hashtable<Space*>;
	Hashtable<List<const char*>*> *staleVarListPerLps = new Hashtable<List<const char*>*>;

	// first organized the stale variables as groups under different LPSes
	Iterator<EnvVarStat*> iterator = varStatMap->GetIterator();
	EnvVarStat *varStat = NULL;
	while ((varStat = iterator.GetNextValue()) != NULL) {
		const char *varName = varStat->getVarName();
		List<Space*> *staleLpses = varStat->getStaleLpses();
		for (int i = 0; i < staleLpses->NumElements(); i++) {
			Space *lps = staleLpses->Nth(i);
			const char *lpsName = lps->getName();
			List<const char*> *staleVarList = staleVarListPerLps->Lookup(lpsName);
			if (staleVarList == NULL) {
				staleVarList = new List<const char*>;
			}
			staleVarList->Append(varName);
			staleVarListPerLps->Enter(lpsName, staleVarList);
			lpsTable->Enter(lpsName, lps);
		}
	}

	// then create a sync stage for each LPS where the stage reads all stale data structures
	List<FlowStage*> *syncStageList = new List<FlowStage*>;
	Iterator<Space*> lpsIterator = lpsTable->GetIterator();
	Space *staleLps = NULL;
	while ((staleLps = lpsIterator.GetNextValue()) != NULL) {
		const char *lpsName = staleLps->getName();
		ostringstream stageName;
		stageName << "Space "<< lpsName << " Updater"; 
		SyncStage *stage = new SyncStage(staleLps, Load, Entrance);
		stage->setName(strdup(stageName.str().c_str()));
		
		List<const char*> *staleVarList = staleVarListPerLps->Lookup(lpsName);
		for (int i = 0; i < staleVarList->NumElements(); i++) {
			const char *varName = staleVarList->Nth(i);
			VariableAccess *accessLog = new VariableAccess(varName);
			accessLog->markContentAccess();
			accessLog->getContentAccessFlags()->flagAsRead();
			stage->addAccessInfo(accessLog);
		}
		syncStageList->Append(stage);	
	}

	delete lpsTable;
	delete staleVarListPerLps;
	return syncStageList;
}
