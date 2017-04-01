#include "communication_stat.h"
#include "../semantics/task_space.h"
#include "../semantics/data_access.h"
#include "../semantics/computation_flow.h"
#include "../static-analysis/sync_stat.h"

#include "../../../common-libs/utils/list.h"
#include "../../../common-libs/utils/hashtable.h"
#include "../../../common-libs/utils/binary_search.h"
#include "../../../common-libs/utils/string_utils.h"

#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>

//---------------------------------------------------- Communication Characteristics -------------------------------------------------------/

CommunicationCharacteristics::CommunicationCharacteristics(const char *varName) {
	this->varName = varName;
	communicationRequired = false;
	confinementSpace = NULL;
	senderDataAllocatorSpace = NULL;
	senderSyncSpace = NULL;
	receiverDataAllocatorSpace = NULL;
	receiverSyncSpace = NULL;
	syncRequirement = NULL;
	waitSignalMayDifferFromSyncLpses = false;
	signalerSpace = NULL;
	waitingSpace = NULL;
}

void CommunicationCharacteristics::setSyncRequirement(SyncRequirement *syncRequirement) {
	this->syncRequirement = syncRequirement;
}

SyncRequirement *CommunicationCharacteristics::getSyncRequirement() { 
	return syncRequirement; 
}

bool CommunicationCharacteristics::shouldAllocateGroupResources() {

	// For ghost sync and cross-propagation sync, each segment may have a different group of other segments
	// that it will be communicating with. So it is not possible to divide the segments into smaller groups
	// where communications are restricted to happen within individual groups. The remaining dependency types 
	// may be benefited from having a group communication resource set.
	GhostRegionSync *ghostSync = dynamic_cast<GhostRegionSync*>(syncRequirement);
	CrossPropagationSync *crossSync = dynamic_cast<CrossPropagationSync*>(syncRequirement);
	return (ghostSync == NULL && crossSync == NULL); 	
}

