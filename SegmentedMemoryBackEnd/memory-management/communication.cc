#include "communication.h"
#include "allocation.h"
#include "../codegen/structure.h"
#include "../utils/list.h"
#include "../utils/interval_utils.h"
#include "../utils/hashtable.h"

//---------------------------------------------------------- Communication Group ----------------------------------------------------------/

List<int> *CommGroup::combineGroups(List<int> *others) {
	List<int> *group = new List<int>;
	group->AppendAll(ppuIds);
	for (int i = 0; i < others->NumElements(); i++) {
		int member = others->Nth(i);
		bool found = false;
		for (int j = 0; j < ppuIds->NumElements(); j++) {
			if (ppuIds->Nth(j) == member) {
				found = true;
				break;
			}
		}
		if (!found) group->Append(member);
	}
	return group;
}

//---------------------------------------------------------- Communication Buffer ---------------------------------------------------------/

CommBuffer::CommBuffer(CommGroup *group, IntervalSet *bufferConfig) {
	this->group = group;
	this->bufferConfig = bufferConfig;
	length = 0;
	intervalSeqCount = 0;
	List<HyperplaneInterval*> *intervals = bufferConfig->getIntervalList();
	for (int i = 0; i < intervals->NumElements(); i++) {
		HyperplaneInterval *interval = intervals->Nth(i);
		length += interval->getTotalElements();
		intervalSeqCount++;
	}
	uptodateElements = 0;
	needDataCopy = true;
	data = NULL;
}

//-------------------------------------------------------- Communication Buffer List ------------------------------------------------------/

CommBufferList::CommBufferList(List<CommBuffer*> *buffers) {
	this->buffers = buffers;
	bufferCount = buffers->NumElements();
	version = 0;
	updatedBuffers = 0;
}

void CommBufferList::reset() {
	for (int i = 0; i < bufferCount; i++) {
		CommBuffer *buffer = buffers->Nth(i);
		buffer->resetState();
	}
	version++;
	updatedBuffers = 0;
}

CommBufferList *CommBufferList::generateList(List<int> *ppuIds, 
			List<IntervalSet*> *intervalConfigs, IntervalSet *currentConfig, int lpsId) {
	
	// first calculate interval intersections of individual PPUs with the current PPU
	List<CommBuffer*> *overlappingPairs = new List<CommBuffer*>;
	for (int i = 0; i < ppuIds->NumElements(); i++) {
		int ppu = ppuIds->Nth(i);
		IntervalSet *ppuInterval = intervalConfigs->Nth(i);
		IntervalSet *intersect = currentConfig->getIntersection(ppuInterval);
		if (intersect != NULL) {
			List<int> *groupIds = new List<int>;
			groupIds->Append(ppu);
			CommGroup *group = new CommGroup(lpsId, groupIds);
			CommBuffer *buffer = new CommBuffer(group, intersect);
			overlappingPairs->Append(buffer);
		}	
	}
	
	// if there is no overlapping pairs then there is no need of communication for the underlying synchronization
	if (overlappingPairs->NumElements() == 0) return NULL;

	// otherwise, invoke the recursive communication group determining algorithm
	List<CommBuffer*> *initialList = new List<CommBuffer*>;
	List<CommBuffer*> *groups = CommBufferList::generateCommBuffers(initialList, overlappingPairs, 0);
	return new CommBufferList(groups);
}

List<CommBuffer*> *CommBufferList::generateCommBuffers(List<CommBuffer*> *updatedBuffers,
                        List<CommBuffer*> *overlappingPairs, int probeIndex) {

	// get the current group from the overlappingPairs that will be investigated for further breakdown into smaller
	// buffers but with more participants	
	CommBuffer *probedBuffer = overlappingPairs->Nth(probeIndex);
	CommGroup *probedGroup = probedBuffer->getGroup();
	int lpsId = probedGroup->getLpsId();
	// create an interval-set description to hold sequences or parts of sequences that match none in the updated
	// buffers
	IntervalSet *remainingInProbed = probedBuffer->getBufferConfig();
	// create a list to hold the communication buffers for the next iteration
	List<CommBuffer*> *nextUpdateList = new List<CommBuffer*>;

	// overlapping calculation should continue until all buffers in the currently updated list are checked or there
	// is no entry remaining in the probed interval set 
	int i = 0;
	for (; i < updatedBuffers->NumElements() && remainingInProbed != NULL; i++) {
		CommBuffer *next = updatedBuffers->Nth(i);
		IntervalSet *nextInterval = next->getBufferConfig();
		IntervalSet *intersect = remainingInProbed->getIntersection(nextInterval);
		if (intersect != NULL) {
			List<int> *intersectPpus = probedGroup->combineGroups(next->getGroup()->getPpuIds());
			CommBuffer *intersectBuffer = new CommBuffer(new CommGroup(lpsId, intersectPpus), intersect);
			nextUpdateList->Append(intersectBuffer);
			remainingInProbed = remainingInProbed->getSubtraction(intersect);
			IntervalSet *remainingInNext = nextInterval->getSubtraction(intersect);
			if (remainingInNext != NULL) {
				CommBuffer *remainingNext = new CommBuffer(next->getGroup(), remainingInNext);
				nextUpdateList->Append(remainingNext);
			}
		}
	}
	// if there are remaining entries in the update list that are not tested because the probed buffer has become 
	// empty then they should be added in the updated list
	for (; i< updatedBuffers->NumElements(); i++) nextUpdateList->Append(updatedBuffers->Nth(i));

	// if there are sequences or part of sequences left in the probed buffer then add a new buffer in the list for
	// it containing the remaining part
	if (remainingInProbed != NULL) {
		CommBuffer *remainingProb = new CommBuffer(probedGroup, remainingInProbed);
		nextUpdateList->Append(remainingProb);
	}

	// continue or terminate the recursion
	int nextProbeIndex = probeIndex + 1;
	if (nextProbeIndex < overlappingPairs->NumElements()) {
		return CommBufferList::generateCommBuffers(nextUpdateList, overlappingPairs, nextProbeIndex);
	} else return nextUpdateList;
}
