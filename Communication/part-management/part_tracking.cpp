#include "part_tracking.h"
#include "../utils/list.h"
#include "../utils/binary_search.h"
#include "part_folding.h"

#include <vector>
#include <iostream>
#include <sstream>
#include <cstdlib>

using namespace std;

//-------------------------------------------------------------- Super Part -------------------------------------------------------------/

SuperPart::SuperPart(List<int*> *partId, int dataDimensions) {
	this->partId = new List<int*>;
	for (int i = 0; i < partId->NumElements(); i++) {
		int *idAtLevel = new int[dataDimensions];
		for (int d = 0; d < dataDimensions; d++) {
			idAtLevel[d] = partId->Nth(i)[d];
		}
		this->partId->Append(idAtLevel);
	}
}

bool SuperPart::isMatchingId(int dimensions, List<int*> *candidatePartId) {
	for (int i = 0; i < partId->NumElements(); i++) {
		int *currentId = partId->Nth(i);
		int *currentCandidate = candidatePartId->Nth(i);
		for (int d = 0; d < dimensions; d++) {
			if (currentId[d] != currentCandidate[d]) return false;
		}
	}
	return true;
}

//--------------------------------------------------------- Part Id Container -----------------------------------------------------------/
        
PartIdContainer::PartIdContainer(DimConfig dimConfig) {
	this->level = dimConfig.getLevel();
	this->dimNo = dimConfig.getDimNo();
	this->packed = false;
}
        
int PartIdContainer::getCurrentLevelPartIndex(List<int*> *partId) {
	int currentId = partId->Nth(level)[dimNo];
	return binsearch::locateKey(partArray, currentId);
}

int PartIdContainer::getCurrentLevelIndexOfExistingPart(List<int*> *partId) {
	int currentId = partId->Nth(level)[dimNo];
	if (packed) return (currentId - partArray[0]);
	return binsearch::locateKey(partArray, currentId);
}

PartIterator *PartIdContainer::getIterator() {
	PartIdContainer *container = this;
	int partIdSteps = 1;
	while (dynamic_cast<PartListContainer*>(container) != NULL) {
		partIdSteps++;
		container = ((PartListContainer *) container)->getNestedContainerAtIndex(0);
	}
	PartIterator *iterator = new PartIterator(partIdSteps);
	iterator->initiate(this);
	return iterator;
}

SuperPart *PartIdContainer::getPart(List<int*> *partId, PartIterator *iterator, int dataDimension) {
	SuperPart *part = iterator->getCurrentPart();
	if (part != NULL && part->isMatchingId(dataDimension, partId)) {
		iterator->recordDirectAccess();
		return part;
	}
	if (iterator->advance()) {
		part = iterator->getCurrentPart();
		if (part != NULL && part->isMatchingId(dataDimension, partId)) {
			iterator->recordOneStepAdvance();
			return part;
		}
	}
	iterator->reset();
	iterator->recordMove();
	return getPart(partId, iterator);
}

void PartIdContainer::postProcess() {
	std::vector<int>(partArray).swap(partArray);
	int startId = partArray[0];
	int endId = partArray[partArray.size() - 1];
	packed = ((unsigned int) (endId - startId + 1) == partArray.size());
}

int PartIdContainer::getPartCount() {
	PartIterator *iterator = getIterator();
	int count = 0;
	while (iterator->getCurrentPart() != NULL) {
		count++;
		iterator->advance();
	}
	delete iterator;
	return count;
}

void PartIdContainer::print(int indentLevel, std::ostream &stream) {
	std::ostringstream indent;
	for (int i = 0; i < indentLevel; i++) indent << '\t';
	stream << '\n' << indent.str();
	for (unsigned int i = 0; i < partArray.size(); i++) {
		stream << partArray[i] << ' ';
	}
}

//----------------------------------------------------------- Part Container ------------------------------------------------------------/

PartContainer::~PartContainer() {
	for (unsigned int i = 0; i < partArray.size(); i++) {
		SuperPart *dataPart = dataPartList[i];
		delete dataPart;
	}
}
        
bool PartContainer::insertPartId(List<int*> *partId,
                int dataDimensions,
		vector<DimConfig> dimOrder, unsigned int position) {
	int index = getCurrentLevelPartIndex(partId);
	if (index != KEY_NOT_FOUND) return false;
	int currentId = partId->Nth(level)[dimNo];
	int insertIndex = binsearch::locatePointOfInsert(partArray, currentId);
	partArray.insert(partArray.begin() + insertIndex, currentId);
	dataPartList.insert(dataPartList.begin() + insertIndex, new SuperPart(partId, dataDimensions));
	return true;
}
        
void PartContainer::postProcess() {
	PartIdContainer::postProcess();
	std::vector<SuperPart*>(dataPartList).swap(dataPartList);
}

SuperPart *PartContainer::getPart(List<int*> *partId, PartIterator *iterator) {
	int index = getCurrentLevelIndexOfExistingPart(partId);
	iterator->addStep(this, index);
	return dataPartList[index];
}

void PartContainer::replacePartAtIndex(SuperPart *repacement, int index) {
	SuperPart *oldPart = dataPartList[index];
	dataPartList[index] = repacement;
	delete oldPart;
}

void PartContainer::foldContainer(List<PartFolding*> *fold) {
	List<Range*> *rangeList = new List<Range*>;
	if (partArray.size() > 0) {
		rangeList->Append(new Range(partArray[0]));
	}
	int lastIndex = 0;
	for (unsigned int i = 1; i < partArray.size(); i++) {
		int partId = partArray[i];
		if (rangeList->Nth(lastIndex)->max == partId - 1) {
			rangeList->Nth(lastIndex)->max = partId;
		} else {
			rangeList->Append(new Range(partId));
			lastIndex++;
		}
	}
	while(rangeList->NumElements() > 0) {
		Range *range = rangeList->Nth(0);
		fold->Append(new PartFolding(range, dimNo, level));
		rangeList->RemoveAt(0);
		delete range;
	}
	delete rangeList;
}

//------------------------------------------------------- Part List Container -----------------------------------------------------------/

PartListContainer::~PartListContainer() {
	for (unsigned int i = 0; i < partArray.size(); i++) {
		PartIdContainer *nextContainer = nextLevelContainers[i];
		delete nextContainer;
	}
}
        
bool PartListContainer::insertPartId(List<int*> *partId,
                int dataDimensions,
		vector<DimConfig> dimOrder, unsigned int position) {
	
	int index = getCurrentLevelPartIndex(partId);
	position++;
	if (index == KEY_NOT_FOUND) {
		int currentId = partId->Nth(level)[dimNo];
		int insertIndex = binsearch::locatePointOfInsert(partArray, currentId);
		partArray.insert(partArray.begin() + insertIndex, currentId);
		PartIdContainer *nextContainer;
		if (position < dimOrder.size() - 1) {
			nextContainer = new PartListContainer(dimOrder[position]);
		} else {
			nextContainer = new PartContainer(dimOrder[position]);
		}
		nextLevelContainers.insert(nextLevelContainers.begin() + insertIndex, nextContainer);
		nextContainer->insertPartId(partId, dataDimensions, dimOrder, position);
		return true;
	} else {
		PartIdContainer *nextContainer = nextLevelContainers[index];
		return nextContainer->insertPartId(partId, dataDimensions, dimOrder, position);
	}
}

void PartListContainer::postProcess() {
	PartIdContainer::postProcess();
	std::vector<PartIdContainer*>(nextLevelContainers).swap(nextLevelContainers);
	for (unsigned int i = 0; i < partArray.size(); i++) {
		PartIdContainer *nextContainer = nextLevelContainers[i];
		nextContainer->postProcess();
	}
}
        
void PartListContainer::print(int indentLevel, std::ostream &stream) {
	std::ostringstream indent;
	for (int i = 0; i < indentLevel; i++) indent << '\t';
	for (unsigned int i = 0; i < partArray.size(); i++) {
		stream << '\n' << indent.str() << partArray[i] << ':';
		PartIdContainer *nextContainer = nextLevelContainers[i];
		nextContainer->print(indentLevel + 1, stream);
	}
}

SuperPart *PartListContainer::getPart(List<int*> *partId, PartIterator *iterator) {
	int index = getCurrentLevelIndexOfExistingPart(partId);
	PartIdContainer *nextContainer = nextLevelContainers[index];
	iterator->addStep(this, index);
	return nextContainer->getPart(partId, iterator);
}

void PartListContainer::foldContainer(List<PartFolding*> *fold) {
	for (unsigned int i = 0; i < partArray.size(); i++) {
		PartFolding *contentFold = new PartFolding(partArray[i], dimNo, level);
		nextLevelContainers[i]->foldContainer(contentFold->getDescendants());
		int foldedEntries = fold->NumElements();
		if (foldedEntries > 0) {
			PartFolding *previousFold = fold->Nth(foldedEntries - 1);
			if (previousFold->isEqualInContent(contentFold)
					&& (previousFold->getIdRange().max == partArray[i] - 1)) {
				previousFold->coalesce(contentFold->getIdRange());
				delete contentFold;
			} else {
				fold->Append(contentFold);
			}
		} else {
			fold->Append(contentFold);
		}
	}
}

//---------------------------------------------------------- Part Iterator --------------------------------------------------------------/

void IteratorStatistics::print(std::ostream &stream, int indent) {
	std::ostringstream indentStr;
	for (int i = 0; i < indent; i++) indentStr << '\t';
	stream << indentStr.str() << "Iterator Statistics:\n";
	indentStr << '\t';
	stream << indentStr.str() << "#times current position is the searched position  (direct return): ";
	stream << directAccess << '\n';
	stream << indentStr.str() << "#times next position is the searched position  (one step advance): ";
	stream << oneStepAdvance << '\n';
	stream << indentStr.str() << "#times iterator has been put to a new location (container search): ";
	stream << nonAdjacentMoves << '\n';
}

PartIterator::PartIterator(int partIdSteps) {
	this->partIdSteps = partIdSteps;
	containerStack.reserve(partIdSteps);
	indexStack.reserve(partIdSteps);
	this->partIdTemplate = NULL;
	this->stats = IteratorStatistics();
}

SuperPart *PartIterator::getCurrentPart() {
	if (containerStack.size() < partIdSteps) return NULL;
	PartContainer *container = (PartContainer*) containerStack[partIdSteps - 1];
	int index = indexStack[partIdSteps - 1];
	return container->getPartAtIndex(index);
}
        
void PartIterator::replaceCurrentPart(SuperPart *replacement) {
	PartContainer *container = (PartContainer*) containerStack[partIdSteps - 1];
	int index = indexStack[partIdSteps - 1];
	container->replacePartAtIndex(replacement, index);
}
        
void PartIterator::initiate(PartIdContainer *topContainer) {
	containerStack.push_back(topContainer);
	indexStack.push_back(0);
	int accessPoint = containerStack.size() - 1;
	PartIdContainer *container = topContainer;
	while (accessPoint < partIdSteps - 1) {
		PartListContainer *listContainer = reinterpret_cast<PartListContainer*>(container);
		PartIdContainer *nextContainer = listContainer->getNestedContainerAtIndex(0);
		containerStack.push_back(nextContainer);
		indexStack.push_back(0);
		accessPoint++;
		container = nextContainer;
	}
}

void PartIterator::initiatePartIdTemplate(int dataDimensions, int idLevels) {
	partIdTemplate = new List<int*>(idLevels);
	for (int i = 0; i < idLevels; i++) {
		partIdTemplate->Append(new int[dataDimensions]);
	}
}

void PartIterator::reset() {
	while (!containerStack.empty()) {
		containerStack.clear();
		indexStack.clear();
	}
}

void PartIterator::addStep(PartIdContainer *container, int index) {
	containerStack.push_back(container);
	indexStack.push_back(index);
}

bool PartIterator::advance(int lastAccessPoint) {
	if (lastAccessPoint < 0) return false;
	PartIdContainer *container = containerStack[lastAccessPoint];
	int lastIndex = indexStack[lastAccessPoint];
	int  index = lastIndex + 1;
	int size = container->getSize();
	if (index < size) {
		indexStack[lastAccessPoint] = index;
		if (lastAccessPoint < partIdSteps - 1) {
			PartListContainer *listContainer = reinterpret_cast<PartListContainer*>(container);
			PartIdContainer *nextContainer = listContainer->getNestedContainerAtIndex(index);
			initiate(nextContainer);
		}
		return true;
	} else {
		containerStack.pop_back();
		indexStack.pop_back();
		return advance(lastAccessPoint - 1);
	}
}
