#include <cstdlib>
#include <vector>
#include <time.h>
#include <iostream>
#include <sstream>
#include <stack>
#include "list.h"

using namespace std;

class PartIterator;
static const int KEY_NOT_FOUND = -1;

int locateKey(vector<int> array, int key) {
	int minIndex = 0;
	int maxIndex = array.size() - 1;
	while (maxIndex >= minIndex) {
		int midpoint = minIndex + ((maxIndex - minIndex) / 2);
		if (array[midpoint] == key) {
			return midpoint;
		} else if (array[midpoint] < key) {
			minIndex = midpoint + 1;
		} else {
			maxIndex = midpoint - 1;
		}
	}
	return KEY_NOT_FOUND;
}

int locatePointOfInsert(vector<int> array, int key) {

	if (array.empty()) return 0;
	int minIndex = 0;
	int maxIndex = array.size() - 1;

	if (array[0] > key) return 0;
	if (array[maxIndex] < key) return array.size();

	while (maxIndex >= minIndex) {
		int midpoint = minIndex + ((maxIndex - minIndex) / 2);
		if (array[midpoint] < key) {
			minIndex = midpoint + 1;
		} else {
			maxIndex = midpoint - 1;
		}
	}

	return maxIndex + 1;
}

class SuperPart {
protected:
	List<int*> *partId;
public:
	SuperPart(List<int*> *partId, int dataDimensions) {
		this->partId = new List<int*>;
		for (int i = 0; i < partId->NumElements(); i++) {
			int *idAtLevel = new int[dataDimensions];
			for (int d = 0; d < dataDimensions; d++) {
				idAtLevel[d] = partId->Nth(i)[d];
			}
			this->partId->Append(idAtLevel);
		}
	}
	List<int*> *getPartId() { return partId; }
	bool isMatchingId(int dimensions, List<int*> *candidatePartId) {
		for (int i = 0; i < partId->NumElements(); i++) {
			int *currentId = partId->Nth(i);
			int *currentCandidate = candidatePartId->Nth(i);
			for (int d = 0; d < dimensions; d++) {
				if (currentId[d] != currentCandidate[d]) return false;
			}
		}
		return true;
	}
};

class DimConfig {
private:
	int level;
	int dimNo;
public:
	DimConfig(int level, int dimNo) {
		this->level = level;
		this->dimNo = dimNo;
	}
	inline int getLevel() { return level; }
	inline int getDimNo() { return dimNo; }
};

class PartIdContainer {
protected:
	int level;
	int dimNo;
	vector<int> partArray;
	bool packed;
public:
	PartIdContainer(DimConfig dimConfig) {
		this->level = dimConfig.getLevel();
		this->dimNo = dimConfig.getDimNo();
		this->packed = false;
	}
	virtual ~PartIdContainer() {}
	int getCurrentLevelPartIndex(List<int*> *partId) {
		int currentId = partId->Nth(level)[dimNo];
		return locateKey(partArray, currentId);
	}
	int getCurrentLevelIndexOfExistingPart(List<int*> *partId) {
		int currentId = partId->Nth(level)[dimNo];
		if (packed) return (currentId - partArray[0]);
		return locateKey(partArray, currentId);
	}
	int getSize() { return partArray.size(); }
	int getPartCount();
	PartIterator *getIterator();
	SuperPart *getPart(List<int*> *partId, PartIterator *iterator, int dataDimension);
	virtual bool insertPartId(List<int*> *partId,
			int dataDimensions,
			const vector<DimConfig> dimOrder, unsigned int position) = 0;
	virtual void postProcess() {
		std::vector<int>(partArray).swap(partArray);
		int startId = partArray[0];
		int endId = partArray[partArray.size() - 1];
		packed = ((unsigned int) (endId - startId + 1) == partArray.size());
	}
	virtual void print(int indentLevel, std::ostream &stream) {
		std::ostringstream indent;
		for (int i = 0; i < indentLevel; i++) indent << '\t';
		stream << '\n' << indent.str();
		for (unsigned int i = 0; i < partArray.size(); i++) {
			stream << partArray[i] << ' ';
		}
	}
	virtual SuperPart *getPart(List<int*> *partId, PartIterator *iterator) = 0;
};

class PartContainer : public PartIdContainer {
private:
	vector<SuperPart*> dataPartList;
public:
	PartContainer(DimConfig dimConfig) : PartIdContainer(dimConfig) {}
	~PartContainer() {
		for (unsigned int i = 0; i < partArray.size(); i++) {
			SuperPart *dataPart = dataPartList[i];
			delete dataPart;
		}
	}
	bool insertPartId(List<int*> *partId,
			int dataDimensions,
			const vector<DimConfig> dimOrder, unsigned int position) {
		int index = getCurrentLevelPartIndex(partId);
		if (index != KEY_NOT_FOUND) return false;
		int currentId = partId->Nth(level)[dimNo];
		int insertIndex = locatePointOfInsert(partArray, currentId);
		partArray.insert(partArray.begin() + insertIndex, currentId);
		dataPartList.insert(dataPartList.begin() + insertIndex, new SuperPart(partId, dataDimensions));
		return true;
	}
	void postProcess() {
		PartIdContainer::postProcess();
		std::vector<SuperPart*>(dataPartList).swap(dataPartList);
	}
	void replacePartAtIndex(SuperPart *repacement, int index) {
		SuperPart *oldPart = dataPartList[index];
		dataPartList[index] = repacement;
		delete oldPart;
	}
	SuperPart *getPartAtIndex(int index) { return dataPartList[index]; }
	SuperPart *getPart(List<int*> *partId, PartIterator *iterator);
};

class PartListContainer : public PartIdContainer {
protected:
	vector<PartIdContainer*> nextLevelContainers;
public:
	PartListContainer(DimConfig dimConfig) : PartIdContainer(dimConfig) {}
	~PartListContainer() {
		for (unsigned int i = 0; i < partArray.size(); i++) {
			PartIdContainer *nextContainer = nextLevelContainers[i];
			delete nextContainer;
		}
	}
	bool insertPartId(List<int*> *partId,
			int dataDimensions,
			const vector<DimConfig> dimOrder, unsigned int position) {
		int index = getCurrentLevelPartIndex(partId);
		position++;
		if (index == KEY_NOT_FOUND) {
			int currentId = partId->Nth(level)[dimNo];
			int insertIndex = locatePointOfInsert(partArray, currentId);
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
	void postProcess() {
		PartIdContainer::postProcess();
		std::vector<PartIdContainer*>(nextLevelContainers).swap(nextLevelContainers);
		for (unsigned int i = 0; i < partArray.size(); i++) {
			PartIdContainer *nextContainer = nextLevelContainers[i];
			nextContainer->postProcess();
		}
	}
	void print(int indentLevel, std::ostream &stream) {
		std::ostringstream indent;
		for (int i = 0; i < indentLevel; i++) indent << '\t';
		for (unsigned int i = 0; i < partArray.size(); i++) {
			stream << '\n' << indent.str() << partArray[i] << ':';
			PartIdContainer *nextContainer = nextLevelContainers[i];
			nextContainer->print(indentLevel + 1, stream);
		}
	}
	PartIdContainer *getNestedContainerAtIndex(int index) { return nextLevelContainers[index]; }
	SuperPart *getPart(List<int*> *partId, PartIterator *iterator);
};

class PartIterator {
protected:
	vector<PartIdContainer*> containerStack;
	vector<int> indexStack;
	unsigned int partIdSteps;
public:
	PartIterator(int partIdSteps) {
		this->partIdSteps = partIdSteps;
		containerStack.reserve(partIdSteps);
		indexStack.reserve(partIdSteps);
	}
	SuperPart *getCurrentPart() {
		if (containerStack.size() < partIdSteps) return NULL;
		PartContainer *container = (PartContainer*) containerStack[partIdSteps - 1];
		int index = indexStack[partIdSteps - 1];
		return container->getPartAtIndex(index);
	}
	void replaceCurrentPart(SuperPart *replacement) {
		PartContainer *container = (PartContainer*) containerStack[partIdSteps - 1];
		int index = indexStack[partIdSteps - 1];
		container->replacePartAtIndex(replacement, index);
	}
	void initiate(PartIdContainer *topContainer) {
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
	bool advance() { return advance(partIdSteps - 1); }
	void reset() {
		while (!containerStack.empty()) {
			containerStack.clear();
			indexStack.clear();
		}
	}
	void addStep(PartIdContainer *container, int index) {
		containerStack.push_back(container);
		indexStack.push_back(index);
	}
private:
	bool advance(int lastAccessPoint) {
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
};

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
		cout << "direct matching from the iterator: ";
		for (int i = 0; i < partId->NumElements(); i++) {
			cout << "|";
			for (int j = 0; j < dataDimension; j++) {
				cout << "-" << partId->Nth(i)[j] << "-";
			}
			cout << "|";
		}
		cout << "\n";
		return part;
	}
	if (iterator->advance()) {
		part = iterator->getCurrentPart();
		if (part != NULL && part->isMatchingId(dataDimension, partId)) {
			cout << "iterator advance matching: ";
			for (int i = 0; i < partId->NumElements(); i++) {
				cout << "|";
				for (int j = 0; j < dataDimension; j++) {
					cout << "-" << partId->Nth(i)[j] << "-";
				}
				cout << "|";
			}
			cout << "\n";
			return part;
		}
	}
	iterator->reset();
	cout << "going to do binary search for part: ";
	for (int i = 0; i < partId->NumElements(); i++) {
		cout << "|";
		for (int j = 0; j < dataDimension; j++) {
			cout << "-" << partId->Nth(i)[j] << "-";
		}
		cout << "|";
	}
	cout << "\n";
	return getPart(partId, iterator);
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

SuperPart *PartContainer::getPart(List<int*> *partId, PartIterator *iterator) {
	int index = getCurrentLevelIndexOfExistingPart(partId);
	iterator->addStep(this, index);
	return dataPartList[index];
}

SuperPart *PartListContainer::getPart(List<int*> *partId, PartIterator *iterator) {
	int index = getCurrentLevelIndexOfExistingPart(partId);
	PartIdContainer *nextContainer = nextLevelContainers[index];
	iterator->addStep(this, index);
	return nextContainer->getPart(partId, iterator);
}

int mainPartMgmt() {

	vector<DimConfig> dimOrder;
	dimOrder.push_back(DimConfig(0, 0));
	dimOrder.push_back(DimConfig(0, 1));
	dimOrder.push_back(DimConfig(1, 0));
	dimOrder.push_back(DimConfig(1, 1));

	PartIdContainer *partIdContainer = NULL;
	if (dimOrder.size() > 1) partIdContainer = new PartListContainer(dimOrder[0]);
	else partIdContainer = new PartContainer(dimOrder[0]);

	vector<int> array;
	srand(time(NULL));

	List<List<int*>*> *partStorage = new List<List<int*>*>;
	int uniquePartsCount = 0;
	for (int i = 0; i < 5; i++) {
		int i0 = rand() % 10;
		for (int l = 0; l < 2; l++) {
			int i1 = rand() % 2;
			for (int j = 0; j < 5; j++) {
				int j0 = rand() % 10;
				for (int k = 0; k < 5; k++) {
					int j1 = rand() % 5;
					List<int*> *partId = new List<int*>;
					partId->Append(new int[2]);
					partId->Nth(0)[0] = i0;
					partId->Nth(0)[1] = i1;
					partId->Append(new int[2]);
					partId->Nth(1)[0] = j0;
					partId->Nth(1)[1] = j1;
					partStorage->Append(partId);
					bool status = partIdContainer->insertPartId(partId, 2, dimOrder, 0);
					if (status) uniquePartsCount++;
				}
			}
		}
	}

	partIdContainer->postProcess();
	partIdContainer->print(0, std::cout);
	cout.flush();


	int partsFound = 0;
	PartIterator *iterator = partIdContainer->getIterator();
	SuperPart *part = NULL;
	List<List<int*>*> *partStorage2 = new List<List<int*>*>;
	while ((part = iterator->getCurrentPart()) != NULL) {
		partStorage2->Append(part->getPartId());
		iterator->advance();
		partsFound++;
	}

	cout << "\n\nIterator validation:";
	cout << "\nParts inserted: " << uniquePartsCount << " Parts found: " << partsFound;

	cout << "\n\nSearch Validation #1:\n";
	for (int i = 0; i < partStorage->NumElements(); i++) {
		partIdContainer->getPart(partStorage->Nth(i), iterator, 2);
	}

	cout << "\n\nSearch Validation #2:\n";
	for (int i = 0; i < partStorage2->NumElements(); i++) {
		partIdContainer->getPart(partStorage2->Nth(i), iterator, 2);
	}

	cout << "\n\nCount function validation:";
	cout << "\nParts inserted: " << uniquePartsCount << " Parts count: " << partIdContainer->getPartCount();

	return 0;
}



