#include "../task_space.h"
#include "../symbol.h"
#include "../partition_function.h"
#include "../../common/constant.h"
#include "../../common/location.h"
#include "../../common/errors.h"
#include "../../syntax/ast.h"
#include "../../syntax/ast_def.h"
#include "../../static-analysis/usage_statistic.h"
#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/utils/hashtable.h"

#include <deque>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <cstdlib>

//----------------------------------------------------- Token ---------------------------------------------------/

int Token::wildcardTokenId = -1;

//------------------------------------------------ Coordinate ---------------------------------------------------/

Token *Coordinate::getTokenForDataStructure(const char *dataStructureName) {
	for (int i = 0; i < tokenList->NumElements(); i++) {
		Token *token = tokenList->Nth(i);
		DataStructure *structure = token->getData();
		if (strcmp(structure->getName(), dataStructureName) == 0) {
			return token;
		}
	}
	return NULL;
}

//--------------------------------------------- Coordinate System -----------------------------------------------/

CoordinateSystem::CoordinateSystem(int dimensionCount) {
	this->dimensionCount = dimensionCount;
	dimensions = new List<Coordinate*>;
	for (int i = 0; i < dimensionCount; i++) {
		dimensions->Append(new Coordinate(i + 1));
	}
}

bool CoordinateSystem::isBalanced() {
	if (dimensionCount == 0) return true;
	int tokenCountInDim1 = dimensions->Nth(0)->getTokenCount();
	for (int i = 1; i < dimensions->NumElements(); i++) {
		if (dimensions->Nth(i)->getTokenCount() != tokenCountInDim1) {
			return false;
		}
	}
	return true;
}

Coordinate *CoordinateSystem::getCoordinate(int dimensionNo) {
	int index = dimensionNo - 1;
	return dimensions->Nth(index);
}

//-------------------------------------------------- Space ------------------------------------------------------/

const char *Space::RootSpaceName = "Root";
const char *Space::SubSpaceSuffix = "_Sub";

Space::Space(const char *name, int dimensions, bool dynamic, bool subpartitionSpace) {
	
	this->id = name;
	this->dynamic = dynamic;
	this->subpartitionSpace = subpartitionSpace;
	this->dimensions = dimensions;
	if (dimensions > 0) {
		coordSys = new CoordinateSystem(dimensions);
	} else {
		coordSys = NULL;
	}
	this->parent = NULL;
	this->dataStructureList = new Hashtable<DataStructure*>;
	this->subpartition = NULL;
	this->children = new List<Space*>;
	this->executesCode = false;

	// a mark of an invalid PPS id as PPS ids are positive integers
	this->ppsId = 0;
	this->segmentedPPS = 0;
}

void Space::setStructureList(Hashtable<DataStructure*> *dataStructureList) {
	this->dataStructureList = dataStructureList;
}

void Space::initEmptyStructureList() {
	dataStructureList = new Hashtable<DataStructure*>;
}

void Space::addDataStructure(DataStructure *structure) {
	dataStructureList->Enter(structure->getName(), structure, false);
}

DataStructure *Space::getStructure(const char *name) {
	DataStructure *structure = dataStructureList->Lookup(name);
	if (structure == NULL && parent != NULL) {
		return parent->getStructure(name);
	}
	return structure;
}

bool Space::isInSpace(const char *structName) {
	bool exists = (dataStructureList->Lookup(structName) != NULL);
	if (exists) return true;
	if (subpartitionSpace) return parent->isInSpace(structName);
	else return false;
}

DataStructure *Space::getLocalStructure(const char *name) {
	DataStructure *structure = dataStructureList->Lookup(name);
	if (structure != NULL) return structure;
	if (subpartitionSpace) return parent->getLocalStructure(name);
	else return NULL;
}

void Space::storeToken(int coordinate, Token *token) {
	Coordinate *coordinateDim = coordSys->getCoordinate(coordinate);
	coordinateDim->storeToken(token);
}

bool Space::isParentSpace(Space *suspectedParent) {
	if (this->parent == NULL) return false;
	if (this->parent == suspectedParent) return true;
	return this->parent->isParentSpace(suspectedParent);
}

Space *Space::getClosestSubpartitionRoot() {
	if (subpartitionSpace) return this;
	if (parent == NULL) return NULL;
	return parent->getClosestSubpartitionRoot();
}

List<const char*> *Space::getLocallyUsedArrayNames() {
	List<const char*> *localArrays = new List<const char*>;
	Iterator<DataStructure*> iterator = dataStructureList->GetIterator();
	DataStructure *structure;
	while ((structure = iterator.GetNextValue()) != NULL) {
		ArrayDataStructure *array = dynamic_cast<ArrayDataStructure*>(structure);
		if (array != NULL) localArrays->Append(array->getName());
	}
	if (subpartitionSpace) {
		List<const char*> *parentArrays = parent->getLocallyUsedArrayNames();
		for (int i = 0; i < parentArrays->NumElements(); i++) {
			const char *parentArray = parentArrays->Nth(i);
			bool existsInCurrent = false;
			for (int j = 0; j < localArrays->NumElements(); j++) {
				if (strcmp(parentArray, localArrays->Nth(j)) == 0) {
					existsInCurrent = true;
					break;
				}
			}
			if (!existsInCurrent) localArrays->Append(parentArray);
		}
	}
	return localArrays;
}

List<const char*> *Space::getLocalDataStructureNames() {
	List<const char*> *localVars = new List<const char*>;
	Iterator<DataStructure*> iterator = dataStructureList->GetIterator();
	DataStructure *structure;
	while ((structure = iterator.GetNextValue()) != NULL) {
		localVars->Append(structure->getName());
	}
	return localVars;
}

bool Space::isReplicatedInCurrentSpace(const char *dataStructureName) {
	if (dimensions == 0) return false;
	DataStructure *structure = dataStructureList->Lookup(dataStructureName);
	if (structure == NULL) return false;
	ArrayDataStructure *array = dynamic_cast<ArrayDataStructure*>(structure);
	if (array == NULL) return true;
	for (int i = 1; i <= dimensions; i++) {
		Coordinate *coordinateDim = coordSys->getCoordinate(i);
		Token *token = coordinateDim->getTokenForDataStructure(dataStructureName);
		if (token->isWildcard()) return true;
	}
	return false;		
}

bool Space::isReplicated(const char *dataStructureName) {
	if (isReplicatedInCurrentSpace(dataStructureName)) return true;
	if (parent == NULL) return false;
	return parent->isReplicatedInCurrentSpace(dataStructureName);
}

List<Space*> *Space::getConnetingSpaceSequenceForSpacePair(Space *first, Space *last) {
	List<Space*> *spaceList = new List<Space*>;
	if (first == last) return spaceList;
	if (first->isParentSpace(last)) {
		spaceList->Append(first);
		Space *nextSpace = first;
		while ((nextSpace = nextSpace->getParent()) != last) {
			spaceList->Append(nextSpace);
		}
		spaceList->Append(last);
	} else if (last->isParentSpace(first)) {
		spaceList->Append(last);
		Space *nextSpace = last;
		while ((nextSpace = nextSpace->getParent()) != first) {
			spaceList->InsertAt(nextSpace, 0);
		}
		spaceList->InsertAt(first, 0);
	} else {
		spaceList->Append(first);
		Space *nextSpace = first;
		while ((nextSpace = nextSpace->getParent()) != NULL) {
			spaceList->Append(nextSpace);
			if (last->isParentSpace(nextSpace)) break;
		}
		Space *commonParent = nextSpace;
		int listSize = spaceList->NumElements();
		nextSpace = last;
		spaceList->Append(nextSpace);
		while ((nextSpace = nextSpace->getParent()) != commonParent) {
			spaceList->InsertAt(nextSpace, listSize);
		}
	}
	return spaceList;		
}

List<const char*> *Space::getLocalDataStructuresWithOverlappedPartitions() {
	List<const char*> *overlappingList = new List<const char*>;
	Iterator<DataStructure*> iter = dataStructureList->GetIterator();
	DataStructure *structure;
	while ((structure = iter.GetNextValue()) != NULL) {
		if (structure->hasOverlappingsAmongPartitions()) {
			overlappingList->Append(structure->getName());
		}	
	}
	return overlappingList; 
}

List<const char*> *Space::getNonStorableDataStructures() {
	List<const char*> *nonStorableStructureList = new List<const char*>;
	Iterator<DataStructure*> iter = dataStructureList->GetIterator();
	DataStructure *structure;
	while ((structure = iter.GetNextValue()) != NULL) {
		if (structure->isNonStorable()) {
			nonStorableStructureList->Append(structure->getName());
		}	
	}
	return nonStorableStructureList; 
}

Symbol *Space::getLpuIdSymbol() {
	if (dimensions == 0) return NULL;
	StaticArrayType *type = new StaticArrayType(yylloc, Type::intType, 1);
	List<int> *dimLengths = new List<int>;
	dimLengths->Append(dimensions);
	type->setLengths(dimLengths);
	VariableSymbol *symbol = new VariableSymbol("lpuId", type);
	return symbol;	
}

bool Space::allocateStructures() {
	Iterator<DataStructure*> iter = dataStructureList->GetIterator();
	DataStructure *structure;
	while ((structure = iter.GetNextValue()) != NULL) {
		if (structure->getUsageStat()->isAllocated()) return true;	
	}
	return false;
}

bool Space::allocateStructure(const char *structureName) {
	DataStructure *structure = dataStructureList->Lookup(structureName);
	if (structure == NULL) return false;
	return structure->getUsageStat()->isAllocated();
}
