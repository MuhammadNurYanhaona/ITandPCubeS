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

//------------------------------------------------- DataStructure ---------------------------------------------------/

DataStructure::DataStructure(VariableDef *definition) {
	Assert(definition != NULL);
	this->definition = definition;
	this->source = NULL;
	this->dependents = new List<DataStructure*>;
	this->space = NULL;
	this->nonStorable = false;
	this->usageStat = new LPSVarUsageStat;
	this->allocator = NULL;
	this->versionCount = 0;
}

DataStructure::DataStructure(DataStructure *source) {
	Assert(source != NULL);
	this->definition = NULL;
	this->source = source;
	this->dependents = new List<DataStructure*>;
	this->space = NULL;
	this->nonStorable = false;
	this->usageStat = new LPSVarUsageStat;
	this->allocator = NULL;
	this->versionCount = 0;
}

void DataStructure::setSpaceReference(Space *space) {
	this->space = space;
}

const char *DataStructure::getName() {
	if (this->definition != NULL) {
		return definition->getId()->getName();
	}
	return source->getName();
}

Type *DataStructure::getType() {
	if (this->definition != NULL) {
		return definition->getType();
	}
	return source->getType();
}

DataStructure *DataStructure::getClosestAllocation() {
	if (usageStat->isAllocated()) return this;
	if (source == NULL) return NULL;
	return source->getClosestAllocation();
}

bool DataStructure::useSameAllocation(DataStructure *other) {
	DataStructure *myAllocation = getClosestAllocation();
	DataStructure *otherAllocation = other->getClosestAllocation();
	return myAllocation == otherAllocation;
}

void DataStructure::updateVersionCount(int version) {
	
	int oldVersionCount = versionCount;
	versionCount = std::max(oldVersionCount, version);
	
	// if the version count has been changed then we should update the version count
	// of the reference in the root space that will be used to determine how to set
	// variables within generated LPUs
	if (oldVersionCount != versionCount) {
		DataStructure *structure = getPrimarySource();
		structure->versionCount = versionCount;
	}
}

int DataStructure::getVersionCount() {
	DataStructure *structure = getPrimarySource();
	return structure->versionCount;
}

DataStructure *DataStructure::getPrimarySource() {
	if (source == NULL) return this;
	else return source->getPrimarySource();
}

