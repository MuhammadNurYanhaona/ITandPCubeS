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

//--------------------------------------------- ArrayDataStructure -----------------------------------------------/

ArrayDataStructure::ArrayDataStructure(VariableDef *definition) : DataStructure(definition) {

	ArrayType *type = (ArrayType*) definition->getType();
	int dimensionCount = type->getDimensions();
	sourceDimensions = new List<int>;
	for (int i = 0; i < dimensionCount; i++) {
		sourceDimensions->Append(i + 1);
	}
	afterPartitionDimensions = new List<int>;
	afterPartitionDimensions->AppendAll(sourceDimensions);
	partitionSpecs = new List<PartitionFunctionConfig*>;
}

ArrayDataStructure::ArrayDataStructure(ArrayDataStructure *source) : DataStructure(source) {
	
	sourceDimensions = new List<int>;
	sourceDimensions->AppendAll(source->getRemainingDimensions());
	partitionSpecs = new List<PartitionFunctionConfig*>;
	
	afterPartitionDimensions = new List<int>;
	afterPartitionDimensions->AppendAll(sourceDimensions);
}

void ArrayDataStructure::addPartitionSpec(PartitionFunctionConfig *partitionConfig) {
	partitionSpecs->Append(partitionConfig);
}

bool ArrayDataStructure::isOrderDependent() {
	for (int i = 0; i < partitionSpecs->NumElements(); i++) {
		if (partitionSpecs->Nth(i)->isOrdered()) return true;
	}
	return false;
}

bool ArrayDataStructure::hasOverlappingsAmongPartitions() {
	for (int i = 0; i < partitionSpecs->NumElements(); i++) {
		PartitionFunctionConfig *config = partitionSpecs->Nth(i);
		if (config->hasOverlappingsAmongPartitions()) return true;
	}
	return false;
}

List<int> *ArrayDataStructure::getOverlappingPartitionDims() {
	List<int> *dimensionList = new List<int>;
	for (int i = 0; i < partitionSpecs->NumElements(); i++) {
		PartitionFunctionConfig *config = partitionSpecs->Nth(i);
		if (config->hasOverlappingsAmongPartitions()) {
			dimensionList->AppendAll(config->getOverlappingPartitionDims());
		}
	}
	return dimensionList;
}

PartitionFunctionConfig *ArrayDataStructure::getPartitionSpecForDimension(int dimensionNo) {
	for (int i = 0; i < partitionSpecs->NumElements(); i++) {
		PartitionFunctionConfig *config = partitionSpecs->Nth(i);
		List<int> *dataDimensions = config->getPartitionedDimensions();
		for (int j = 0; j < dataDimensions->NumElements(); j++) {
			if (dataDimensions->Nth(j) == dimensionNo) return config;
		}
	}
	return NULL;
}

bool ArrayDataStructure::isPartitionedAlongDimension(int dimensionNo) {
	PartitionFunctionConfig *partConfig = getPartitionSpecForDimension(dimensionNo);
	return partConfig != NULL;
}

bool ArrayDataStructure::isPartitionedEarlier() {
	if (source == NULL) return false;
	ArrayDataStructure *parentArray = (ArrayDataStructure*) source;
	if (parentArray->isPartitioned()) return true;
	return parentArray->isPartitionedEarlier();
}

bool ArrayDataStructure::isPartitionedAlongDimensionEarlier(int dimensionNo) {
	if (source == NULL) return false;
	ArrayDataStructure *parentArray = (ArrayDataStructure*) source;
	if (parentArray->isPartitionedAlongDimension(dimensionNo)) return true;
	return parentArray->isPartitionedAlongDimensionEarlier(dimensionNo);
}

int ArrayDataStructure::getDimensionality() {
	if (source != NULL) {
		return ((ArrayDataStructure*) source)->getDimensionality(); 
	} else return sourceDimensions->NumElements();
}

bool ArrayDataStructure::doesGenerateOverlappingParts() {
	if (hasOverlappingsAmongPartitions()) return true;
	if (source == NULL) return false;
	ArrayDataStructure *parentArray = (ArrayDataStructure *) source;
	return parentArray->doesGenerateOverlappingParts();
}

bool ArrayDataStructure::isDimensionReordered(int dimensionNo, Space *comparisonBound) {
	if (isDimensionLocallyReordered(dimensionNo)) return true;
	if (this->space == comparisonBound) return false;
	if (source == NULL) return false;
	ArrayDataStructure *sourceArray = (ArrayDataStructure*) source;
	if (source->getSpace() == comparisonBound) {
		return sourceArray->isDimensionLocallyReordered(dimensionNo);
	}
	else if (source->getSpace()->isParentSpace(comparisonBound)) {
		return sourceArray->isDimensionReordered(dimensionNo, comparisonBound);		
	} else return false;
}

bool ArrayDataStructure::isDimensionLocallyReordered(int dimensionNo) {
	PartitionFunctionConfig *partConfig = getPartitionSpecForDimension(dimensionNo);
	if (partConfig == NULL) return false;
	return partConfig->doesReorderStoredData();
}

bool ArrayDataStructure::isLocallyReordered() {
	for (int i = 0; i < sourceDimensions->NumElements(); i++) {
		int dimensionNo = sourceDimensions->Nth(i);
		if (isDimensionLocallyReordered(dimensionNo)) return true;
	}
	return false;
}

bool ArrayDataStructure::isReordered(Space *comparisonBound) {
	if (isLocallyReordered()) return true;
	if (this->space == comparisonBound) return false;
        if (source == NULL) return false;
	ArrayDataStructure *sourceArray = (ArrayDataStructure*) source;
	if (source->getSpace() == comparisonBound) {
		return sourceArray->isLocallyReordered();
	}
	else if (source->getSpace()->isParentSpace(comparisonBound)) {
		return sourceArray->isReordered(comparisonBound);		
	} else return false;
}

bool ArrayDataStructure::isReorderedAfter(Space *allocatorSpace) {
	if (this->space == allocatorSpace) return false;
	if (isLocallyReordered()) return true;
        if (source == NULL) return false;
	ArrayDataStructure *sourceArray = (ArrayDataStructure*) source;
	if (source->getSpace()->isParentSpace(allocatorSpace)) {
		return sourceArray->isReorderedAfter(allocatorSpace);		
	} else return false;
}

bool ArrayDataStructure::isSingleEntryInDimension(int dimensionNo) {
	for (int i = 0; i < afterPartitionDimensions->NumElements(); i++) {
		if (afterPartitionDimensions->Nth(i) == dimensionNo) return false;
	}
	return true;
}

void ArrayDataStructure::print() {
	std::cout << "Variable " << getName() << " coordinate arrangements:\n";
	for (int i = 0; i < sourceDimensions->NumElements(); i++) {
		int dimensionNo = sourceDimensions->Nth(i);
		PartitionFunctionConfig *config = getPartitionSpecForDimension(dimensionNo);
		if (config == NULL) {
			std::cout << "\treplicated in Dimension: " << dimensionNo;
		} else {
			std::cout << "\tpartitioned in Dimension: " << dimensionNo;
		}
		std::cout << std::endl;
	}
}

const char *ArrayDataStructure::getIndexXfromExpr(int dimensionNo, const char *indexName) {
        PartitionFunctionConfig *partConfig = getPartitionSpecForDimension(dimensionNo);
        if (partConfig == NULL) return NULL;
        bool copiedInLps = usageStat->isAllocated();
        return partConfig->getTransformedIndex(dimensionNo, indexName, copiedInLps);
}

const char *ArrayDataStructure::getReorderedInclusionCheckExpr(int dimensionNo, const char *indexName) {
        PartitionFunctionConfig *partConfig = getPartitionSpecForDimension(dimensionNo);
        if (partConfig == NULL) return NULL;
        bool copiedInLps = usageStat->isAllocated();
        return partConfig->getInclusionTestExpr(dimensionNo, indexName, copiedInLps);
}

const char *ArrayDataStructure::getReverseXformExpr(int dimensionNo, const char *xformIndex) {
        PartitionFunctionConfig *partConfig = getPartitionSpecForDimension(dimensionNo);
        if (partConfig == NULL) return NULL;
        bool copiedInLps = usageStat->isAllocated();
        return partConfig->getOriginalIndex(dimensionNo, xformIndex, copiedInLps);
}

const char *ArrayDataStructure::getImpreciseBoundOnXformedIndex(int dimensionNo,
                const char *indexName, bool lowerBound, int indent) {
        PartitionFunctionConfig *partConfig = getPartitionSpecForDimension(dimensionNo);
        if (partConfig == NULL) return NULL;
        bool copiedInLps = usageStat->isAllocated();
        return partConfig->getImpreciseBoundOnXformedIndex(dimensionNo,
                        indexName, lowerBound, copiedInLps, indent);
}

