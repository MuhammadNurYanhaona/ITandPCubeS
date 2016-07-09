#include "task_space.h"
#include "../utils/list.h"
#include "../utils/hashtable.h"
#include "../utils/string_utils.h"
#include "../utils/code_constant.h"
#include "../utils/decorator_utils.h"
#include "../utils/string_utils.h"
#include "../syntax/ast.h"
#include "../syntax/ast_def.h"
#include "../syntax/location.h"
#include "partition_function.h"
#include "../syntax/errors.h"
#include "../static-analysis/usage_statistic.h"
#include "../codegen/space_mapping.h"
#include "symbol.h"

#include <deque>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <cstdlib>

//------------------------------------------------- DataDimensionConfig -----------------------------------------------/

DataDimensionConfig::DataDimensionConfig(int dimensionNo, Node *dividingArg) {
	this->dimensionNo = dimensionNo;
	this->dividingArg = dividingArg;
	this->frontPaddingArg = NULL;
	this->backPaddingArg = NULL;
}

DataDimensionConfig::DataDimensionConfig(int dimensionNo, Node *dividingArg, Node *frontPaddingArg,
                        Node *backPaddingArg) {
	this->dimensionNo = dimensionNo;
	this->dividingArg = dividingArg;
	this->frontPaddingArg = frontPaddingArg;
	this->backPaddingArg = backPaddingArg;
}

const char *DataDimensionConfig::getArgumentString(Node *arg, const char *prefix) {
	std::ostringstream stream;
	IntConstant *intConst = dynamic_cast<IntConstant*>(arg);
	if (intConst != NULL) {
		stream << intConst->getValue();
	} else {
		if (prefix != NULL) stream << prefix;
		Identifier *identifier = (Identifier*) arg;
		stream << identifier->getName();
	}
	return strdup(stream.str().c_str());
}

//---------------------------------------------- PartitionFunctionConfig ---------------------------------------------/

PartitionFunctionConfig::PartitionFunctionConfig(yyltype *location, const char *functionName) {
	this->location = location;
	this->functionName = functionName;
	this->arguments = new List<DataDimensionConfig*>;
}

void PartitionFunctionConfig::setDimensionIds(List<int> *dimensionIds) {
	if (dimensionIds->NumElements() != arguments->NumElements()) {
		ReportError::ArgumentCountMismatchInPartitionFunction(functionName);
	} else {
		for (int i = 0; i < dimensionIds->NumElements(); i++) {
			DataDimensionConfig *dataConfig = arguments->Nth(i);
			dataConfig->setDimensionNo(dimensionIds->Nth(i));
		}
	}
}

List<int> *PartitionFunctionConfig::getPartitionedDimensions() {
	List<int> *dimensionIds = new List<int>;
	for (int i = 0; i < arguments->NumElements(); i++) {
		dimensionIds->Append(arguments->Nth(i)->getDimensionNo());
	}
	return dimensionIds;
}

PartitionFunctionConfig *PartitionFunctionConfig::generateConfig(yyltype *location,
		const char *functionName,
                List<PartitionArg*> *dividingArgs,
                List<PartitionArg*> *paddingArgs) {

	PartitionFunctionConfig *config = NULL;
	if (strcmp(functionName, BlockSize::name) == 0) {
		config = new BlockSize(location);
	} else if (strcmp(functionName, BlockCount::name) == 0) {
		config = new BlockCount(location);
	} else if (strcmp(functionName, StridedBlock::name) == 0) {
		config = new StridedBlock(location);
	} else if (strcmp(functionName, Strided::name) == 0) {
		config = new Strided(location);
	} else {
		ReportError::UnknownPartitionFunction(location, functionName);
		config = new PartitionFunctionConfig(location, functionName);
	}

	config->processArguments(dividingArgs, paddingArgs);

	return config;	
}

void PartitionFunctionConfig::processArguments(List<PartitionArg*> *dividingArgs, 
		List<PartitionArg*> *paddingArgs) {
	int dividingArgsCount = dividingArgs->NumElements();
	for (int i = 0; i < dividingArgsCount; i++) {
		arguments->Append(new DataDimensionConfig(i + 1, NULL));
	}
}

bool PartitionFunctionConfig::hasOverlappingsAmongPartitions() {
	for (int i = 0; i < arguments->NumElements(); i++) {
		DataDimensionConfig *argument = arguments->Nth(i);
		if (argument->hasPadding()) return true;
	}
	return false;
}

List<int> *PartitionFunctionConfig::getOverlappingPartitionDims() {
	List<int> *dimensionList = new List<int>;
	for (int i = 0; i < arguments->NumElements(); i++) {
		DataDimensionConfig *argument = arguments->Nth(i);
		if (argument->hasPadding()) {
			dimensionList->Append(argument->getDimensionNo());
		}
	}
	return dimensionList;
}

DataDimensionConfig *PartitionFunctionConfig::getArgsForDimension(int dimensionNo) {
	for (int i = 0; i < arguments->NumElements(); i++) {
		DataDimensionConfig *argument = arguments->Nth(i);
		if (argument->getDimensionNo() == dimensionNo) return argument;
	}
	return NULL;	
}

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

void Space::genLpuCountInfoForGpuKernelExpansion(std::ofstream &stream, 
		const char *indentStr,
		List<const char*> *accessibleArrays,
		int topmostGpuPps, 
		bool perWarpCountInfo) {

	// for warp level mapping, there must be one count variable per warp; so this set up the variable suffix
	std::string countIndex = std::string("");
	std::string countSuffix = std::string("");
	if (perWarpCountInfo) {
		countIndex = std::string("[warpId]");
		countSuffix = std::string("[WARP_COUNT]");
	}

	// declare the lpu counter(s)
	stream << indentStr << "// determining LPU counts for Space " << id << "\n";	
	stream << indentStr << "__shared__ int " << "space" << id << "LpuCount" << countSuffix;
	stream << "[" << dimensions << "]" << stmtSeparator;

	// determine the current LPSes arrays that can be used to calculate the LPU count
	List<const char*> *localArrays = getLocallyUsedArrayNames();
	List<const char*> *usefulArrays = string_utils::intersectLists(accessibleArrays, localArrays);

	// let only one thread to do the count determination
	if (perWarpCountInfo) {
		stream << indentStr << "if (threadId == 0) {\n";
	} else {
		stream << indentStr << "if (warpId == 0 && threadId == 0) {\n";
	}

	// iterate over the LPS dimensions and calculate LPU count along individual dimensions separately
	for (int i = 0; i < dimensions; i++) {

		// get the LPS coordinate configuration for the current dimension (dimension numbering starts from 
		// one -- not for zero)
		Coordinate *coordinate = coordSys->getCoordinate(i + 1);
		
		// identify an array that has been partitioned along the current dimension in the current LPS
		ArrayDataStructure *partitioningArray = NULL;
		int alignmentDimension = i;
		for (int j = 0; j < usefulArrays->NumElements(); j++) {
			const char *arrayName = usefulArrays->Nth(j);
			Token *token = coordinate->getTokenForDataStructure(arrayName);
			if (token != NULL && !token->isWildcard()) {
				partitioningArray = (ArrayDataStructure *) token->getData();
				alignmentDimension = token->getDimensionId();
				break;
			}
		}

		// retrieve the partition function configuration for the array
		PartitionFunctionConfig *partitionFnConf 
			= partitioningArray->getPartitionSpecForDimension(alignmentDimension);

		// generate library routine call expression for the partition function
		const char *arrayName = partitioningArray->getName();
		Space *parentLps = partitioningArray->getSource()->getSpace();
		Space *parentSubpartition = parentLps->getSubpartition();
		if (parentSubpartition != NULL && parentSubpartition != this) {
			parentLps = parentSubpartition;
		}
		const char *parentLpsName = parentLps->getName();

		stream << indentStr << indent << "space" << id << "LpuCount";
		stream << countIndex << "[" << i << "] = ";
		
		// the first default argument is the array dimension range to be partitioned
		stream << partitionFnConf->getName() << "_part_count(&" << arrayName << "Space";
		stream << parentLpsName << "PRanges[" << alignmentDimension - 1 << "]" << paramSeparator;

		
		// the second default argument is the PPU count for the current space
		stream << paramIndent << indentStr << indent;
		if (topmostGpuPps == ppsId) stream << "1";
		else if (topmostGpuPps - ppsId == 1) stream << "SM_COUNT";
		else stream << "WARP_COUNT";

		// the remaining parameters depend on the configuration of the partition function itself; but if
		// exist the parameters follow a particular ordering
		DataDimensionConfig *partDimArgs = partitionFnConf->getArgsForDimension(alignmentDimension);
		Node *dividingArg = partDimArgs->getDividingArg();
		if (dividingArg != NULL)  {
			stream << paramSeparator;
			stream << DataDimensionConfig::getArgumentString(dividingArg, "partition.");			
		}
		Node *frontPaddingArg = partDimArgs->getFrontPaddingArg();
		if (frontPaddingArg != NULL) {
			stream << paramSeparator;
			stream << DataDimensionConfig::getArgumentString(frontPaddingArg, "partition.");			
		}
		Node *backPaddingArg = partDimArgs->getBackPaddingArg();
		if (backPaddingArg != NULL) {
			stream << paramSeparator;
			stream << DataDimensionConfig::getArgumentString(backPaddingArg, "partition.");			
		}

		stream << ")" << stmtSeparator;
	}
	
	stream << indentStr << "}\n";
	if (!perWarpCountInfo) {
		stream << indentStr << "__syncthreads()" << stmtSeparator;
	}
}

void Space::genArrayDimInfoForGpuKernelExpansion(std::ofstream &stream, 
		const char *indentStr,
		List<const char*> *arrayNames,
		int topmostGpuPps,
		bool perWarpCountInfo, bool perWarpDimensionInfo) {

	// for warp level mapping, there must be one array metadata variable per warp; so this set up the variable suffix
	std::string metadataIndex = std::string("");
	std::string metadataSuffix = std::string("");
	if (perWarpDimensionInfo) {
		metadataIndex = std::string("[warpId]");
		metadataSuffix = std::string("[WARP_COUNT]");
	}

	// determine the arrays among the accessed arrays that are part of the current LPS
	List<const char*> *localArrayNames = getLocallyUsedArrayNames();
	List<const char*> *localArrays = string_utils::intersectLists(arrayNames, localArrayNames);
	
	if (localArrays->NumElements() == 0) return;

	decorator::writeCommentHeader(&stream, "array part dimension calculation start", indentStr);

	// iterate over the local arrays and construct new part dimension information for each
	const char *lpsName = id;
	for (int i = 0; i < localArrays->NumElements(); i++) {
		
		const char *arrayName = localArrays->Nth(i);
		stream << "\n" << indentStr << "// array '" << arrayName << "\"\n";	
		
		ArrayDataStructure *array = (ArrayDataStructure *) getLocalStructure(arrayName);
		int arrayDimensions = array->getDimensionality();

		// determine the parent array dimension that this LPU will be dividing; notice the catch with the sub-
		// partitioned LPSes
		Space *parentLps = array->getSource()->getSpace();
		Space *parentSubpartition = parentLps->getSubpartition();
		if (parentSubpartition != NULL && parentSubpartition != this) {
			parentLps = parentSubpartition;
		}
		const char *parentArrayLpsName = parentLps->getName();

		std::string parentMetadataIndex = std::string("");
		if (topmostGpuPps - parentLps->getPpsId() == 2) {
			parentMetadataIndex = std::string("[WARP_ID]");
		}

		std::ostringstream dimRangeVar, parentDimRangeVar;
		dimRangeVar << arrayName << "Space" << lpsName << "PRanges" << metadataIndex;
		parentDimRangeVar << arrayName << "Space" << parentArrayLpsName;
		parentDimRangeVar << "PRanges" << parentMetadataIndex;
		
		// first declare one or more shared partition dimension metadata variables
		stream << indentStr << "__shared__ GpuDimension " << arrayName;
		stream << "Space" << lpsName << "PRanges";
		stream << metadataSuffix << "[" << arrayDimensions << "]" << stmtSeparator; 

		// let some specific thread(s) do the dimension range calculation
		if (perWarpDimensionInfo) {
			stream << indentStr << "if (threadId == 0) {\n";		
		} else {
			stream << indentStr << "if (warpId == 0 && threadId == 0) {\n";
		}

		// iterate over the array dimensions and process partition instruction for them one-by-one
		for (int j = 0; j < arrayDimensions; j++) {
		
			// if the array is not partitioned along a particular dimension then its part dimension ranges 
			// can be directly copied down from its source
			// again note that dimension numbering starts from 1 not from 0
			if (!array->isPartitionedAlongDimension(j + 1)) {
				stream << indentStr << indent << dimRangeVar.str();
				stream << "[" << j << "] = " << parentDimRangeVar.str();
				stream << "[" << j << "]" << stmtSeparator;
				continue;
			}
			
			// retrieve partition function configuration for the array dimension
			PartitionFunctionConfig *partFnConf = array->getPartitionSpecForDimension(j + 1);
			DataDimensionConfig *partDimArgs = partFnConf->getArgsForDimension(j + 1);

			// determine what dimension of the LPS the array dimension has been aligned to; this is needed
			// to identify the proper LPU/part ID and count to calculate the dimension ranges
			int alignedDimension = j;
			for (int k = 0; k < dimensions; k++) {
				Coordinate *coordinate = coordSys->getCoordinate(k + 1);
				Token *token = coordinate->getTokenForDataStructure(arrayName);
				if (token != NULL && token->getDimensionId() == j + 1) {
					alignedDimension = k;
					break;
				}
			}

			// invoke the dimension range determining library function for the partition function with
			// appropriate parameters
			
			// first two arguments are the current part and parent part dimension references
			stream << indentStr << indent << partFnConf->getName() << "_part_range(";
			stream << "&" << dimRangeVar.str() << "[" << j << "]" << paramSeparator;
			stream << paramIndent << indentStr << indent;
			stream << "&" << parentDimRangeVar.str() << "[" << j << "]" << paramSeparator;
			stream << paramIndent << indentStr << indent;
			
			// the next two argument are the LPU count and ID along the aligned dimension
			stream << "space" << lpsName << "LpuCount";
			if (perWarpCountInfo) stream << metadataIndex;
			stream << "[" << alignedDimension << "]" << paramSeparator;
			stream << "space" << lpsName << "LpuId" << metadataIndex; 
			stream << "[" << alignedDimension << "]";
			stream << paramIndent << indentStr << indent;

			// the remaining parameters are partition function specific but are applied in an specific order
			Node *dividingArg = partDimArgs->getDividingArg();
			if (dividingArg != NULL)  {
				stream << paramSeparator;
				stream << DataDimensionConfig::getArgumentString(dividingArg, "partition.");			
			}
			Node *frontPaddingArg = partDimArgs->getFrontPaddingArg();
			if (frontPaddingArg != NULL) {
				stream << paramSeparator;
				stream << DataDimensionConfig::getArgumentString(frontPaddingArg, "partition.");			
			}
			Node *backPaddingArg = partDimArgs->getBackPaddingArg();
			if (backPaddingArg != NULL) {
				stream << paramSeparator;
				stream << DataDimensionConfig::getArgumentString(backPaddingArg, "partition.");			
			}
			
			// close the function call
			stream << ")" << stmtSeparator;		
		}

		// close the if block that encompasses the dimension calculation
		stream << indentStr << "}\n";
	}
	
	if (!perWarpDimensionInfo) {
		stream << indentStr << "__syncthreads()" << stmtSeparator;
	}

	decorator::writeCommentHeader(&stream, "array part dimension calculation end", indentStr);
}

//-------------------------------------------- Partition Hierarchy -------------------------------------------------/

PartitionHierarchy::PartitionHierarchy() {
	spaceHierarchy = new Hashtable<Space*>;
	pcubesModel = NULL;
}

void PartitionHierarchy::setPCubeSModel(PCubeSModel *pcubesModel) {
	this->pcubesModel = pcubesModel;
}

PCubeSModel *PartitionHierarchy::getPCubeSModel() { return pcubesModel; }
        
bool PartitionHierarchy::isMappedToHybridModel() {
	if (pcubesModel == NULL) return false;
	return (pcubesModel->getModelType() == PCUBES_MODEL_TYPE_HYBRID);
}

Space *PartitionHierarchy::getSpace(char spaceId) {
	char *key = (char *) malloc(sizeof(char) * 2);
	key[0] = spaceId;
	key[1] = '\0';
	Space *space = spaceHierarchy->Lookup(key);
	free(key);
	return space;
}

Space *PartitionHierarchy::getSubspace(char spaceId) {
	int suffixLength = strlen(Space::SubSpaceSuffix);
	char *key = (char *) malloc(sizeof(char) * (suffixLength + 2));
	key[0] = spaceId;
	key[1] = '\0';
	strcat(key, Space::SubSpaceSuffix);
	Space *space = spaceHierarchy->Lookup(key);
	free(key);
	return space;
}

Space *PartitionHierarchy::getRootSpace() {
	return spaceHierarchy->Lookup(Space::RootSpaceName);
}

bool PartitionHierarchy::addNewSpace(Space *space) {
	Space *duplicateSpace = spaceHierarchy->Lookup(space->getName());
	bool duplicateFound = false;
	if (duplicateSpace != NULL) duplicateFound = true;
	spaceHierarchy->Enter(space->getName(), space, false);
	return !duplicateFound;
}

Space *PartitionHierarchy::getCommonAncestor(Space *space1, Space *space2) {
	if (space1 == space2) return space1;
	if (space1->isParentSpace(space2)) return space2;
	if (space2->isParentSpace(space1)) return space1;
	Space *nextSpace = space1;
	while ((nextSpace = nextSpace->getParent()) != NULL) {
		if (space2->isParentSpace(nextSpace)) return nextSpace;
	}
	return getRootSpace();
}

void PartitionHierarchy::performAllocationAnalysis(int segmentedPPS) {
	
	// determine if the current task's LPS hierarchy has been mapped to a hybrid PCubeS model
	// and if yes, determine the top-most PPS ID for the GPU
	bool hybridModel = (pcubesModel->getModelType() ==  PCUBES_MODEL_TYPE_HYBRID);
	int gpuPpsId = -1;
	if (hybridModel) {
		gpuPpsId = pcubesModel->getGpuTransitionSpaceId();
	}

	Space *root = getRootSpace();
	List<const char*> *variableList = root->getLocalDataStructureNames();
	
	// instruction to allocate all scalar and non-partitionable variables in the root LPS
	for (int i = 0; i < variableList->NumElements(); i++) {
		DataStructure *variable = root->getStructure(variableList->Nth(i));
		ArrayDataStructure *array = dynamic_cast<ArrayDataStructure*>(variable);
		if (array == NULL) {
			variable->setAllocator(root);
			variable->getUsageStat()->flagAllocated();
		}
	}

	// then do breadth first search in the partition hierarchy to make allocation decisions
	std::deque<Space*> lpsQueue;
        lpsQueue.push_back(root);
	while (!lpsQueue.empty()) {
		
		Space *lps = lpsQueue.front();
                lpsQueue.pop_front();

		// setup the segmented PPS property in the LPS here since we are traversing all
		// LPSes and have the information for the setup
		lps->setSegmentedPPS(segmentedPPS);

		List<Space*> *children = lps->getChildrenSpaces();	
                for (int i = 0; i < children->NumElements(); i++) {
                        lpsQueue.push_back(children->Nth(i));
                }
		if (lps->getSubpartition() != NULL) lpsQueue.push_back(lps->getSubpartition());

		int ppsId = lps->getPpsId();

		// if the current LPS is mapped at or below the PPS designating host to GPU 
		// transition then we do GPU specific allocation analysis
		if (ppsId <= gpuPpsId) {
			performAllocAnalysisOnLpsMappedToGpu(lps, gpuPpsId);
			continue;
		}
		
		// iterate over all the arrays of current LPS and consider only those for memory
		// allocation that have been used within any of LPS's compute stages.
		variableList = lps->getLocallyUsedArrayNames();
		for (int i = 0; i < variableList->NumElements(); i++) {
			DataStructure *variable = lps->getLocalStructure(variableList->Nth(i));
			LPSVarUsageStat *stat = variable->getUsageStat();
			if (!(stat->isAccessed() || stat->isReduced())) continue;
			
			// if there are boundary overlappings among the partitions of the array
			// in this LPS then memory need to be allocated for this variable
			ArrayDataStructure *array = (ArrayDataStructure*) variable;
			bool hasOverlapping = array->hasOverlappingsAmongPartitions();
			
			// check if the variable has been allocated before in any ancestor LPS
			DataStructure *lastAllocation = array->getClosestAllocation();

			// if there is no previous allocation for this structure then it should
			// be allocated
			bool notAllocatedBefore = (lastAllocation == NULL);

			// if the structure has been allocated before then a checking should be
			// done to see if the array has been reordered since last allocation. If
			// it has been reordered then again a new allocation is needed.
			bool reordered = false; 
			if (lastAllocation != NULL) {
				Space *allocatingSpace = lastAllocation->getSpace();
				reordered = array->isReorderedAfter(allocatingSpace);
			}

			// even if the array has not been reordered since last allocation, if its 
			// last allocation was above the segmented PPS layer and on a different 
			// PPS than the current LPS has been mapped to then, again, it should be 
			// allocated
			bool lastAllocInaccessible = false;
			if (lastAllocation != NULL) {
				Space *allocatingSpace = lastAllocation->getSpace();
				int lastPpsId = allocatingSpace->getPpsId();
				lastAllocInaccessible = (lastPpsId > segmentedPPS) 
						&& (ppsId != lastPpsId);
			}

			if (hasOverlapping || notAllocatedBefore 
					|| reordered || lastAllocInaccessible) {
				array->setAllocator(lps);
				array->getUsageStat()->flagAllocated();
			}
		}	
	}

	// finally do another breadth first search to set up appropriate allocation references for
	// data structures in LPSes that do not allocate them themselves.			
        lpsQueue.push_back(root);
	while (!lpsQueue.empty()) {
		Space *lps = lpsQueue.front();
                lpsQueue.pop_front();
		List<Space*> *children = lps->getChildrenSpaces();	
                for (int i = 0; i < children->NumElements(); i++) {
                        lpsQueue.push_back(children->Nth(i));
                }
		if (lps->getSubpartition() != NULL) lpsQueue.push_back(lps->getSubpartition());
		
		variableList = lps->getLocalDataStructureNames();
		for (int i = 0; i < variableList->NumElements(); i++) {
			DataStructure *structure = lps->getLocalStructure(variableList->Nth(i));
			// if the structure is not allocated then try to find a source reference
			// in some ancestor LPS where it has been allocated (this sets up the 
			// forward pointer from a lower to upper LPS)
			if (!structure->getUsageStat()->isAllocated()) {
				DataStructure *lastAllocation = structure->getClosestAllocation();
				if (lastAllocation != NULL) {
					structure->setAllocator(lastAllocation->getSpace());
				}
			// on the other hand, if the structure has been allocated in current LPS
			// then this allocation can be used to set up back references to ancestor
			// LPSes that neither allocate this structure themselves nor have any
			// forward reference to their own ancestor LPSes been set for structure 	
			} else {
				DataStructure *source = structure->getSource();
				while (source != NULL && source->getAllocator() == NULL) {
					source->setAllocator(lps);
					source = source->getSource();
				}
			}
		}
	}
}

void PartitionHierarchy::performAllocAnalysisOnLpsMappedToGpu(Space *lps, int gpuPpsId) {
	
	Space *parentLps = lps->getParent();
	Space *rootLps = getRootSpace();

	bool topmostLpsMappedToGpu = false;
	if (parentLps == rootLps || parentLps->getPpsId() > gpuPpsId) {
		topmostLpsMappedToGpu = true;
	}

	// if the current LPS is the topmost LPS in its hierarchy that has been been mapped inside the GPU then we flag 
	// all data structures specified in its partition configuration for allocation
	if (topmostLpsMappedToGpu) {
		
		// an special case for allocation occurs when the topmost GPU LPS is sub-partitioned; then we allocate the
		// non sub-partitioned variables in the parent LPS and the rest in the sub-partition LPS. This is needed 
		// to restrict the data parts from being too large so much so that a part may not even fit in the GPU card.   
		List<const char*> *variableList = lps->getLocallyUsedArrayNames();
		List<const char*> *subpartitionedVarList = new List<const char*>;
		Space *subpartition = lps->getSubpartition();
		if (subpartition != NULL) {
			subpartitionedVarList = subpartition->getLocalDataStructureNames();
		}

		List<const char*> *topmostLpsVars = string_utils::subtractList(variableList, subpartitionedVarList);
		for (int i = 0; i < topmostLpsVars->NumElements(); i++) {
			DataStructure *variable = lps->getLocalStructure(topmostLpsVars->Nth(i));
			variable->setAllocator(lps);
                        variable->getUsageStat()->flagAllocated();
			
			// flag the variable as been accessed in this LPS to ensure that the LPU generation routine does
			// not ignore putting in proper data part reference into the LPU if it has not been accessed here.
			if (!variable->getUsageStat()->isAccessed()) {
				variable->getUsageStat()->addAccess();
			} 
		}

		for (int i = 0; i < subpartitionedVarList->NumElements(); i++) {
			DataStructure *variable = subpartition->getLocalStructure(subpartitionedVarList->Nth(i));
			variable->setAllocator(subpartition);
                        variable->getUsageStat()->flagAllocated();
			if (!variable->getUsageStat()->isAccessed()) {
				variable->getUsageStat()->addAccess();
			} 
		}	

	// if the current LPS is not the topmost LPS in its hierarchy then we validate that all variables used in it have 
	// already been allocated in its ancestor top-most GPU level LPS
	} else {
		Space *ancestorGpuLps = parentLps;
		while (parentLps != rootLps && parentLps->getPpsId() <= gpuPpsId) {
			ancestorGpuLps = parentLps;
			parentLps = parentLps->getParent();
		}
		Space *ancestorSubpartition = ancestorGpuLps->getSubpartition();		
		const char *ancestorLpsName = ancestorGpuLps->getName();

		List<const char*> *variableList = lps->getLocallyUsedArrayNames();
		for (int i = 0; i < variableList->NumElements(); i++) {
			DataStructure *variable = lps->getLocalStructure(variableList->Nth(i));
			const char *varName = variable->getName();
			DataStructure *lastAllocation = variable->getClosestAllocation();
			if (lastAllocation != NULL) {
				Space *allocatingSpace = lastAllocation->getSpace();
				if (allocatingSpace != ancestorGpuLps 
						&& allocatingSpace != ancestorSubpartition) {
					ReportError::UnsupportedGpuPartitionConfiguration(varName, ancestorLpsName);
				}
			} else {
				ReportError::UnsupportedGpuPartitionConfiguration(varName, ancestorLpsName);
			}
		}
	}
}
