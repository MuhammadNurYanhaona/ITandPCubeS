#include "task_space.h"
#include "list.h"
#include "hashtable.h"
#include "ast.h"
#include "ast_def.h"
#include "location.h"
#include "partition_function.h"
#include "errors.h"

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
}

DataStructure::DataStructure(DataStructure *source) {
	Assert(source != NULL);
	this->definition = NULL;
	this->source = source;
	this->dependents = new List<DataStructure*>;
	this->space = NULL;
	this->nonStorable = false;
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

int ArrayDataStructure::getDimensionality() {
	if (source != NULL) {
		return ((ArrayDataStructure*) source)->getDimensionality(); 
	} else return sourceDimensions->NumElements();
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
const char *Space::SubSpaceSuffix = "_subspace";

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

//-------------------------------------------- Partition Hierarchy -------------------------------------------------/

PartitionHierarchy::PartitionHierarchy() {
	spaceHierarchy = new Hashtable<Space*>;
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
