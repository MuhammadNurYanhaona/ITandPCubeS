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

//------------------------------------------------- DataStructure ---------------------------------------------------/

DataStructure::DataStructure(VariableDef *definition) {
	Assert(definition != NULL);
	this->definition = definition;
	this->source = NULL;
	this->dependents = new List<DataStructure*>;
	this->space = NULL;
}

DataStructure::DataStructure(DataStructure *source) {
	Assert(source != NULL);
	this->definition = NULL;
	this->source = source;
	this->dependents = new List<DataStructure*>;
	this->space = NULL;
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

//----------------------------------------------------- Token ---------------------------------------------------/

int Token::wildcardTokenId = -1;

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

const char *Space::RootSpaceName = "rootSpace";
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

DataStructure *Space::getLocalStructure(const char *name) {
	return dataStructureList->Lookup(name);
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
	return getRootSpace();
}
