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

