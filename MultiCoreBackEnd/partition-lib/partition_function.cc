#include "../syntax/ast.h"
#include "../syntax/ast_expr.h"
#include "../syntax/ast_partition.h"
#include "../semantics/task_space.h"
#include "../utils/list.h"
#include "../syntax/errors.h"
#include "partition_function.h"

#include <sstream>

//-------------------------------------------- Single Argument Function -------------------------------------------/

void SingleArgumentPartitionFunction::processArguments(List<PartitionArg*> *dividingArgs, 
		List<PartitionArg*> *paddingArgs, const char *argumentName) {
	
	if (dividingArgs == NULL || dividingArgs->NumElements() == 0) {
		ReportError::ArgumentMissingInPartitionFunction(location, functionName, argumentName);
	} else {
		// Note that it might seem confusing at first that a single argument partition function supporting
		// multiple arguments. The answer is each argument here is used for a different dimension of the
		// underlying data structure. Single argument here means one argument par partitionable dimension.
		int dividingArgsCount = dividingArgs->NumElements();
		if (paddingArgs == NULL || paddingArgs->NumElements() == 0) {
			for (int i = 0; i < dividingArgsCount; i++) {
				Node *actualArg = dividingArgs->Nth(i)->getContent();
				arguments->Append(new DataDimensionConfig(i + 1, actualArg));
			}
		} else {
			int paddingArgsCount  = paddingArgs->NumElements();
			if (dividingArgsCount == paddingArgsCount) {
				for (int i = 0; i < dividingArgsCount; i++) {
					Node *actualArg = dividingArgs->Nth(i)->getContent();
					// the default dimension number assigned here gets updated with appropriate
					// number during validation.
					DataDimensionConfig *dataConfig = new DataDimensionConfig(i + 1, actualArg);
					Node *actualPadding = paddingArgs->Nth(i)->getContent();
					dataConfig->setPaddingArg(actualPadding);
					arguments->Append(dataConfig);
				}
			} else if (dividingArgsCount * 2 == paddingArgsCount) {
				for (int i = 0; i < dividingArgsCount; i++) {
					Node *actualArg = dividingArgs->Nth(i)->getContent();
					DataDimensionConfig *dataConfig = new DataDimensionConfig(i + 1, actualArg);
					Node *frontPadding = paddingArgs->Nth(i * 2)->getContent();
					Node *backPadding = paddingArgs->Nth(i * 2 + 1)->getContent();
					dataConfig->setPaddingArg(frontPadding, backPadding);
					arguments->Append(dataConfig);
				}
			} else {
				ReportError::InvalidPadding(location, functionName);
				for (int i = 0; i < dividingArgsCount; i++) {
					Node *actualArg = dividingArgs->Nth(i)->getContent();
					arguments->Append(new DataDimensionConfig(i + 1, actualArg));
				}
			}
		}	
	}
}

//---------------------------------------------------- Block Size ------------------------------------------------/

const char *BlockSize::name = "block_size";

void BlockSize::processArguments(List<PartitionArg*> *dividingArgs, 
		List<PartitionArg*> *paddingArgs, const char *argumentName) {
	SingleArgumentPartitionFunction::processArguments(dividingArgs, paddingArgs, "block size");
}

List<int> *BlockSize::getBlockedDimensions(Type *structureType) {
	List<int> *blockedDimensions = new List<int>;
	for (int i = 0; i < arguments->NumElements(); i++) {
		DataDimensionConfig *dataConfig = arguments->Nth(i);
		if (dataConfig->hasPadding()) continue;
		Node *dividingArg = dataConfig->getDividingArg();
		IntConstant *constant = dynamic_cast<IntConstant*>(dividingArg);
		if (constant != NULL && constant->getValue() == 1) {
			blockedDimensions->Append(dataConfig->getDimensionNo());
		}
	}
	return blockedDimensions;
}

//--------------------------------------------------- Block Count ------------------------------------------------/

const char *BlockCount::name = "block_count";

void BlockCount::processArguments(List<PartitionArg*> *dividingArgs, 
		List<PartitionArg*> *paddingArgs, const char *argumentName) {
	SingleArgumentPartitionFunction::processArguments(dividingArgs, paddingArgs, "block count");
}

List<int> *BlockCount::getBlockedDimensions(Type *structureType) {
	return new List<int>;
}

//--------------------------------------------------- Strided Block ----------------------------------------------/

const char *StridedBlock::name = "block_stride";

void StridedBlock::processArguments(List<PartitionArg*> *dividingArgs, 
		List<PartitionArg*> *paddingArgs, const char *argumentName) {
	if (paddingArgs != NULL && paddingArgs->NumElements() != 0) {
		ReportError::PaddingArgumentsNotSupported(location, functionName);	
	}
	SingleArgumentPartitionFunction::processArguments(dividingArgs, paddingArgs, "block size");
}

const char *StridedBlock::getTransformedIndex(int dimensionNo, const char *origIndexName, bool copyMode) {
	
	DataDimensionConfig *argument = getArgsForDimension(dimensionNo);
	Node *dividingArg = argument->getDividingArg();
	bool argNeeded = false;
	int argValue = 0;
	const char *argName = NULL;
	Identifier *identifier = dynamic_cast<Identifier*>(dividingArg);
	if (identifier != NULL) {
		argNeeded = true;
		argName = identifier->getName();
	} else {
		IntConstant *constant = dynamic_cast<IntConstant*>(dividingArg);
		argValue = constant->getValue();
	}

	std::ostringstream expr;
	expr << "((" << origIndexName << " / (";
	expr << "partConfig.count " << " * ";
	if (argNeeded) {
		expr << "partition." << argName;
		expr << "))";
		expr << " * partition." << argName;
		expr << " + " << origIndexName << " % " << "partition." << argName << ")";
	} else {
		expr << argValue;
		expr << "))";
		expr << " * " << argValue;
		expr << " + " << origIndexName << " % " << argValue << ")";
	}

	return strdup(expr.str().c_str());	
}

const char *StridedBlock::getOriginalIndex(int dimensionNo, const char *xformIndexName, bool copyMode) {
	
	DataDimensionConfig *argument = getArgsForDimension(dimensionNo);
	Node *dividingArg = argument->getDividingArg();
	std::ostringstream sizeParam;
	Identifier *identifier = dynamic_cast<Identifier*>(dividingArg);
	if (identifier != NULL) {
		sizeParam << "partition.";
		sizeParam << identifier->getName();
	} else {
		IntConstant *constant = dynamic_cast<IntConstant*>(dividingArg);
		sizeParam << constant->getValue();
	}

	std::ostringstream expr;
	expr << "((" << xformIndexName << " / " << sizeParam.str() << ")";
	expr << " * partConfig.count";
	expr << " + partConfig.index) * " << sizeParam.str();
	expr << " + " << xformIndexName << " % " << sizeParam.str();
	return strdup(expr.str().c_str());	
}

const char *StridedBlock::getInclusionTestExpr(int dimensionNo, const char *origIndexName, bool copyMode) {

	DataDimensionConfig *argument = getArgsForDimension(dimensionNo);
	Node *dividingArg = argument->getDividingArg();
	std::ostringstream sizeParam;
	Identifier *identifier = dynamic_cast<Identifier*>(dividingArg);
	if (identifier != NULL) {
		sizeParam << "partition.";
		sizeParam << identifier->getName();
	} else {
		IntConstant *constant = dynamic_cast<IntConstant*>(dividingArg);
		sizeParam << constant->getValue();
	}

	std::ostringstream expr;
	expr << "(" << origIndexName << " % (";
	expr << sizeParam.str() << " * partConfig.count))";
	expr << " / " << sizeParam.str() << " == partConfig.index";
	return strdup(expr.str().c_str());	
}

const char *StridedBlock::getImpreciseBoundOnXformedIndex(int dimensionNo, 
		const char *index, bool lowerBound, bool copyMode) {
	
	DataDimensionConfig *argument = getArgsForDimension(dimensionNo);
	Node *dividingArg = argument->getDividingArg();
	std::ostringstream sizeParam;
	Identifier *identifier = dynamic_cast<Identifier*>(dividingArg);
	if (identifier != NULL) {
		sizeParam << "partition.";
		sizeParam << identifier->getName();
	} else {
		IntConstant *constant = dynamic_cast<IntConstant*>(dividingArg);
		sizeParam << constant->getValue();
	}

	std::ostringstream expr;
	expr << "(" << index << " / (";
	expr << "partConfig.count " << " * " << sizeParam.str();
	expr << ")";
	if (!lowerBound) expr << " + 1";  
	expr << ") * " << sizeParam.str();
	return strdup(expr.str().c_str());	
}

//----------------------------------------------------- Strided --------------------------------------------------/

const char *Strided::name = "stride";

void Strided::processArguments(List<PartitionArg*> *dividingArgs, List<PartitionArg*> *paddingArgs) {
	
	if (dividingArgs != NULL && dividingArgs->NumElements() != 0) {
		ReportError::PartitionArgumentsNotSupported(location, functionName);	
	}
	if (paddingArgs != NULL && paddingArgs->NumElements() != 0) {
		ReportError::PaddingArgumentsNotSupported(location, functionName);	
	}
	arguments->Append(new DataDimensionConfig(1, NULL));
}

const char *Strided::getTransformedIndex(int dimensionNo, const char *origIndexName, bool copyMode) {
	std::ostringstream expr;
	expr << "(" << origIndexName << " / " << "partConfig.count)";
	return strdup(expr.str().c_str());
}

const char *Strided::getOriginalIndex(int dimensionNo, const char *xformIndexName, bool copyMode) {
	std::ostringstream expr;
	expr << "(" << "partConfig.index" <<  " + ";
	expr << xformIndexName << " * " << "partConfig.count" << ")";
	return strdup(expr.str().c_str());
}

const char *Strided::getInclusionTestExpr(int dimensionNo, const char *origIndexName, bool copyMode) {
	std::ostringstream expr;
	expr << "(" << origIndexName << " % " << "partConfig.count == partConfig.index)";
	return strdup(expr.str().c_str());
}

const char *Strided::getImpreciseBoundOnXformedIndex(int dimensionNo, 
		const char *origIndexName, bool lowerBound, bool copyMode) {
	std::ostringstream expr;
	expr << "(" << origIndexName << " / " << "partConfig.count";
	if (!lowerBound) expr << " + 1";
	expr << ")";
	return strdup(expr.str().c_str());
}
