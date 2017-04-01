#include "../partition_function.h"
#include "../task_space.h"
#include "../../common/errors.h"
#include "../../syntax/ast.h"
#include "../../syntax/ast_expr.h"
#include "../../syntax/ast_partition.h"
#include "../../../../common-libs/utils/list.h"

#include <iostream>
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
			} else if (dividingArgsCount == 2 * paddingArgsCount) {
				for (int i = 0; i < dividingArgsCount; i++) {
					Node *actualArg = dividingArgs->Nth(i)->getContent();
					DataDimensionConfig *dataConfig = new DataDimensionConfig(i + 1, actualArg);
					Node *padding = paddingArgs->Nth(i/2)->getContent();
					dataConfig->setPaddingArg(padding);
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
