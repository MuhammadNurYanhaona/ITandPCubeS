#include "ast_partition.h"
#include "task_space.h"
#include "list.h"
#include "errors.h"
#include "partition_function.h"

//-------------------------------------------- Single Argument Function -------------------------------------------/

void SingleArgumentPartitionFunction::processArguments(List<PartitionArg*> *dividingArgs, 
		List<PartitionArg*> *paddingArgs, const char *argumentName) {
	
	if (dividingArgs == NULL || dividingArgs->NumElements() == 0) {
		ReportError::ArgumentMissingInPartitionFunction(location, functionName, argumentName);
	} else {
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

const char *StridedBlock::name = "strided_block";

void StridedBlock::processArguments(List<PartitionArg*> *dividingArgs, 
		List<PartitionArg*> *paddingArgs, const char *argumentName) {
	SingleArgumentPartitionFunction::processArguments(dividingArgs, paddingArgs, "block size");
}

//----------------------------------------------------- Strided --------------------------------------------------/

const char *Strided::name = "strided";

void Strided::processArguments(List<PartitionArg*> *dividingArgs, List<PartitionArg*> *paddingArgs) {
	
	if (dividingArgs != NULL && dividingArgs->NumElements() != 0) {
		ReportError::PartitionArgumentsNotSupported(location, functionName);	
	}
	if (paddingArgs != NULL && paddingArgs->NumElements() != 0) {
		ReportError::PaddingArgumentsNotSupported(location, functionName);	
	}
	arguments->Append(new DataDimensionConfig(1, NULL));
}
