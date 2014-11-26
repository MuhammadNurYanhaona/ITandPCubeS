#ifndef _H_partition_function
#define _H_partition_function

#include "task_space.h"
#include "location.h"
#include "ast_type.h"

class PartitionArg;

class SingleArgumentPartitionFunction : public PartitionFunctionConfig {
  public:
	SingleArgumentPartitionFunction(yyltype *location, const char *name) 
			: PartitionFunctionConfig(location, name) {} 
	void processArguments(List<PartitionArg*> *dividingArgs, List<PartitionArg*> *paddingArgs) {
		processArguments(dividingArgs, paddingArgs, "");
	}
	virtual void processArguments(List<PartitionArg*> *dividingArgs, 
			List<PartitionArg*> *paddingArgs, const char *argumentName);
};

class BlockSize : public SingleArgumentPartitionFunction {
  public:
	static const char *name;
	BlockSize(yyltype *location) : SingleArgumentPartitionFunction(location, name) {}
	void processArguments(List<PartitionArg*> *dividingArgs, 
			List<PartitionArg*> *paddingArgs, const char *argumentName);
	List<int> *getBlockedDimensions(Type *structureType);
};

class BlockCount : public SingleArgumentPartitionFunction {
  public:
	static const char *name;
	BlockCount(yyltype *location) : SingleArgumentPartitionFunction(location, name) {}
	void processArguments(List<PartitionArg*> *dividingArgs, 
			List<PartitionArg*> *paddingArgs, const char *argumentName);
	List<int> *getBlockedDimensions(Type *structureType);
};

class StridedBlock : public SingleArgumentPartitionFunction {
  public:
	static const char *name;
	StridedBlock(yyltype *location) : SingleArgumentPartitionFunction(location, name) {}
	void processArguments(List<PartitionArg*> *dividingArgs, 
			List<PartitionArg*> *paddingArgs, const char *argumentName);
};

class Strided : public PartitionFunctionConfig {
  public:
	static const char *name;
	Strided(yyltype *location) : PartitionFunctionConfig(location, name) {}
	void processArguments(List<PartitionArg*> *dividingArgs, List<PartitionArg*> *paddingArgs);
};

#endif
