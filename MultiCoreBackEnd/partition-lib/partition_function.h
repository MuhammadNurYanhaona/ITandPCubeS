#ifndef _H_partition_function
#define _H_partition_function

#include "../semantics/task_space.h"
#include "../syntax/location.h"
#include "../syntax/ast_type.h"

/*	This header file lists the configurations of built-in partition functions. Currently only
	a few partition functions are included. Our future plan is to include user defined partition
	functions.
	
	Note that these classes hold only configuration information regarding corresponding
	partition functions. Such information is used to validate the partition specification for a 
	task where a partition function is been used. Actual implementation of a partition function 
	should depend on the target hardware platform.

	We hope to have a set of configuration methods and properties that are sufficient for 
	constructing user defined partition functions. 
*/

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
	bool doesSupportGhostRegion() { return true; }
};

class BlockCount : public SingleArgumentPartitionFunction {
  public:
	static const char *name;
	BlockCount(yyltype *location) : SingleArgumentPartitionFunction(location, name) {}
	void processArguments(List<PartitionArg*> *dividingArgs, 
			List<PartitionArg*> *paddingArgs, const char *argumentName);
	List<int> *getBlockedDimensions(Type *structureType);
	bool doesSupportGhostRegion() { return true; }
};

class StridedBlock : public SingleArgumentPartitionFunction {
  public:
	static const char *name;
	StridedBlock(yyltype *location) : SingleArgumentPartitionFunction(location, name) {}
	void processArguments(List<PartitionArg*> *dividingArgs, 
			List<PartitionArg*> *paddingArgs, const char *argumentName);
	bool doesReorderStoredData() { return true; }

	// Currently the third argument is not considered in the implementation of these functions
	// as we are not copying data for different LPSes; rather we are assigning the reference to
	// a single allocation -- made for the Root Space -- to all other spaces. TODO we need to
	// include copy mode into consideration for optimized compiler implementations as a lot of
	// time copying will make more sense then reusing a single allocation.
	const char *getTransformedIndex(int dimensionNo, const char *origIndexName, bool copyMode);
        const char *getOriginalIndex(int dimensionNo, const char *xformIndexName, bool copyMode);
        const char *getInclusionTestExpr(int dimensionNo, const char *origIndexName, bool copyMode);
	const char *getImpreciseLowerXformedIndex(int dimension, const char *index, bool copyMode); 
};

class Strided : public PartitionFunctionConfig {
  public:
	static const char *name;
	Strided(yyltype *location) : PartitionFunctionConfig(location, name) {}
	void processArguments(List<PartitionArg*> *dividingArgs, List<PartitionArg*> *paddingArgs);
	bool doesReorderStoredData() { return true; }

	// The same comment of the above is applicable for these implementations too. 
	const char *getTransformedIndex(int dimensionNo, const char *origIndexName, bool copyMode);
        const char *getOriginalIndex(int dimensionNo, const char *xformIndexName, bool copyMode);
        const char *getInclusionTestExpr(int dimensionNo, const char *origIndexName, bool copyMode);
	const char *getImpreciseLowerXformedIndex(int dimension, const char *index, bool copyMode); 
};

#endif
