#ifndef _H_name_transfer
#define _H_name_transfer

#include "../utils/list.h"
#include <iostream>
#include <string>

class TaskDef;
class Type;

namespace ntransform {

	/* This is an utility class for appending appropriate prefix and suffixes to variable names during
	   code generation. Note that since we place different IT data structures and even sometimes their 
	   different attributes to different target data structures, we cannot apply the structure names 
	   mentioned in the IT source directly in the translated low level C/Fortran output. We have to 
	   append prefix/suffix based on the containing data structure for a specific IT variable. This 
	   scheme works because we use a predefined placement strategy that is uniform across tasks and 
	   their computation stages and the argument names of generated function calls for computation stages
	   also follow a consistent naming strategy.

	   According to this strategy the following rule is used for the host code:
	   
	   1. Global scalars should be prefixed with threadLocals or taskGlobals depending on which category
	      they belong to
	   2. Global array access is always prefixed with an lpu as LPU
	   3. Accessing an LPU specific metadata of an array should have a PartDims suffix. This should work
	      as we will add some lines by default at the beginning of any function representing a part of the
	      code to copy lpu array metadata to local dimension variables of appropriate names. 
	   4. Global array metadata should have arrayMetadata prefix as all metadata is stored in a static
	      variable of such name and their should be a Dims suffix.

	   This class is supposed to be accessed before the beginning of translation of code within compute
	   and initialize sections to generate a static transformation map. Later it should be accessed 
	   wherever necessary to transform variable names by conducting that map.			
	*/


	class NameTransformer {
	  protected:
		List<const char*> *taskGlobals;
		List<const char*> *threadLocals;
		List<const char*> *globalArrays;
		std::string lpuPrefix;
		
		// this flag is used to indicate that the name transformer is working outside of the compute 
		// block. This helps to handle name transformation in initialize functions for a task. TODO 
		// in the future, however, we have to come up with a better mechanism to make the name 
		// transformer flexible as we need to handle other contexts such as the coordinator program 
		//and functions.
		bool localAccessDisabled;
		
	  public:
		NameTransformer();
		static NameTransformer *transformer;
		
		// given a variable name in the source code, construct the name for its representative in the
		// generated code	
		virtual const char *getTransformedName(const char *varName, 
				bool metadata, bool local, Type *type = NULL);
	
		// when an element of an array has been accessed, the 'array[index1][index2]' like expression
		// in the source code needs to be translated as one that always work on one dimensional arrays.
		// This function returns the translated index for a single array dimension. 
		virtual const char *getArrayIndexStorageSuffix(const char *arrayName, 
				int dimensionNo, int dimensionCount, int indentLevel);

		bool isTaskGlobal(const char *varName);
		bool isThreadLocal(const char *varName);
		bool isGlobalArray(const char *varName);
		void setLpuPrefix(const char *prefix) { lpuPrefix = std::string(prefix); }
		std::string getLpuPrefix() { return lpuPrefix; }
		void disableLocalAccess() { localAccessDisabled = true; }
		void enableLocalAccess() { localAccessDisabled = false; }
		void reset();
		void initializePropertyLists(TaskDef *taskDef);
	};

	/* The organization of variables is different in the GPU memory from that of the host memory. Although
	   even in case of CUDA code we maintain a consistent naming strategy, the strategy is more context
	   sensitive than that of the host and fundamentally different from the latter's naming strategy due to
 	   the nature of intra-Kernel LPS expansion. So we need an extansion of the name transformer that can 
	   address GPU environment. Given that when GPU (or hybrid) mapping has been used for a task, some code
	   can run in the host and some in the GPU, we can have shifts in the naming strategy within a single
	   task. This transformer thus has a naming strategy swithing support.

	   For name translation inside a GPU kernel the naming scheme is:

	   1. an array is always refered to by its original name in the source code, no prefix/suffix will be
	      added to it
	   2. scalar variables will be addressed in the same way they are being addressed in the host code as
	      we have the same task-globals and thread-locals pointers
	   3. global array metadata will be accessed with an 'arrayMetadata.' prefix as unlike in the host 
	      functions it is a value reference (not a pointer reference) inside a CUDA kernel
	   4. Array partition dimension related metadata will be prefixed by 'space${Space-Name}' prefix and
	      when applicable gets a '[warpId]' suffix.
	   5. Array partition and storage dimension metadata variables have the following naming scheme
				variable-name 'Space' LPS-Name 'S'|'P' 'Ranges'		  	 	 
	*/
	class HybridNameTransformer : public NameTransformer {
	  protected:
		bool gpuMode;
		bool applyWarpSuffix;
		const char *currentLpsName;
	  public:
		HybridNameTransformer();
		void setToGpuMode() { gpuMode = true; }
		void setToHostMode() { gpuMode = false; }
		bool isGpuMode() { return gpuMode; }
		void setWarpSuffixStat(bool stat) { applyWarpSuffix = stat; }
		void setCurrentLpsName(const char *lpsName) { this->currentLpsName = lpsName; }
		const char *getTransformedName(const char *varName,
                                bool metadata, bool local, Type *type = NULL);
		const char *getArrayIndexStorageSuffix(const char *arrayName, 
				int dimensionNo, int dimensionCount, int indentLevel);

	};
	
	// function to initialize the name transformer for a task
	void setTransformer(TaskDef *taskDef, bool needHybridTransformer);
}

#endif
