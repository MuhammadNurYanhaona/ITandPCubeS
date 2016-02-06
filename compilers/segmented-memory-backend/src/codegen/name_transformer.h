#ifndef _H_name_transfer
#define _H_name_transfer

#include "../utils/list.h"
#include <iostream>
#include <string>

class TaskDef;
class Type;

/* This is an utility class for appending appropriate prefix and suffixes to variable names during
   code generation. Note that since we place different IT data structures and even sometimes their 
   different attributes to different target data structures, we cannot apply the structure names 
   mentioned in the IT source directly in the translated low level C/Fortran output. We have to 
   append prefix/suffix based on the containing data structure for a specific IT variable. This 
   scheme works because we use a predefined placement strategy that is uniform across tasks and 
   their computation stages and the argument names of generated function calls for computation stages
   also follow a consistent naming strategy.

   According to this strategy the following rule is used:
   
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

namespace ntransform {

	class NameTransformer {
	  private:
		List<const char*> *taskGlobals;
		List<const char*> *threadLocals;
		List<const char*> *globalArrays;
		std::string lpuPrefix;
		// this flag is used to indicate that the name transformer is working outside of the
		// compute block. This helps to handle name transformation in initialize functions for
		// a task. TODO in the future, however, we have to come up with a better mechanism to
		// make the name transformer flexible as we need to handle other contexts such as 
		// the coordinator program and functions.
		bool localAccessDisabled;
		NameTransformer();
	  public:
		static NameTransformer *transformer;
		static void setTransformer(TaskDef *taskDef);
		const char *getTransformedName(const char *varName, 
				bool metadata, bool local, Type *type = NULL);	
		bool isTaskGlobal(const char *varName);
		bool isThreadLocal(const char *varName);
		bool isGlobalArray(const char *varName);
		void setLpuPrefix(const char *prefix) { lpuPrefix = std::string(prefix); }
		std::string getLpuPrefix() { return lpuPrefix; }
		void disableLocalAccess() { localAccessDisabled = true; }
		void enableLocalAccess() { localAccessDisabled = false; }
		void reset();
	};
}

#endif
