#ifndef _H_semantic_helper
#define _H_semantic_helper

/* This header file contains the definitions of all auxiliary classes that are needed at various
   sub-phases of semantic analysis but aren't required after the underlying sub-phases are done
*/

#include "scope.h"
#include "task_space.h"
#include "../lex/scanner.h"

namespace semantic_helper {

	class ArrayDimConfig {
	  private:
		const char *name;
		int dimensions;
	  public:
		ArrayDimConfig(const char *name, int dimensions) {
			this->name = name;
			this->dimensions = dimensions;
		}
		const char *getName() { return name; }
		int getDimensions() { return dimensions; }	
	};

	// this helper class is needed to create flow-stages that are the main ingradients of the
        // intermediate represention from the elements of the abstract syntax tree
        class FlowStageConstrInfo {
          private:
                int lastStageIndex;
                int currGroupIndex;
                int currRepeatBlockIndex;
                Space *currSpace;
		PartitionHierarchy *lpsHierarchy;
                Scope *scope;
          public:
                FlowStageConstrInfo(Space *rootLps, 
				Scope *taskScope, PartitionHierarchy *partitionHierarchy) {
                        lastStageIndex = 0;
                        currGroupIndex = 0;
                        currRepeatBlockIndex = 0;
                        currSpace = rootLps;
                        scope = taskScope;
			lpsHierarchy = partitionHierarchy;
                }
                void advanceLastStageIndex() { lastStageIndex++; }
                int getLastStageIndex() { return lastStageIndex; }
                void setGroupIndex(int index) { currGroupIndex = index; }
                int getCurrGroupIndex() { return currGroupIndex; }
                void setRepeatBlockIndex(int index) { currRepeatBlockIndex = index; }
                int getCurrRepeatBlockIndex() { return currRepeatBlockIndex; }
                void enterSpace(Space *space) { currSpace = space; }
                Space *getCurrSpace() { return currSpace; }
		PartitionHierarchy *getLpsHierarchy() { return lpsHierarchy; }
                Scope *getScope() { return scope; }
        };
};

#endif
