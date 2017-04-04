#ifndef _H_task_global
#define _H_task_global

#include "../../../../common-libs/utils/list.h"

#include "../../../../frontend/src/syntax/ast_task.h"
#include "../../../../frontend/src/syntax/ast_type.h"

/* Note that scalar variables that are used as indexes in repeat loops or that are epoch variables
   need not be synchronzed. Such variables can be stored as individual copies within each physical
   processor. Other scalar variables need to be synchronized after each update. This library holds
   the classes needed for distinguishing these two types of scalars.
 */

class TaskGlobalScalar {
  protected:
	const char *name;
	bool locallyManageable;
	Type *type;
  public:
	TaskGlobalScalar(const char *name, bool locallyManageable, Type *type) {
		this->name = name;
		this->locallyManageable = locallyManageable;
		this->type = type;
	}
	const char *getName() { return name; }
	bool isLocallyManageable() { return locallyManageable; }
	Type *getType() { return type; }	
};

class TaskGlobalCalculator {
  public:
	static List<TaskGlobalScalar*> *calculateTaskGlobals(TaskDef *taskDef);	
};

#endif
