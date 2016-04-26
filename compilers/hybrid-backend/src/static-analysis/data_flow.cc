#include "data_access.h"
#include "../syntax/ast.h"
#include "../syntax/ast_stmt.h"
#include "../syntax/ast_task.h"
#include "../utils/list.h"
#include "../utils/hashtable.h"
#include "data_flow.h"
#include "../semantics/task_space.h"

//--------------------------------------------- Data Flow Stage ----------------------------------------------------/

DataFlowStage::DataFlowStage(yyltype loc) : Node(loc) { 
	nestingIndex = 0; 
	computeIndex = 0;
	nestingController = NULL;	 
}

DataDependencies *DataFlowStage::getDataDependencies() { return dataDependencies; }
