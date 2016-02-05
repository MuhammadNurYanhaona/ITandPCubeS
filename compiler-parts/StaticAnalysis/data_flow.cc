#include "data_access.h"
#include "ast.h"
#include "ast_stmt.h"
#include "ast_task.h"
#include "list.h"
#include "hashtable.h"
#include "data_flow.h"
#include "task_space.h"

//--------------------------------------------- Data Flow Stage ----------------------------------------------------/

DataFlowStage::DataFlowStage(yyltype loc) : Node(loc) { 
	nestingIndex = 0; 
	computeIndex = 0;
	nestingController = NULL;	 
}

DataDependencies *DataFlowStage::getDataDependencies() { return dataDependencies; }
