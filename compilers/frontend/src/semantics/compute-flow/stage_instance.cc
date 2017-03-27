#include "../computation_flow.h"
#include "../scope.h"
#include "../symbol.h"
#include "../task_space.h"
#include "../data_access.h"
#include "../../common/errors.h"
#include "../../common/location.h"
#include "../../syntax/ast.h"
#include "../../syntax/ast_expr.h"
#include "../../syntax/ast_stmt.h"
#include "../../syntax/ast_task.h"
#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/utils/hashtable.h"

#include <iostream>
#include <fstream>

//------------------------------------------------------ Stage Instanciation ----------------------------------------------------/

StageInstanciation::StageInstanciation(Space *space) : FlowStage(space) {
	this->code = NULL;
}

void StageInstanciation::performDataAccessChecking(Scope *taskScope) {
	accessMap = validateDataAccess(taskScope, NULL, code);
}

