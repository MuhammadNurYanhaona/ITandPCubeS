#include "../ast.h"
#include "../ast_stmt.h"
#include "../ast_expr.h"
#include "../ast_def.h"
#include "../ast_task.h"
#include "../../common/errors.h"
#include "../../common/location.h"
#include "../../semantics/scope.h"
#include "../../semantics/symbol.h"
#include "../../semantics/helper.h"
#include "../../semantics/data_access.h"
#include "../../static-analysis/reduction_info.h"
#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/utils/hashtable.h"

#include <iostream>
#include <sstream>
#include <cstdlib>

//------------------------------------------------------------ Loop Statement ----------------------------------------------------------/

LoopStmt::LoopStmt() : Stmt() {
	this->body = NULL;
	this->scope = NULL;
}

LoopStmt::LoopStmt(Stmt *body, yyltype loc) : Stmt(loc) {
	Assert(body != NULL);
	this->body = body;
	this->body->SetParent(this);
	this->scope = NULL;
}

void LoopStmt::extractReductionInfo(List<ReductionMetadata*> *infoSet,
                PartitionHierarchy *lpsHierarchy, 
		Space *executingLps) {
        body->extractReductionInfo(infoSet, lpsHierarchy, executingLps);
}

