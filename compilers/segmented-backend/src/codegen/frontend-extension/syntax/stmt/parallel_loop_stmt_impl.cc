#include "../../../../../../common-libs/utils/list.h"
#include "../../../../../../frontend/src/syntax/ast_stmt.h"
#include "../../../../../../frontend/src/semantics/task_space.h"
#include <sstream>

void PLoopStmt::generateCode(std::ostringstream &stream, int indentLevel, Space *space) {
	List<LogicalExpr*> *restrictions = getIndexRestrictions();
        LoopStmt::generateIndexLoops(stream, indentLevel, space, body, restrictions);
}
