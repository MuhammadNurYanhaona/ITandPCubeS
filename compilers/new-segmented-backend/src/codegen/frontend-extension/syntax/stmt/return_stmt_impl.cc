#include "../../../utils/code_constant.h"
#include "../../../../../../common-libs/utils/list.h"
#include "../../../../../../frontend/src/syntax/ast_stmt.h"
#include "../../../../../../frontend/src/syntax/ast_expr.h"
#include "../../../../../../frontend/src/semantics/task_space.h"
#include <sstream>

void ReturnStmt::generateCode(std::ostringstream &stream, int indentLevel, Space *space) {
	for (int i = 0; i < indentLevel; i++) stream << indent;
	stream << "return ";
	expr->translate(stream, 0, 0, space);
	stream << stmtSeparator;
}
