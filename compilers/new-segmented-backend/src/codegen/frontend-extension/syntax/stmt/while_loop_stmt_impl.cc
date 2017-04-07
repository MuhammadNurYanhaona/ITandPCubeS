#include "../../../utils/code_constant.h"
#include "../../../../../../common-libs/utils/list.h"
#include "../../../../../../frontend/src/syntax/ast_stmt.h"
#include "../../../../../../frontend/src/syntax/ast_expr.h"
#include "../../../../../../frontend/src/semantics/task_space.h"
#include <sstream>

void WhileStmt::generateCode(std::ostringstream &stream, int indentLevel, Space *space) {
        for (int i = 0; i < indentLevel; i++) stream << indent;
        stream << "do {\n";
        body->generateCode(stream, indentLevel + 1, space);
        for (int i = 0; i < indentLevel; i++) stream << indent;
        stream << "} while(";
        if (condition != NULL) {
                condition->translate(stream, indentLevel, 0, space);
        } else {
                stream << "true";
        }
        stream << ")" << stmtSeparator;
}
