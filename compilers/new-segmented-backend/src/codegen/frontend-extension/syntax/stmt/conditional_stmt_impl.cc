#include "../../../utils/code_constant.h"
#include "../../../../../../common-libs/utils/list.h"
#include "../../../../../../frontend/src/syntax/ast_stmt.h"
#include "../../../../../../frontend/src/syntax/ast_expr.h"
#include "../../../../../../frontend/src/semantics/task_space.h"
#include <sstream>

void ConditionalStmt::generateCode(std::ostringstream &stream, int indentLevel, bool first, Space *space) {
        if (first) {
                for (int i = 0; i < indentLevel; i++) stream << indent;
                stream << "if (";
                if (condition != NULL) {
                        condition->translate(stream, indentLevel, 0, space);
                } else {
                        stream << "true";
                }
                stream << ") {\n";
                stmt->generateCode(stream, indentLevel + 1, space);
                for (int i = 0; i < indentLevel; i++) stream << indent;
                stream << "}";
        } else {
                if (condition != NULL) {
                        stream << " else if (";
                        condition->translate(stream, indentLevel, 0, space);
                        stream << ") {\n";
                } else {
                        stream << " else {\n";
                }
                stmt->generateCode(stream, indentLevel + 1, space);
                for (int i = 0; i < indentLevel; i++) stream << indent;
                stream << "}";
        }
}
