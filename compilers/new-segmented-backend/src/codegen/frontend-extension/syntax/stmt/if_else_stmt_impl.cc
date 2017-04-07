#include "../../../../../../common-libs/utils/list.h"
#include "../../../../../../frontend/src/syntax/ast_stmt.h"
#include "../../../../../../frontend/src/semantics/task_space.h"
#include <sstream>

void IfStmt::generateCode(std::ostringstream &stream, int indentLevel, Space *space) {
        for (int i = 0; i < ifBlocks->NumElements(); i++) {
                ConditionalStmt *stmt = ifBlocks->Nth(i);
                stmt->generateCode(stream, indentLevel, i == 0, space);
        }
        stream << '\n';
}
