#include "../../../../../../common-libs/utils/list.h"
#include "../../../../../../frontend/src/syntax/ast_stmt.h"
#include "../../../../../../frontend/src/semantics/task_space.h"
#include <sstream>

void StmtBlock::generateCode(std::ostringstream &stream, int indentLevel, Space *space) {
	for (int i = 0; i < stmts->NumElements(); i++) {
                Stmt *stmt = stmts->Nth(i);
                stmt->generateCode(stream, indentLevel, space);
        }
}
