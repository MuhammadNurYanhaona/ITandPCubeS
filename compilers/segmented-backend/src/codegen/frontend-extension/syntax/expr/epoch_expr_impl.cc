#include "../../../../../../frontend/src/syntax/ast_expr.h"
#include "../../../../../../frontend/src/semantics/task_space.h"
#include "../../../utils/code_constant.h"

#include <sstream>
#include <iostream>
#include <cstdlib>

void EpochExpr::translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space) {
	stream << "(";
        root->translate(stream, indentLevel, currentLineLength, space);
        stream << ")";
}
