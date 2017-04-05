#include "../../../../../../frontend/src/syntax/ast_expr.h"
#include "../../../../../../frontend/src/semantics/task_space.h"
#include "../../../utils/code_constant.h"

#include <sstream>
#include <iostream>
#include <cstdlib>

void Expr::generateCode(std::ostringstream &stream, int indentLevel, Space *space) {
        for (int i = 0; i < indentLevel; i++) stream << indent;
        translate(stream, indentLevel, 0, space);
        stream << stmtSeparator;
}

void Expr::translate(std::ostringstream &stream, 
		int indentLevel,
		int currentLineLength, 
		Space *space) {
        std::cout << "A sub-class of expression didn't implement the code generation method\n";
        std::exit(EXIT_FAILURE);
}
