#include "../../../../../../frontend/src/syntax/ast_expr.h"
#include "../../../../../../frontend/src/semantics/task_space.h"
#include "../../../utils/code_constant.h"

#include <sstream>
#include <iostream>
#include <cstdlib>

void IndexRange::translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space) {
	
	if (begin == NULL && end == NULL) {
		stream << "Range()";
	} else if (begin == NULL) {
		stream << "Range(";
		end->translate(stream, 0, 0, space);
		stream << ")";
	} else if (end == NULL) {
		stream << "Range(";
		begin->translate(stream, 0, 0, space);
		stream << ")";
	} else {
		stream << "Range(";
		begin->translate(stream, 0, 0, space);
		stream << paramSeparator;
		end->translate(stream, 0, 0, space);
		stream << ")";
	}
}

