#include "../../../utils/name_transformer.h"
#include "../../../../../../frontend/src/syntax/ast_expr.h"
#include "../../../../../../frontend/src/semantics/task_space.h"

#include <sstream>
#include <iostream>
#include <cstdlib>

void IntConstant::translate(std::ostringstream &s, int i, int c, Space *space) { s << value; }

void FloatConstant::translate(std::ostringstream &s, int i, int c, Space *space) { s << value; }

void DoubleConstant::translate(std::ostringstream &s, int i, int c, Space *space) { s << value; }

void BoolConstant::translate(std::ostringstream &s, int i, int c, Space *space) { s << value; }

void CharConstant::translate(std::ostringstream &s, int i, int c, Space *space) { s << value; }

void StringConstant::translate(std::ostringstream &s, int i, int c, Space *space) { s << '\"' << value << '\"'; }

void ReductionVar::translate(std::ostringstream &s, int i, int c, Space *space) {

	ntransform::NameTransformer *transformer = ntransform::NameTransformer::transformer;
	const char *transformedName = transformer->getTransformedName(name, false, false);
	s << transformedName;
}
