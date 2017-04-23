#include "../../../utils/name_transformer.h"
#include "../../../../../../common-libs/utils/list.h"
#include "../../../../../../frontend/src/syntax/ast.h"
#include "../../../../../../frontend/src/syntax/ast_stmt.h"
#include "../../../../../../frontend/src/syntax/ast_expr.h"
#include "../../../../../../frontend/src/semantics/task_space.h"
#include "../../../../../../frontend/src/semantics/loop_index.h"

#include <sstream>

void SLoopStmt::generateCode(std::ostringstream &stream, int indentLevel, Space *space) {

	if(arrayIndexTraversal) {
		List<LogicalExpr*> *indexRestrictions = NULL;
                if (restriction != NULL) {
                        indexRestrictions = new List<LogicalExpr*>;
                        indexRestrictions->Append((LogicalExpr*) restriction);
                }
                LoopStmt::generateIndexLoops(stream, indentLevel, space, body, indexRestrictions);
	} else {
		IndexScope::currentScope->enterScope(indexScope);

		std::ostringstream indent;
		for (int i = 0; i < indentLevel; i++) indent << '\t';	
		// create a scope for loop
        	stream << indent.str() << "{ // scope entrance for sequential loop\n";
		// declares any variable created in the nested scope of this loop
		declareVariablesInScope(stream, indentLevel);
		// create a range expression representing the sequential loop	
		RangeExpr *range = new RangeExpr(id, rangeExpr, stepExpr, *id->GetLocation());
        	// translate the range expression into a for loop
        	std::ostringstream rangeLoop;
        	range->generateLoopForRangeExpr(rangeLoop, indentLevel, space);
        	stream << rangeLoop.str();
	
		// if there is an additional restriction that says what values within the range should be skipped
                // then apply the restriction here as a continue block inside the loop
                if (restriction != NULL) {
                        stream << indent.str() << '\t';
                        stream << "if (!(";
                        restriction->translate(stream, indentLevel + 1, 0, space);
                        stream << ")) continue;\n";
                }
	
		// translate the body of the for loop
		body->generateCode(stream, indentLevel + 1, space);
	
        	// close the range loop
        	stream << indent.str() << "}\n";
        	// exit the scope created for the loop 
        	stream << indent.str() << "} // scope exit for sequential loop\n";

		IndexScope::currentScope->goBackToOldScope();
	}
}

