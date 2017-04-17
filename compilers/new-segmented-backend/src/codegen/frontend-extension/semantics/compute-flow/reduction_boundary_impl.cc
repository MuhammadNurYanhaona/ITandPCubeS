#include "../../../utils/code_constant.h"
#include "../../../utils/name_transformer.h"
#include "../../../../../../frontend/src/syntax/ast_expr.h"
#include "../../../../../../frontend/src/semantics/task_space.h"
#include "../../../../../../frontend/src/semantics/computation_flow.h"
#include "../../../../../../frontend/src/static-analysis/reduction_info.h"

#include <fstream>
#include <sstream>
#include <cstdlib>

void ReductionBoundaryBlock::generateInvocationCode(std::ofstream &stream, int indentation, Space *containerSpace) {
	
	std::ostringstream indents;
        for (int i = 0; i < indentation; i++) indents << indent;

        stream << std::endl << indents.str() << "{ // beginning of a reduction boundary\n";

	// initialize partial reduction results holder 
        stream << std::endl;
        stream << indents.str() << "// initializing thread-local reduction result variables\n";
        for (int i = 0; i < assignedReductions->NumElements(); i++) {

                ReductionMetadata *reduction = assignedReductions->Nth(i);
                const char *resultVar = reduction->getResultVar();
                Space *executingLps = reduction->getReductionExecutorLps();
                const char *execLpsName = executingLps->getName();
                stream << indents.str() << "if(threadState->isValidPpu(Space_" << execLpsName << ")) {\n";
                stream << indents.str() << indent;
                stream << "reduction::Result *" << resultVar << "Local = reductionResultsMap->";
                stream << "Lookup(\"" << resultVar << "\")" << stmtSeparator;
                stream << indents.str() << indent;
                stream << "ReductionPrimitive *rdPrimitive = rdPrimitiveMap->Lookup(\"";
                stream << resultVar << "\")" << stmtSeparator;
                stream << indents.str() << indent << "rdPrimitive->resetPartialResult(";
                stream << resultVar << "Local)" << stmtSeparator;
                stream << indents.str() << "}\n";
        }
        stream << std::endl;

	// invoke the code inside the repeat boundary
	CompositeStage::generateInvocationCode(stream, indentation, containerSpace);

	// execute the final step of all nested reductions
	stream << std::endl;
        stream << indents.str() << "// executing the final step of reductions\n";
        for (int i = 0; i < assignedReductions->NumElements(); i++) {

                ReductionMetadata *reduction = assignedReductions->Nth(i);
                const char *resultVar = reduction->getResultVar();
                Space *executingLps = reduction->getReductionExecutorLps();
                const char *execLpsName = executingLps->getName();
                stream << indents.str() << "if(threadState->isValidPpu(Space_" << execLpsName << ")) {\n";
                stream << indents.str() << indent;
                stream << "reduction::Result *localResult = reductionResultsMap->";
                stream << "Lookup(\"" << resultVar << "\")" << stmtSeparator;
                stream << indents.str() << indent;

		ntransform::NameTransformer *transformer = ntransform::NameTransformer::transformer;
		const char *propertyName = transformer->getTransformedName(resultVar, false, false);
                stream << "void *target = &(" << propertyName << ")" << stmtSeparator;

                stream << indents.str() << indent;
                stream << "ReductionPrimitive *rdPrimitive = rdPrimitiveMap->Lookup(\"";
                stream << resultVar << "\")" << stmtSeparator;
                stream << indents.str() << indent;
                stream << "rdPrimitive->reduce(localResult" << paramSeparator << "target)" << stmtSeparator;
                stream << indents.str() << "}\n";
        }

        stream << std::endl << indents.str() << "} // ending of a reduction boundary\n";
}
