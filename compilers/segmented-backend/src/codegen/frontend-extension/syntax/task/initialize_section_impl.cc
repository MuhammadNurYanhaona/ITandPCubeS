#include "../../../utils/code_constant.h"
#include "../../../../../../frontend/src/syntax/ast_type.h"
#include "../../../../../../frontend/src/syntax/ast_stmt.h"
#include "../../../../../../frontend/src/syntax/ast_task.h"
#include "../../../../../../frontend/src/semantics/scope.h"
#include "../../../../../../frontend/src/semantics/symbol.h"
#include "../../../../../../frontend/src/semantics/task_space.h"

#include <sstream>

void InitializeSection::generateCode(std::ostringstream &stream) {

        // declare all local variables found in the scope
        Iterator<Symbol*> iterator = scope->get_local_symbols();
        Symbol *symbol;
        while ((symbol = iterator.GetNextValue()) != NULL) {
                VariableSymbol *variable = dynamic_cast<VariableSymbol*>(symbol);
                if (variable == NULL) continue;
                Type *type = variable->getType();
                const char *name = variable->getName();
                stream << indent << type->getCppDeclaration(name) << stmtSeparator;
        }

        TaskDef *taskDef = (TaskDef*) this->parent;
        Space *space = taskDef->getPartitionHierarchy()->getRootSpace();

        // translate statements into C++ code
        code->generateCode(stream, 1, space);
}

