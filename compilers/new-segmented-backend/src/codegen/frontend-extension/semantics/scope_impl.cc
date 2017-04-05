#include "../../utils/code_constant.h"
#include "../../../../../frontend/src/semantics/scope.h"
#include "../../../../../frontend/src/semantics/symbol.h"
#include "../../../../../frontend/src/syntax/ast_type.h"

#include <sstream>

void Scope::declareVariables(std::ostringstream &stream, int indentLevel) {

        std::ostringstream stmtIndent;
        for (int i = 0; i < indentLevel; i++) stmtIndent << indent;

        Iterator<Symbol*> iterator = this->get_local_symbols();
        Symbol *symbol;
        while ((symbol = iterator.GetNextValue()) != NULL) {

                VariableSymbol *variable = dynamic_cast<VariableSymbol*>(symbol);
                if (variable == NULL) continue;
                Type *type = variable->getType();
                const char *name = variable->getName();
                stream << stmtIndent.str() << type->getCppDeclaration(name) << stmtSeparator;

                // if the variable is a dynamic array then we need to a declare metadata variable
                // alongside its own declaration
                ArrayType *array = dynamic_cast<ArrayType*>(type);
                StaticArrayType *staticArray = dynamic_cast<StaticArrayType*>(type);
                if (array != NULL && staticArray == NULL) {
                        int dimensions = array->getDimensions();
                        stream << stmtIndent.str() << "PartDimension " << name << "Dims";
                        stream << "[" << dimensions << "]" << stmtSeparator;
                        stream << stmtIndent.str() << "ArrayTransferConfig " << name;
                        stream << "TransferConfig = ArrayTransferConfig()" << stmtSeparator;
                }
        }
}
