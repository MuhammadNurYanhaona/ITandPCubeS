#include "../syntax/ast_def.h"
#include "../semantics/scope.h"
#include <sstream>
#include <fstream>

void CoordinatorDef::declareVariablesInScope(std::ostringstream &stream, int indent) {}
void CoordinatorDef::generateCode(std::ostringstream &stream, Scope *scope) {}
void FunctionInstance::generateCode(std::ofstream &headerFile, std::ofstream &programFile) {}
