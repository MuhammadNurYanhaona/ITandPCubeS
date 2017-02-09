#include "ast.h"
#include "ast_expr.h"
#include "ast_type.h"
#include "ast_library_fn.h"
#include "../common/errors.h"
#include "../../../common-libs/utils/list.h"

#include <iostream>

//-------------------------------------------------------- Static Constants -----------------------------------------------------/

const char *Root::Name = "root";
const char *Random::Name = "random";
const char *LoadArray::Name = "load_array";
const char *LoadListOfArrays::Name = "load_list_of_arrays";
const char *StoreArray::Name = "store_array";
const char *StoreListOfArrays::Name = "store_list_of_arrays";
const char *BindInput::Name = "bind_input";
const char *BindOutput::Name = "bind_output";

//-------------------------------------------------------- Library Function -----------------------------------------------------/

LibraryFunction::LibraryFunction(int argumentCount, Identifier *functionName,
                List<Expr*> *arguments, yyltype loc) : Expr(loc) {
        this->argumentCount = argumentCount;
        this->functionName = functionName;
        this->arguments = arguments;
}

LibraryFunction::LibraryFunction(int argumentCount, Identifier *functionName, List<Expr*> *arguments) : Expr() {
        this->argumentCount = argumentCount;
        this->functionName = functionName;
        this->arguments = arguments;
}

bool LibraryFunction::isLibraryFunction(Identifier *id) {
        const char* name = id->getName();
        return (strcmp(name, Root::Name) == 0 || strcmp(name, Random::Name) == 0
                || strcmp(name, LoadArray::Name) == 0 || strcmp(name, LoadListOfArrays::Name) == 0
                || strcmp(name, StoreArray::Name) == 0
                || strcmp(name, StoreListOfArrays::Name) == 0
                || strcmp(name, BindInput::Name) == 0
                || strcmp(name, BindOutput::Name) == 0);
}

LibraryFunction *LibraryFunction::getFunctionExpr(Identifier *id, List<Expr*> *arguments, yyltype loc) {

        const char* name = id->getName();
        LibraryFunction *function = NULL;

        // note that there should never be a default 'else' block here; then the system will fail to find user defined functions
        if (strcmp(name, Root::Name) == 0) {
                function = new Root(id, arguments, loc);
        } else if (strcmp(name, Random::Name) == 0) {
                function = new Random(id, arguments, loc);
        } else if (strcmp(name, LoadArray::Name) == 0) {
                function = new LoadArray(id, arguments, loc);
        } else if (strcmp(name, LoadListOfArrays::Name) == 0) {
                function = new LoadListOfArrays(id, arguments, loc);
        } else if (strcmp(name, StoreArray::Name) == 0) {
                function = new StoreArray(id, arguments, loc);
        } else if (strcmp(name, StoreListOfArrays::Name) == 0) {
                function = new StoreListOfArrays(id, arguments, loc);
        } else if (strcmp(name, BindInput::Name) == 0) {
                function = new BindInput(id, arguments, loc);
        } else if (strcmp(name, BindOutput::Name) == 0) {
                function = new BindOutput(id, arguments, loc);
        }

        return function;
}

void LibraryFunction::PrintChildren(int indentLevel) {
        PrintLabel(indentLevel + 1, "Arguments");
        arguments->PrintAll(indentLevel + 2);
}
