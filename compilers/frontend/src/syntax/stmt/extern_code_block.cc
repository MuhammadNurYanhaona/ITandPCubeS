#include "../ast.h"
#include "../ast_stmt.h"
#include "../ast_expr.h"
#include "../ast_def.h"
#include "../ast_task.h"
#include "../../common/errors.h"
#include "../../common/location.h"
#include "../../semantics/scope.h"
#include "../../semantics/symbol.h"
#include "../../../../common-libs/utils/list.h"

#include <iostream>
#include <sstream>
#include <cstdlib>

//-------------------------------------------------------- External Code Block -------------------------------------------------------/

ExternCodeBlock::ExternCodeBlock(const char *language,
                        List<const char*> *headerIncludes,
                        List<const char*> *libraryLinks,
                        const char *codeBlock, yyltype loc) : Stmt(loc) {

        Assert(language != NULL && codeBlock != NULL);
        this->language = language;
        this->headerIncludes = headerIncludes;
        this->libraryLinks = libraryLinks;
        this->codeBlock = codeBlock;
}

void ExternCodeBlock::PrintChildren(int indentLevel) {
        std::ostringstream indent;
        for (int i = 0; i < indentLevel; i++) {
                indent << '\t';
        }
        std::cout << indent.str() << "Language: " << language << "\n";
        if (headerIncludes != NULL) {
                std::cout << indent.str() << "Included Headers:\n";
                for (int i = 0; i < headerIncludes->NumElements(); i++) {
                        std::cout << indent.str() << '\t' << headerIncludes->Nth(i) << "\n";
                }
        }
        if (libraryLinks != NULL) {
                std::cout << indent.str() << "Linked Libraries:\n";
                for (int i = 0; i < libraryLinks->NumElements(); i++) {
                        std::cout << indent.str() << '\t' << libraryLinks->Nth(i) << "\n";
                }
        }
        std::cout << indent.str() << "Code Block:" << codeBlock << "\n";
}

Node *ExternCodeBlock::clone() {
	const char *newLng = strdup(language);
	List<const char*> *newIncls = NULL;
	if (headerIncludes != NULL) {
		newIncls = new List<const char*>;
		for (int i = 0; i < headerIncludes->NumElements(); i++) {
			newIncls->Append(strdup(headerIncludes->Nth(i)));
		}
	}
	List<const char*> *newLibs = NULL;
	if (libraryLinks != NULL) {	
		newLibs = new List<const char*>;
		for (int i = 0; i < libraryLinks->NumElements(); i++) {
			newLibs->Append(strdup(libraryLinks->Nth(i)));
		}
	}
	const char *newCode = strdup(codeBlock);
	return new ExternCodeBlock(newLng, newIncls, newLibs, newCode, *GetLocation());
}
