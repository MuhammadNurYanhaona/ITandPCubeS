#include "../ast.h"
#include "../ast_stmt.h"
#include "../ast_expr.h"
#include "../ast_def.h"
#include "../ast_task.h"
#include "../../common/errors.h"
#include "../../common/location.h"
#include "../../semantics/scope.h"
#include "../../semantics/symbol.h"
#include "../../semantics/data_access.h"
#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/utils/hashtable.h"

#include <iostream>
#include <sstream>
#include <cstdlib>

//-------------------------------------------------------- Statement ------------------------------------------------------------/

void Stmt::mergeAccessedVariables(Hashtable<VariableAccess*> *first,
                        Hashtable<VariableAccess*> *second) {
        if (second == NULL) return;
        Iterator<VariableAccess*> iter = second->GetIterator();
        VariableAccess *accessLog;
        while ((accessLog = iter.GetNextValue()) != NULL) {
                if (first->Lookup(accessLog->getName()) == NULL) {
                        first->Enter(accessLog->getName(), new VariableAccess(accessLog->getName()), true);
                }
                first->Lookup(accessLog->getName())->mergeAccessInfo(accessLog);
        }
}
