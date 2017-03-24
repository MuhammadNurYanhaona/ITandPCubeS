#include "../ast.h"
#include "../ast_type.h"
#include "../ast_def.h"
#include "../ast_stmt.h"
#include "../ast_type.h"
#include "../ast_task.h"
#include "../ast_partition.h"
#include "../../common/errors.h"
#include "../../common/location.h"
#include "../../common/constant.h"
#include "../../semantics/scope.h"
#include "../../semantics/symbol.h"
#include "../../semantics/helper.h"
#include "../../semantics/computation_flow.h"
#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/utils/hashtable.h"
#include "../../../../common-libs/utils/string_utils.h"

#include <sstream>

//-------------------------------------------------------- Define Section ---------------------------------------------------------/

DefineSection::DefineSection(List<VariableDef*> *def, yyltype loc) : Node(loc) {
        Assert(def != NULL);
        define = def;
        for (int i = 0; i < define->NumElements(); i++) {
                define->Nth(i)->SetParent(this);
        }
}

void DefineSection::PrintChildren(int indentLevel) {
        define->PrintAll(indentLevel + 1);
}

