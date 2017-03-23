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

//----------------------------------------------------- Environment Section -------------------------------------------------------/

//------------------------------------------------------------------------------------------------------------------Environment Link
EnvironmentLink::EnvironmentLink(Identifier *v, LinkageType m) : Node(*v->GetLocation()) {
        Assert(v != NULL);
        var = v;
        var->SetParent(this);
        mode = m;
}

List<EnvironmentLink*> *EnvironmentLink::decomposeLinks(List<Identifier*> *idList, LinkageType mode) {
        List<EnvironmentLink*> *links = new List<EnvironmentLink*>;
        for (int i = 0; i < idList->NumElements(); i++) {
           links->Append(new EnvironmentLink(idList->Nth(i), mode));
        }
        return links;
}

void EnvironmentLink::PrintChildren(int indentLevel) {
        var->Print(indentLevel + 1);
}

const char *EnvironmentLink::GetPrintNameForNode() {
        return (mode == TypeCreate) ? "Create"
                : (mode == TypeLink) ? "Link" : "Create if Not Linked";
}

//---------------------------------------------------------------------------------------------------------------Environment Config
EnvironmentSection::EnvironmentSection(List<EnvironmentLink*> *l, yyltype loc) : Node(loc) {
        Assert(l != NULL);
        links = l;
        for (int i = 0; i < links->NumElements(); i++) {
                links->Nth(i)->SetParent(this);
        }
}

void EnvironmentSection::PrintChildren(int indentLevel) {
        links->PrintAll(indentLevel + 1);
}

