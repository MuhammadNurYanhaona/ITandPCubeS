#include "../../../../../../common-libs/utils/list.h"
#include "../../../../../../frontend/src/semantics/computation_flow.h"
#include "../../../../../../frontend/src/static-analysis/sync_stat.h"
#include "../../../../../../frontend/src/static-analysis/data_dependency.h"

void FlowStage::generateInvocationCode(std::ofstream &stream, int indentation, Space *containerSpace) {}

List<const char*> *FlowStage::getAllOutgoingDependencyNamesAtNestingLevel(int nestingLevel) {
        if (this->repeatIndex != nestingLevel) return NULL;
        List<const char*> *arcNameList = new List<const char*>;
        List<SyncRequirement*> *syncList = synchronizationReqs->getAllNonSignaledSyncReqs();
        for (int i = 0; i < syncList->NumElements(); i++) {
                DependencyArc *arc = syncList->Nth(i)->getDependencyArc();
                if (arc->getNestingIndex() == nestingLevel) {
                        arcNameList->Append(arc->getArcName());
                }
        }
        return arcNameList;
}
