#include "ast.h"
#include "ast_partition.h"
#include "../common/errors.h"
#include "../../../common-libs/utils/list.h"

//----------------------------------------- Linking Spaces Together ---------------------------------------------/

SpaceLinkage::SpaceLinkage(PartitionLinkType l, char p, yyltype loc) : Node(loc) {
        linkType = l;
        parentSpace = p;
}

void SpaceLinkage::PrintChildren(int indentLevel) {
        PrintLabel(indentLevel + 1, "Type");
        printf((linkType == LinkTypePartition) ? "Partition" : "Sup-partition");
        PrintLabel(indentLevel + 1, "Parent");
        printf("%c", parentSpace);
}
        
//-------------------------------------------- Partition Argument -----------------------------------------------/

PartitionArg::PartitionArg(Identifier *i) : Node(*i->GetLocation()) {
        Assert(i != NULL);
        constant = false;
        id = i;
        id->SetParent(this);
        value = NULL;
}

PartitionArg::PartitionArg(IntConstant *v) : Node(*v->GetLocation()) {
        Assert(v != NULL);
        constant = true;
        id = NULL;
        value = v;
        value->SetParent(this);
}

void PartitionArg::PrintChildren(int indentLevel) {
        if (id != NULL) id->Print(indentLevel + 1);
        if (value != NULL) value->Print(indentLevel + 1);
}

//----------------------------------------- Partition Instruction -----------------------------------------------/

PartitionInstr::PartitionInstr(yyltype loc) : Node(loc) {
        replicated = true;
        partitionFn = NULL;
        dividingArgs = NULL;
        padded = false;
        paddingArgs = NULL;
        order = RandomOrder;
}

PartitionInstr::PartitionInstr(Identifier *pF, List<PartitionArg*> *dA,
                bool p, List<PartitionArg*> *pA, yyltype loc) : Node(loc) {
        Assert(pF != NULL && dA != NULL);
        replicated = false;
        partitionFn = pF;
        partitionFn->SetParent(this);
        dividingArgs = dA;
        for (int i = 0; i < dividingArgs->NumElements(); i++) {
                dividingArgs->Nth(i)->SetParent(this);
        }
        padded = p;
        paddingArgs = pA;
        if (paddingArgs != NULL) {
                for (int i = 0; i < paddingArgs->NumElements(); i++) {
                        paddingArgs->Nth(i)->SetParent(this);
                }
        }
        order = RandomOrder;
}

void PartitionInstr::PrintChildren(int indentLevel) {
        if (replicated) printf("replicated");
        else {
                partitionFn->Print(indentLevel + 1, "(Function) ");
                PrintLabel(indentLevel + 1, "PartitionArgs");
                dividingArgs->PrintAll(indentLevel + 2);
                PrintLabel(indentLevel + 1, "Order");
                if (order == AscendingOrder) printf("Ascending");
                else if (order == DescendingOrder) printf("Descending");
                else printf("Random");
                PrintLabel(indentLevel + 1, "Padding");
                printf((padded) ? "True" : "False");
                if (paddingArgs != NULL) {
                        PrintLabel(indentLevel + 1, "PaddingArgs");
                        paddingArgs->PrintAll(indentLevel + 2);
                }
        }
}

//--------------------------- A Single Data Structure's Partition Specification ---------------------------------/

DataConfigurationSpec::DataConfigurationSpec(Identifier *v, List<IntConstant*> *d,
                List<PartitionInstr*> *i, SpaceLinkage *p) : Node(*v->GetLocation()) {
        Assert(v != NULL);
        variable = v;
        variable->SetParent(this);
        dimensions = d;
        if (dimensions != NULL) {
                for (int j = 0; j < dimensions->NumElements(); j++) {
                        dimensions->Nth(j)->SetParent(this);
                }
        }
        instructions = i;
        if (instructions != NULL) {
                for (int j = 0; j < instructions->NumElements(); j++) {
                        instructions->Nth(j)->SetParent(this);
                }
        }
        parentLink = p;
        if (parentLink != NULL) {
                parentLink->SetParent(this);
        }
}

List<DataConfigurationSpec*> *DataConfigurationSpec::decomposeDataConfig(List<VarDimensions*> *varList,
                List<PartitionInstr*> *instrList, SpaceLinkage *parentLink) {
        List<DataConfigurationSpec*> *specList = new List<DataConfigurationSpec*>;
        for (int i = 0; i < varList->NumElements(); i++) {
                VarDimensions *vD = varList->Nth(i);
                DataConfigurationSpec *spec
                        = new DataConfigurationSpec(vD->GetVar(), vD->GetDimensions(), instrList, parentLink);
                specList->Append(spec);
        }
        return specList;
}

void DataConfigurationSpec::PrintChildren(int indentLevel) {
        variable->Print(indentLevel + 1, "(Array) ");
        if (parentLink != NULL) parentLink->Print(indentLevel + 1);
        if (dimensions != NULL) {
                PrintLabel(indentLevel + 1, "Dimensions");
                dimensions->PrintAll(indentLevel + 2);
        }
        if (instructions != NULL) {
                PrintLabel(indentLevel + 1, "Instructions");
                instructions->PrintAll(indentLevel + 2);
        }
}

//----------------------------------- Subpartitioning Space Configuration ---------------------------------------/

SubpartitionSpec::SubpartitionSpec(int d, bool o, List<DataConfigurationSpec*> *sl, yyltype loc) : Node(loc) {
        Assert(sl != NULL);
        dimensionality = d;
        ordered = o;
        specList = sl;
        for (int i = 0; i < specList->NumElements(); i++) {
                specList->Nth(i)->SetParent(this);
        }
}

void SubpartitionSpec::PrintChildren(int indentLevel) {
        PrintLabel(indentLevel + 1, "Dimensions");
        printf("%d", dimensionality);
        PrintLabel(indentLevel + 1, "Ordered");
        printf(ordered ? "True" : "False");
        specList->PrintAll(indentLevel + 1);
}

//-------------------------------------- A Single Space Configuration ------------------------------------------/

PartitionSpec::PartitionSpec(char si, int d, List<DataConfigurationSpec*> *sl, bool dy,
                SpaceLinkage *pl, SubpartitionSpec *sp, yyltype loc) : Node(loc) {
        Assert(sl != NULL && d > 0);
        spaceId = si;
        dimensionality = d;
        specList = sl;
        for (int i = 0; i < specList->NumElements(); i++) {
                specList->Nth(i)->SetParent(this);
        }
        dynamic = dy;
        parentLink = pl;
        if (parentLink != NULL) {
                parentLink->SetParent(this);
        }
        subpartition = sp;
        if (subpartition != NULL) {
                subpartition->SetParent(this);
        }
        variableList = NULL;
}

PartitionSpec::PartitionSpec(char s, List<Identifier*> *v, yyltype loc) : Node(loc) {
        Assert(v != NULL);
        spaceId = s;
        variableList = v;
        for (int i = 0; i < variableList->NumElements(); i++) {
                variableList->Nth(i)->SetParent(this);
        }
        dimensionality = 0;
        specList = NULL;
        dynamic = false;
        parentLink = NULL;
        subpartition = NULL;
}

void PartitionSpec::PrintChildren(int indentLevel) {
        PrintLabel(indentLevel + 1, "Id");
        printf("%c", spaceId);
        PrintLabel(indentLevel + 1, "Dimensions");
        if (dimensionality > 0) printf("%d", dimensionality);
        else printf("N/A");
        PrintLabel(indentLevel + 1, "Dynamic");
        printf(dynamic ? "True" : "False");
        if (parentLink != NULL) parentLink->Print(indentLevel + 1);
        if (subpartition != NULL) subpartition->Print(indentLevel + 1);
        if (variableList != NULL) variableList->PrintAll(indentLevel + 1);
        if (specList != NULL) specList->PrintAll(indentLevel + 1);
}

//-------------------------------------------- Partition Section -----------------------------------------------/

PartitionSection::PartitionSection(List<Identifier*> *a, List<PartitionSpec*> *s, yyltype loc) : Node(loc) {
        Assert(a != NULL && s != NULL);
        arguments = a;
        for (int i = 0; i < arguments->NumElements(); i++) {
                arguments->Nth(i)->SetParent(this);
        }
        spaceSpecs = s;
        for (int i = 0; i < spaceSpecs->NumElements(); i++) {
                spaceSpecs->Nth(i)->SetParent(this);
        }
}

void PartitionSection::PrintChildren(int indentLevel) {
        PrintLabel(indentLevel + 1, "Arguments");
        arguments->PrintAll(indentLevel + 2);
        spaceSpecs->PrintAll(indentLevel + 1);
}
