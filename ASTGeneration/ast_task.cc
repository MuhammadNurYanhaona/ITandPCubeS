#include "ast.h"
#include "ast_type.h"
#include "ast_def.h"
#include "ast_stmt.h"
#include "ast_task.h"
#include "ast_partition.h"
#include "list.h"

TaskDef::TaskDef(Identifier *i, DefineSection *d, EnvironmentConfig *e, 
                InitializeInstr *init, ComputeSection *c, PartitionSection *p): Node(*i->GetLocation()) {
        Assert(i != NULL && d != NULL && e != NULL && c != NULL && p != NULL);
        id = i;
	define = d;
	environment = e;
	initialize = init;
	compute = c;
	partition = p;
}

void TaskDef::PrintChildren(int indentLevel) {
        id->Print(indentLevel + 1);
	define->Print(indentLevel + 1);
	environment->Print(indentLevel + 1);
	if (initialize != NULL) initialize->Print(indentLevel + 1);
	compute->Print(indentLevel + 1);
	partition->Print(indentLevel + 1);
}

DefineSection::DefineSection(List<VariableDef*> *def, yyltype loc) : Node(loc) {
	Assert(def != NULL);
	define = def;
}

void DefineSection::PrintChildren(int indentLevel) {
	define->PrintAll(indentLevel + 1);
}

InitializeInstr::InitializeInstr(List<Identifier*> *a, List<Stmt*> *c, yyltype loc) : Node(loc) {
	Assert(a != NULL && c != NULL);
	arguments = a;
	code = c;
}

void InitializeInstr::PrintChildren(int indentLevel) {
	if (arguments->NumElements() > 0) {
		PrintLabel(indentLevel + 1, "Arguments");
		arguments->PrintAll(indentLevel + 2);
	}
	PrintLabel(indentLevel + 1, "Code");
	code->PrintAll(indentLevel + 2);
}

EnvironmentLink::EnvironmentLink(Identifier *v, LinkageType m) : Node(*v->GetLocation()) {
	Assert(v != NULL);
	var = v;
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

EnvironmentConfig::EnvironmentConfig(List<EnvironmentLink*> *l, yyltype loc) : Node(loc) {
	Assert(l != NULL);
	links = l;
}

void EnvironmentConfig::PrintChildren(int indentLevel) {
        links->PrintAll(indentLevel + 1);
}

StageHeader::StageHeader(Identifier *s, char si, Expr *a) : Node(*s->GetLocation()) {
	Assert(s!= NULL);
	stageId = s;
	spaceId = si;
	activationCommand = a;
}

void StageHeader::PrintChildren(int indentLevel) {
	stageId->Print(indentLevel + 1);
	PrintLabel(indentLevel + 1, "Space");
	printf("%c", spaceId);
	if (activationCommand != NULL) { 
		PrintLabel(indentLevel + 1, "Activation Command");
		activationCommand->Print(indentLevel + 2); 
	}
}

ComputeStage::ComputeStage(StageHeader *h, List<Stmt*> *c) : Node(*h->GetLocation()) {
	Assert(h != NULL && c != NULL);
	header = h;
	code = c;
	metaStage = false;
	nestedSequence = NULL;
}

ComputeStage::ComputeStage(StageHeader *h, List<MetaComputeStage*> *mcs) : Node(*h->GetLocation()) {
	Assert(h != NULL && mcs != NULL);
	header = h;
	code = NULL;
	metaStage = true;
	nestedSequence = mcs;
}

void ComputeStage::PrintChildren(int indentLevel) {
	header->Print(indentLevel + 1);
	if (metaStage) {
		PrintLabel(indentLevel + 1, "Nested Stages");
		nestedSequence->PrintAll(indentLevel + 2);
	} else {
		PrintLabel(indentLevel + 1, "Code");
		code->PrintAll(indentLevel + 2);
	}
}

RepeatControl::RepeatControl(Identifier *b, Expr *r, yyltype loc) : Node(loc) {
	Assert(b != NULL && r != NULL);
	begin = b;
	rangeExpr = r;
}

void RepeatControl::PrintChildren(int indentLevel) {
	begin->Print(indentLevel + 1, "(GoTo) ");
	rangeExpr->Print(indentLevel + 1, "(If) ");
}

MetaComputeStage::MetaComputeStage(List<ComputeStage*> *s, RepeatControl *r) : Node() {
	Assert(s != NULL);
	stageSequence = s;
	repeatInstr = r;
}

void MetaComputeStage::PrintChildren(int indentLevel) {
	stageSequence->PrintAll(indentLevel + 1);
	if (repeatInstr != NULL) repeatInstr->Print(indentLevel + 1);
}

ComputeSection::ComputeSection(List<MetaComputeStage*> *s, yyltype loc) : Node(loc) {
	Assert(s != NULL);
	stageSeqList = s;
}

void ComputeSection::PrintChildren(int indentLevel) {
	stageSeqList->PrintAll(indentLevel + 1);
}


