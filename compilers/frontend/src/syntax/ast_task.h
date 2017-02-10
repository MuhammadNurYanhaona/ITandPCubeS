#ifndef _H_ast_task
#define _H_ast_task

#include "ast.h"
#include "ast_type.h"
#include "ast_stmt.h"
#include "ast_def.h"
#include "ast_expr.h"
#include "../../../common-libs/utils/list.h"
#include "../../../common-libs/utils/hashtable.h"

#include <sstream>

class PartitionSection;

enum LinkageType { TypeCreate, TypeLink, TypeCreateIfNotLinked };

class DefineSection : public Node {
  protected:
	List<VariableDef*> *define;
  public:
	DefineSection(List<VariableDef*> *def, yyltype loc);
        const char *GetPrintNameForNode() { return "Define-Section"; }
        void PrintChildren(int indentLevel);
	List<VariableDef*> *getDefinitions() { return define; }
};

class InitializeSection : public Node {
  protected:
	List<Identifier*> *arguments;
	List<Type*> *argumentTypes;
	List<Stmt*> *code;
  public: 
	InitializeSection(List<Identifier*> *arguments, List<Stmt*> *code, yyltype loc);	
        const char *GetPrintNameForNode() { return "Initialize-Section"; }
        void PrintChildren(int indentLevel);
};

class EnvironmentLink : public Node {
  protected:
	Identifier *var;
	LinkageType mode;
  public:
	EnvironmentLink(Identifier *var, LinkageType mode);
	static List<EnvironmentLink*> *decomposeLinks(List<Identifier*> *ids, LinkageType mode);	
        const char *GetPrintNameForNode();
        void PrintChildren(int indentLevel);
	Identifier *getVariable() { return var; }
	LinkageType getMode() { return mode; }
};

class EnvironmentSection : public Node {
  protected:
	List<EnvironmentLink*> *links;
  public:
	EnvironmentSection(List<EnvironmentLink*> *links, yyltype loc);			
	const char *GetPrintNameForNode() { return "Environment-Section"; }
        void PrintChildren(int indentLevel);
	List<EnvironmentLink*> *getLinks() { return links; }
};

class StageDefinition : public Node {
  protected:
	Identifier *name;
	List<Identifier*> *parameters;
	Stmt *codeBody;
  public:
	StageDefinition(Identifier *name, List<Identifier*> *parameters, Stmt *codeBody);
	const char *GetPrintNameForNode() { return "Stage-Definition"; }
        void PrintChildren(int indentLevel);
};

class StagesSection : public Node {
  protected:
	List<StageDefinition*> *stages;
  public:
	StagesSection(List<StageDefinition*> *stages, yyltype loc);
	const char *GetPrintNameForNode() { return "Stages-Section"; }
        void PrintChildren(int indentLevel);
};

class FlowPart : public Node {
  protected:
	int index;
  public:
	static int currentFlowIndex;
	FlowPart(yyltype loc);
	static void resetFlowIndexRef();
};

class StageInvocation : public FlowPart {
  protected:
	Identifier *stageName;
	List<Expr*> *arguments;
  public:
	StageInvocation(Identifier *stageName, List<Expr*> *arguments, yyltype loc);
	const char *GetPrintNameForNode() { return "Stage-Invocation"; }
        void PrintChildren(int indentLevel);
};

class CompositeFlowPart : public FlowPart {
  protected:
	List<FlowPart*> *nestedSubflow;
  public:
	CompositeFlowPart(yyltype loc, List<FlowPart*> *nestedSubflow);
        virtual void PrintChildren(int indentLevel);
};

class LpsTransition : public CompositeFlowPart {
  protected:
	char lpsId;
  public:
	LpsTransition(char lpsId, List<FlowPart*> *nestedSubflow, yyltype loc);
	const char *GetPrintNameForNode() { return "LPS-Transition"; }
        void PrintChildren(int indentLevel);
};

class ConditionalFlowBlock : public CompositeFlowPart {
  protected:
	Expr *conditionExpr;
  public:
	ConditionalFlowBlock(Expr *conditionExpr, List<FlowPart*> *nestedSubflow, yyltype loc);	
	const char *GetPrintNameForNode() { return "Conditional-Subflow"; }
        void PrintChildren(int indentLevel);
};

class EpochBlock : public CompositeFlowPart {
  public:
	EpochBlock(List<FlowPart*> *nestedSubflow, yyltype loc);	
	const char *GetPrintNameForNode() { return "Epoch-Boundary"; }
};

class RepeatControl : public Node {
  public:
	RepeatControl(yyltype loc) : Node(loc) {}
};

class WhileRepeat : public RepeatControl {
  protected:
	Expr *condition;
  public:
	WhileRepeat(Expr *condition, yyltype loc);	
	const char *GetPrintNameForNode() { return "Condition-Traveral"; }
        void PrintChildren(int indentLevel);
};

class SubpartitionRepeat : public RepeatControl {
  public:
	SubpartitionRepeat(yyltype loc) : RepeatControl(loc) {}
	const char *GetPrintNameForNode() { return "Subpartition-Traversal"; }
        void PrintChildren(int indentLevel) {}
};

class ForRepeat : public RepeatControl {
  protected:
	RangeExpr *rangeExpr;
  public:
	ForRepeat(RangeExpr *rangeExpr, yyltype loc);
	const char *GetPrintNameForNode() { return "Range-Traversal"; }
        void PrintChildren(int indentLevel);
};

class RepeatCycle : public CompositeFlowPart {
  protected:
	RepeatControl *control;
  public:
	RepeatCycle(RepeatControl *control, List<FlowPart*> *nestedSubflow, yyltype loc);
	const char *GetPrintNameForNode() { return "Repeat-Cycle"; }
        void PrintChildren(int indentLevel);
};

class ComputationSection : public Node {
  protected:
	List<FlowPart*> *computeFlow;
  public:
	ComputationSection(List<FlowPart*> *computeFlow, yyltype loc);
	const char *GetPrintNameForNode() { return "Computation-Section"; }
        void PrintChildren(int indentLevel);
};

class TaskDef : public Definition {
  protected:
        Identifier *id;
	DefineSection *define;
	EnvironmentSection *environment;
	InitializeSection *initialize;
	StagesSection *stages;
	ComputationSection *compute;
	PartitionSection *partition;
  public:
        TaskDef(Identifier *id, 
		DefineSection *define, 
		EnvironmentSection *environment, 
		InitializeSection *initialize,
		StagesSection *stages, 
		ComputationSection *compute, 
		PartitionSection *partition);
	const char *GetPrintNameForNode() { return "Task"; }
        void PrintChildren(int indentLevel);
};

#endif
