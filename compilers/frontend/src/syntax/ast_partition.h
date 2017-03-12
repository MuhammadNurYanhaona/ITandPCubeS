#ifndef _H_ast_partition
#define _H_ast_partition

#include "ast.h"
#include "ast_expr.h"
#include "../../../common-libs/utils/list.h"

class TaskDef;
class IntConstant;

class SpaceLinkage : public Node {
  protected:
	PartitionLinkType linkType;
	char parentSpace;
  public:
	SpaceLinkage(PartitionLinkType linkType, char parentSpace, yyltype loc);	
	const char *GetPrintNameForNode() { return "Space-Linkage"; }
        void PrintChildren(int indentLevel);
};

class PartitionArg : public Node {
  protected:
	bool constant;
	IntConstant *value;
	Identifier *id;
  public:
	PartitionArg(Identifier *id);
	PartitionArg(IntConstant *value);			
	const char *GetPrintNameForNode() { return "Partition-Arg"; }
        void PrintChildren(int indentLevel);
};

class PartitionInstr : public Node {
  protected:
	bool replicated;
	Identifier *partitionFn;
	List<PartitionArg*> *dividingArgs;
	bool padded;
	List<PartitionArg*> *paddingArgs;
	PartitionOrder order;
  public:
	PartitionInstr(yyltype loc);
	PartitionInstr(Identifier *partitionFn, List<PartitionArg*> *dividingArgs, 
		bool padded, List<PartitionArg*> *paddingArgs, yyltype loc);		
	const char *GetPrintNameForNode() { return "Partition-Instr"; }
        void PrintChildren(int indentLevel);
	void SetOrder(PartitionOrder o) { order = o; }
};

class VarDimensions {
  protected:
	Identifier *variable;
	List<IntConstant*> *dimensions;
  public:
	VarDimensions(Identifier *v, List<IntConstant*> *d) { variable = v; dimensions = d; }
	Identifier *GetVar() { return variable; }
	List<IntConstant*> *GetDimensions() { return dimensions; }	
};

class DataConfigurationSpec : public Node {
  protected:
	Identifier *variable;
	List<IntConstant*> *dimensions;
	List<PartitionInstr*> *instructions;
	SpaceLinkage *parentLink;
  public:
	DataConfigurationSpec(Identifier *variable, List<IntConstant*> *dimensions, 
		List<PartitionInstr*> *instructions, SpaceLinkage *parentLink);
	static List<DataConfigurationSpec*> *decomposeDataConfig(List<VarDimensions*> *varList, 
			List<PartitionInstr*> *instrList, SpaceLinkage *parentLink);	
	const char *GetPrintNameForNode() { return "Data-Item-Spec"; }
	void SetParentLink(SpaceLinkage *p) { parentLink = p; }
        void PrintChildren(int indentLevel);
};

class SubpartitionSpec : public Node {
  protected:
	int dimensionality;
	bool ordered;
	List<DataConfigurationSpec*> *specList;
  public:
	SubpartitionSpec(int dimensionality, bool ordered, 
		List<DataConfigurationSpec*> *specList, yyltype loc);
	const char *GetPrintNameForNode() { return "Subpartition-Configuration"; }
        void PrintChildren(int indentLevel);		
};

class PartitionSpec : public Node {
  protected:
	char spaceId;
	int dimensionality;
	List<DataConfigurationSpec*> *specList;
	bool dynamic;
	SpaceLinkage *parentLink;
	SubpartitionSpec *subpartition;
	List<Identifier*> *variableList;
  public:
	PartitionSpec(char spaceId, int dimensionality, List<DataConfigurationSpec*> *specList,
		bool dynamic, SpaceLinkage *parentLink, SubpartitionSpec *subpartition, yyltype loc);
	PartitionSpec(char spaceId, List<Identifier*> *variableList, yyltype loc);
	const char *GetPrintNameForNode() { return "Space-Configuration"; }
        void PrintChildren(int indentLevel);
	SpaceLinkage *getSpaceLinkage() { return parentLink; }
};

class PartitionSection : public Node {
  protected:
	List<Identifier*> *arguments;
	List<PartitionSpec*> *spaceSpecs;
  public:
	PartitionSection(List<Identifier*> *arguments, List<PartitionSpec*> *spaceSpecs, yyltype loc);
	const char *GetPrintNameForNode() { return "Partition-Section"; }
        void PrintChildren(int indentLevel);
	List<Identifier*> *getArguments() { return arguments; }

	//------------------------------------------------------------------ Helper functions for Semantic Analysis
	
	int getArgumentsCount() { return arguments->NumElements(); }	
};

#endif
