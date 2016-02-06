#ifndef _H_ast_partition
#define _H_ast_partition

#include "ast.h"
#include "ast_expr.h"
#include "../utils/list.h"
#include "../semantics/task_space.h"

class TaskDef;
class IntConstant;

class SpaceLinkage : public Node {
  protected:
	PartitionLinkType linkType;
	char parentSpace;
  public:
	SpaceLinkage(PartitionLinkType linkType, char parentSpace, yyltype loc);	
	const char *GetPrintNameForNode() { return "SpaceLinkage"; }
        void PrintChildren(int indentLevel);
	char getParentId() { return parentSpace; }
	bool linkedToSubpartition() { return linkType == LinkTypeSubpartition; }
	Space *getParentSpace(PartitionHierarchy *partitionHierarchy);		
};

class PartitionArg : public Node {
  protected:
	bool constant;
	IntConstant *value;
	Identifier *id;
  public:
	PartitionArg(Identifier *id);
	PartitionArg(IntConstant *value);			
	const char *GetPrintNameForNode() { return "PartitionArg"; }
        void PrintChildren(int indentLevel);
	void validateScope(Scope *partitionScope);
	Node *getContent();		
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
	const char *GetPrintNameForNode() { return "PartitionInstr"; }
        void PrintChildren(int indentLevel);
	void SetOrder(PartitionOrder o) { order = o; }
	bool isInstructedForReplication() { return replicated; }
	PartitionFunctionConfig *generateConfiguration(List<int> *dataDimensions, 
			int dimensionAccessStartIndex, Scope *partitionScope);		
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
	const char *GetPrintNameForNode() { return "DataSpec"; }
	void SetParentLink(SpaceLinkage *p) { parentLink = p; }
        void PrintChildren(int indentLevel);
	ArrayDataStructure *addPartitionConfiguration(Space *space, 
		Scope *partitionScope, PartitionHierarchy *partitionHierarchy);		
};

class SubpartitionSpec : public Node {
  protected:
	int dimensionality;
	bool ordered;
	List<DataConfigurationSpec*> *specList;
  public:
	SubpartitionSpec(int dimensionality, bool ordered, 
		List<DataConfigurationSpec*> *specList, yyltype loc);
	const char *GetPrintNameForNode() { return "Subpartition Configuration"; }
        void PrintChildren(int indentLevel);		
	void addSpaceConfiguration(TaskDef *taskDef, 
		PartitionHierarchy *currentHierarchy, Space *ownerSpace);		
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
	const char *GetPrintNameForNode() { return "Space Configuration"; }
        void PrintChildren(int indentLevel);
	SpaceLinkage *getSpaceLinkage() { return parentLink; }
	void addSpaceConfiguration(TaskDef *taskDef, PartitionHierarchy *currentHierarchy);		
};

class PartitionSection : public Node {
  protected:
	List<Identifier*> *arguments;
	List<PartitionSpec*> *spaceSpecs;
	PartitionHierarchy *partitionHierarchy;
  public:
	PartitionSection(List<Identifier*> *arguments, List<PartitionSpec*> *spaceSpecs, yyltype loc);
	const char *GetPrintNameForNode() { return "Partition Section"; }
        void PrintChildren(int indentLevel);
	int getArgumentsCount() { return arguments->NumElements(); }
	List<Identifier*> *getArguments() { return arguments; }	
	void constructPartitionHierarchy(TaskDef *taskDef);
	PartitionHierarchy *getPartitionHierarchy() { return partitionHierarchy; }	
};

#endif
