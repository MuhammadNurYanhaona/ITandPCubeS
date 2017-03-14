#ifndef _H_ast_partition
#define _H_ast_partition

#include "ast.h"
#include "ast_expr.h"
#include "../common/constant.h"
#include "../../../common-libs/utils/list.h"

class TaskDef;
class Scope;
class IntConstant;
class ArrayDataStructure;
class PartitionHierarchy;
class PartitionFunctionConfig;

class SpaceLinkage : public Node {
  protected:
	PartitionLinkType linkType;
	char parentSpace;
  public:
	SpaceLinkage(PartitionLinkType linkType, char parentSpace, yyltype loc);	
	const char *GetPrintNameForNode() { return "Space-Linkage"; }
        void PrintChildren(int indentLevel);
	
	//------------------------------------------------------------------ Helper functions for Semantic Analysis

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
	const char *GetPrintNameForNode() { return "Partition-Arg"; }
        void PrintChildren(int indentLevel);
	
	//------------------------------------------------------------------ Helper functions for Semantic Analysis

	Node *getContent();
	void validateScope(Scope *partitionScope);
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

	//------------------------------------------------------------------ Helper functions for Semantic Analysis

	bool isInstructedForReplication() { return replicated; }
        PartitionFunctionConfig *generateConfiguration(List<int> *dataDimensions,
                        int dimensionAccessStartIndex, 
			Scope *partitionScope);	
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

	//------------------------------------------------------------------ Helper functions for Semantic Analysis
	
	ArrayDataStructure *addPartitionConfiguration(Space *space,
                	Scope *partitionScope, 
			PartitionHierarchy *partitionHierarchy);
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

	//------------------------------------------------------------------ Helper functions for Semantic Analysis

	void addSpaceConfiguration(TaskDef *taskDef, 
			PartitionHierarchy *currentHierarchy, 
			Space *ownerSpace);	
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

	//------------------------------------------------------------------ Helper functions for Semantic Analysis
	
	void addSpaceConfiguration(TaskDef *taskDef, PartitionHierarchy *currentHierarchy);	
};

class PartitionSection : public Node {
  protected:
	List<Identifier*> *arguments;
	List<PartitionSpec*> *spaceSpecs;

	// the hierarchical description of the Partition Section as partial ordering of LPSes and their 
	// contents
	PartitionHierarchy *partitionHierarchy;
  public:
	PartitionSection(List<Identifier*> *arguments, List<PartitionSpec*> *spaceSpecs, yyltype loc);
	const char *GetPrintNameForNode() { return "Partition-Section"; }
        void PrintChildren(int indentLevel);
	List<Identifier*> *getArguments() { return arguments; }

	//------------------------------------------------------------------ Helper functions for Semantic Analysis
	
	int getArgumentsCount() { return arguments->NumElements(); }

	// This interpretes the instructions of the Partition Section, validates those instructions, and 
        // constructs a hierarchical description of LPSes and their data content.
	void constructPartitionHierarchy(TaskDef *taskDef);	
};

#endif
