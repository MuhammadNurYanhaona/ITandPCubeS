#ifndef _H_computation_flow
#define _H_computation_flow

#include "scope.h"
#include "task_space.h"
#include "../common/location.h"
#include "../syntax/ast.h"
#include "../syntax/ast_expr.h"
#include "../syntax/ast_stmt.h"
#include "../syntax/ast_task.h"
#include "../../../common-libs/utils/list.h"
#include "../../../common-libs/utils/hashtable.h"

#include <iostream>
#include <fstream>

/*	Base class for representing a stage in the execution flow of a task. Instead of directly using the compute and 
	meta-compute stages that we get from the abstract syntax tree, we derive a modified set of flow stages that are 
	easier to reason with for later part of the compiler. 
*/
class FlowStage {
  protected:
	Space *space;
	FlowStage *parent;
	
	// Index indicates the position of a flow stage compared to other stages and group No specifies its container 
        // stage's, if exists, index. Finally, repeat index is the index of the closest repeat control block that 
	// encircles current stage. These positional properties are needed for different kinds of data dependency and
	// access analyses.
        int index;
        int groupNo;
        int repeatIndex;	
  public:
	FlowStage(Space *space);
	virtual ~FlowStage() {};
	void setParent(FlowStage *parent) { this->parent = parent; }
	FlowStage *getParent() { return parent; }
	Space *getSpace() { return space; }
	virtual void print(int indent) = 0;
	
	void setIndex(int index) { this->index = index; }
	int getIndex() { return index; }
	void setGroupNo(int index) { this->groupNo = index; }
	int getGroupNo() { return groupNo; }
	void setRepeatIndex(int index) { this->repeatIndex = index; }
	int getRepeatIndex() { return repeatIndex; }
};

/*	A stage instanciation represents an invocation done from the Computation Section of a compute stage defined 
	in the Stages Section
*/
class StageInstanciation : public FlowStage {
  protected:
	Stmt *code;
	Scope *scope;
  public:
	StageInstanciation(Space *space);
	void setCode(List<Stmt*> *stmtList);
	void setScope(Scope *scope) { this->scope = scope; }
	Scope *getScope() { return scope; }
	void print(int indent) {}
};

/*	A composite stage is a holder of other flow stages and control blocks as a sub-flow. */
class CompositeStage : public FlowStage {
  protected:
	List<FlowStage*> *stageList;
  public:
	CompositeStage(Space *space);
	virtual ~CompositeStage() {}
	void addStageAtBeginning(FlowStage *stage);
	void addStageAtEnd(FlowStage *stage);
	void insertStageAt(int index, FlowStage *stage);
	void removeStageAt(int stageIndex);
	void setStageList(List<FlowStage*> *stageList);
	List<FlowStage*> *getStageList() { return stageList; }
	virtual void print(int indent) {}
};

/*	A repeat control block is a composite stage being iterated over under the control of a repeat instruction.
*/
class RepeatControlBlock : public CompositeStage {
  protected:
	Expr *condition;
	RepeatCycleType type;
  public:
	RepeatControlBlock(Space *space, RepeatCycleType type, Expr *executeCond);
	void print(int indent) {}
};

/*	A conditional execution block represents a composite stage that has the nested sub-flow set to be executed
	only if a particular condition is true. 
*/
class ConditionalExecutionBlock : public CompositeStage {
  protected:
	Expr *condition;
  public:
	ConditionalExecutionBlock(Space *space, Expr *executeCond);
	void print(int indent) {}
};

/*	This represents a transition in the Computation flow of the task from an ancestor LPS to a descendent LPS.
*/
class LpsTransitionBlock : public CompositeStage {
  protected:
	Space *ancestorSpace;
  public:
	LpsTransitionBlock(Space *space, Space *ancestorSpace);		
	void print(int indent) {}
};

/*	This represents a sub-flow boundary at the end of which the versions of all multi-version data structures
	used within the sub-flow must be advanced by one step.
*/
class EpochBoundaryBlock : public CompositeStage {
  public:
	EpochBoundaryBlock(Space *space);	
	void print(int indent) {}
};

#endif
