#ifndef _H_reduction_info
#define _H_reduction_info

#include "../common/location.h"
#include "../common/constant.h"
#include "../semantics/task_space.h"
#include "../semantics/computation_flow.h"

/*      If a Execution-Stage has a reduction operation then some setup and tear down of auxiliary variables need to be
        done by the surrounding Compound-Stage before and after of the former stage. Furthermore, any synchronization
        and/or communication also needs to be applied by the Compound-Stage. This class holds information regarding a
        reduction that are extracted by analyzing the Execution-Stages so that the aforementioned actions can be taken
        properly.  
*/
class ReductionMetadata {
  protected:
        const char *resultVar;
        ReductionOperator opCode;
        Space *reductionRootLps;
        Space *reductionExecutorLps;

        // this attribute is only needed for error reporting on invalid reduction operations
        yyltype *location;

	// this is another attribute needed for reduction validation and error reporting
	StageInstanciation *executorStage;
  public:
        ReductionMetadata(const char *resultVar,
                        ReductionOperator opCode,
                        Space *reductionRootLps,
                        Space *reductionExecutorLps, yyltype *location) {
		this->resultVar = resultVar;
		this->opCode = opCode;
		this->reductionRootLps = reductionRootLps;
		this->reductionExecutorLps = reductionExecutorLps;
		this->location = location;
	}
        const char *getResultVar() { return resultVar; }
        ReductionOperator getOpCode() { return opCode; }
        Space *getReductionRootLps() { return reductionRootLps; }
        Space *getReductionExecutorLps() { return reductionExecutorLps; }
        yyltype *getLocation() { return location; }
	void setExecutorStage(StageInstanciation *stage) { this->executorStage = stage; }
	StageInstanciation *getExecutorStage() { return executorStage; }

        // A reduction is singleton when there is just a single global result instance of the reduction operation. 
        // Result handling for such a reduction is much easier than that of a normal reduction. In the former case we
        // can maintain a single task-global scalar property; while in the latter case, results instances need to be
        // dynamically created and maintained for individual LPUs.
        bool isSingleton() {
		if (reductionRootLps->getDimensionCount() > 0) return false;
		Space *parentLps = reductionRootLps->getParent();
		return (parentLps == NULL || parentLps->isRoot());
	}
};

#endif
