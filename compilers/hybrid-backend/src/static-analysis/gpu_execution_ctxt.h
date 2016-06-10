#ifndef _H_gpu_execution_ctxt
#define _H_gpu_execution_ctxt

/* When a task has been mapped to the hybrid model, the parts of its computation flow that are dedicated for GPU 
 * execution need to be treated differently from the rests. Recall that in the course of the task execution the 
 * flow of control may enter and leave the GPU many times and have interleaving computations and communicaitons
 * happening in the hosts and the network level. Furthermore depending on the nature of the LPS partitions mapped
 * to the GPU and what GPU PPS they have been mapped to, the generated GPU kernels, GPU LPU offloading logic, etc.
 * will be different. This library holds the classes that maintain information about how GPU execution should be
 * done for different mapping contexts. 
 */

#include "data_flow.h"
#include "../semantics/task_space.h"
#include "../utils/list.h"

/* The execution logic we have chosen for GPU LPUs is that the host will generate the LPUs in batch and ship them in 
 * and out of the GPUs. Sometimes the batch of LPUs shipped to the GPU may be multiplexed to arbitrary PPUs of the 
 * intended PPS. Some other times, what LPUs executed by what PPU needs to be controlled precisely (for example LPUs 
 * of subpartitioned LPSes have such requirement). Code generation for these two scenarios need to be done differently. 
 * The following enum, specifies the type for a particular GPU execution context.
 */
enum GpuContextType { 	LOCATION_SENSITIVE_LPU_DISTR_CONTEXT, 
		 	LOCATION_INDIPENDENT_LPU_DISTR_CONTEXT };

/* This class represents a particular sub-flow in a task's computation flow that should be executed in the GPU */
class GpuExecutionContext {
  protected:
	// the GPU entrance point LPS for the current context
	Space *contextLps;
	// sequence of top level flow stages -- there might be flow stages nested in them -- that form the current
	// context
	List<FlowStage*> *contextFlow;
	// type of LPU distribution to be used for the current context
	GpuContextType contextType;
  public:
	GpuExecutionContext(int topmostGpuPps, List<FlowStage*> *contextFlow);

	// the context ID, which is the index of the first flow stage within the context, is used for searching the 
	// context during code generation
	int getContextId();
	
	// a name based on the context ID is used to name the generated GPU code executor class for this context
	const char *getContextName();

	void describe(int indent);
  private:
	// It can happen that the computation flow dives into a lower level LPS in the GPU directly from a host
	// level LPS without going through the nesting of the upper level GPU LPS that has been mapped to some higher
	// GPU PPS. Even in those scenarios, we take the first LPS in the path to the entry stage's LPS that has been
	// mapped to the GPU as the context LPS. We will rather do the lower level LPU generation within the 
	// generated kernels as opposed to ship in smaller LPUs to the kernels. Furthermore, our data part allocation
	// scheme for GPU LPUs also demand that host to GPU context switching should happen at the topmost LPS mapped
	// to the GPU. 
	Space *getContextLps(int topmostGpuPps, Space *entryStageLps);	
};

#endif
