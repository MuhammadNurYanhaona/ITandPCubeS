#ifndef _H_stencil_gpu_execution
#define _H_stencil_gpu_execution

/* This header file and corresponding implementation file contains all the classes and functions needed for offloading stencil 
 * computation to the GPU for a specific mapping. When integrated in the compiler, the features of this header file will be 
 * dynamically generated.
 */

#include "../../test-case/stencil/stencil_structure.h"
#include "../../runtime/structure.h"
#include "../../gpu-offloader/gpu_code_executor.h"
#include "../../gpu-offloader/lpu_parts_tracking.h"

class StencilLpuBatchController : public LpuBatchController {
  public:
        StencilLpuBatchController(int lpuCountThreshold, long memConsumptionLimit);
        int calculateLpuMemoryRequirement(LPU *lpu);
        void addLpuToTheCurrentBatch(LPU *lpu);
};

/* This class is provided so that lpu Ids can be constructed inside offloading GPU kernels, as opposed to passing another set
 * of pointers for Ids.
 */
class StencilLpuBatchRange {
  public:
        Range lpuIdRange;
        int lpuCount;
};

class StencilGpuCodeExecutor : public GpuCodeExecutor {
  private:
        // For some task, the partition parameters and the overall dimension ranges of arrays are needed during execution of
        // the compute stages. So these two parameters should be passed as values during each GPU kernel launch  
        stencil::Partition partition;
        stencil::ArrayMetadata arrayMetadata;

        // Task-globals and thread-locals hold the scalar variables used by a task. There is just one instance of the former
        // per segment controller when the each thread has its own version of the latter. In the GPU model, there is just one
        // thread per segment. Regardless, we do not coalesce them together to make host to GPU transition and vice-versa easy.
        // Any change made to any property of either object inside the GPU should be synchronized with the CPU version.  
        stencil::TaskGlobals *taskGlobalsCpu, *taskGlobalsGpu;
        stencil::ThreadLocals *threadLocalsCpu, *threadLocalsGpu;
  public:
        StencilGpuCodeExecutor(LpuBatchController *lpuBatchController,
                        stencil::Partition partition,
                        stencil::ArrayMetadata arrayMetadata,
                        stencil::TaskGlobals *taskGlobals,
                        stencil::ThreadLocals *threadLocals);
        void offloadFunction();
        void initialize();
        void cleanup();
};

#endif
