/* A platform specific macro represents the number of threads (equivalent to the number of
   cores) in each multicore platform.
*/
#define THREAD_COUNT 16

/* Comparing mapping configuration file with the PCubeS description of the multicore CPU, we
   should get a set of thread counts for different spaces. This is possible because the policy
   is to have one thread per core regardless of the number LPUs in individual spaces.
*/
// assuming Space A is mapped to the CPU
#define SPACE_A_THREADS 1
// assuming Space B is mapped to individual cores 
#define SPACE_B_THREADS_PER_SPACE_A 16
// assuming Space C is mapped to a NUMA node
#define SPACE_C_THREADS_PER_SPACE_A 2
// assuming Space D is again mapped to individual cores
#define SPACE_D_THREADS_PER_SPACE_C 8

/* There should be some default data structures for holding properties of each array dimension,
   epoch variables, etc.These should become parts of the compiler library in the future as we 
   determine the set of all such data structures. 
*/

typedef struct {
	int min;
	int max;
} Range;

typedef struct {
	int length;
	Range range;
} Dimension;

typedef struct {
	int beginAt;
	int currentValue;
} Epoch;

/* We should have a data structure holding the metadata of all array variables. This is easy to 
   do as we have the dimension information of all arrays from the scope and type checking phase. 
   Generating this structure thus should be as simple as examining the define-scope of a task and 
   adding metadata for each.
   Name of each element in the structure is the name of corresponding array followed by 'Dims'	 
*/

struct LUArrayMetadata {
	Dimension[2] aDims;
	Dimension[2] uDims;
	Dimension[2] lDims;
	Dimension[1] pDims;
	Dimension[1] l_columnDims;
} arrayMetadata;

/* There should be a data structure for metadata and content of variables that are linked
   in the task environment. This is easy to generate, as we know variable types from scope 
   analysis and if they are linked from the abstract syntax tree.
   These variables and their metadata will be loaded from external files in the main program
   based on command line arguments.
   NOTE: we store all multi-dimensional IT arrays as uni-dimensional C arrays	 
*/

struct LUEnvironmentLinks {
	float *a;
	Dimension[2] aDims;
} links;

/* There should be a data structure holding all task-global scalar variables that are neither
   an epoch or a repeat loop index variable. Any update on a element of this structure should
   protected by barrier/semaphore. 
   Epoch and repeat loop indexes will have thread specific instances.
   We can generate this structure by examining the Define scope of a task and filtering out 
   any epoch variable (based on type information) and loop index (after examining the abstract
   syntax tree)	
*/

struct LUTaskGlobalVars {
	int pivot;
} taskGlobalVars;

/* Thread specific task-global scalar variables, that were filtered out in the previous step,
   should be inside another data structure. Each thread should have a personal copy of this
   structure.
   NOTE: having only one copy for these scalar variables indicate an execution strategy for 
   all LPUs (logical processing units) a particular thread handles (remember that a thread
   is a composite PPU (parallel processing unit) executing codes from possibly different 
   spaces). That is, all LPUs multiplexed in a thread first completes a single phase of 
   execution; then the thread updates its task-global scalar variables.
   During executing the initialization section in the main thread, any assignment to underlying
   variable should be repeated for each thread's version of the structure.
*/

typedef struct {
	int k;
	Epoch t;
} LUThreadGlobalVars;

/* A data structure is needed to demarcate the region of a dimension of an array that falls inside
   a single LPU. The Dimension data structure created above is not adequate in this regard as 
   sometimes we may have a single memory block allocated for a sequence of LPUs where each working 
   on a different region of the allocated structure.
   This data structure should be a part of the compiler data structures library too.  
*/

typedef struct {
	Dimension storageDim;
	Dimension partitionDim;
} PartitionDimension;

/* For each logical space specified in the task, we should have a data structure that holds 
   reference of all arrays that exist -- partitioned or unpartitioned -- in that space. These
   structures can be generated easily by examining the partition hierarchy that we got during
   semantic analysis phase.
   Each thread will ask some partition specific library routine with its thread ID and the ID 
   of the LPU it just executed (that should be -1 for the first invocation). The routine will 
   generate a new instance of the structure on the fly and return it.
   This strategy ensures that space consumption for LPU descriptions is negligeable. Furthermore,
   later on for the optimized compiler, the routine may choose a ordering of LPUs that will
   reduce data transfers.
   NOTE: the number of PartitionDimension references per array is equal to the number of 
   dimensions along with the array is been partitioned in corresponding space.    	   
*/

typedef struct {
	int lpuId;
	int *p;
} SpaceA_LPU;

typedef struct {
	int lpuId;
	float *a;
	PartitionDimension[1] aPartDims;
	float *u;
	PartitionDimension[1] uPartDims;
	float *l;
	PartitionDimension[1] lPartDims;
	float *l_column;
} SpaceB_LPU;

typedef struct {
	int lpuId;
	float *u;
	PartitionDimension[1] uPartDims;
	float *l;
	PartitionDimension[1] lPartDims;
	float *l_column;
} SpaceC_LPU;

typedef struct {
	int lpuId;
	float *u;
	PartitionDimension[1] uPartDims;
	float *l;
	PartitionDimension[1] lPartDims;
} SpaceD_LPU;

/* A set of inline index-transformation functions are needed for strided partitioning. These will
   become part of the partition function definition in the library. In the future, if we allow 
   programmer defined partition functions then similar transformation functions will be needed. 
   Before we enable such a feature, we have to finalize the set of function headers that will suffice
   accurate interpretation of indexes inside the compute block.
   NOTE: all partition functions may not need to implement all transformation functions. For 
   example, neither block_size nor block_count involves any transformation.      
*/

inline int stride_transformIndex(int originalIndex, int strideId) {
	return originalIndex * strideId;
}

inline int stride_revertIndex(int stridedIndex, int strideId) {
	return stridedIndex / strideId;
}

inline bool stride_isIndexIncluded(int originalIndex, int strideId) {
	return (originalIndex % strideId == 0);
}

/* Definition for the function that loads all linked environment variables from external file. The
   main function should pass all command line arguments to this function at the very beginning.
   Each file should have dimentionality information for a specific array followed by the actual
   data. File names in the command line should follow the same sequence as corresponding arrays
   are listed in the Environment section.
   NOTE: we are loading data with metadata as simplified initial implementation. Our broader
   goal is to develop a space and partition oriented I/O paradigm in the future. 	  
*/
LUEnvironmentLinks *LU_loadEnvironmentLinks(int argc, char *argv);

/* Definition of the initialization function corresponding to the initialization block of the
   task. The definition -- although task specific -- should follow a defined pattern. Current
   understanding is that it should take 1) structure for linked environment variables, 2) any
   initialization arguments specified in the task, 3) structure holding all arrays' metadata, 
   4) structure for task-global scalars, and 5) an array of structures for thread-global scalars
   as input arguments in that sequence. Here (1) and (2) will be used to initialize the rests.  	 
*/

void LU_initializeTask(LUEnvironmentLinks *links, 
		LUArrayMetadata *arrayMetaData, 
		LUTaskGlobalVars *taskGlobals, 
		LUThreadGlobalVars *threadGlobals[THREAD_COUNT]);

/* Given that LPU descriptions are generated on the fly, a separate mechanism is needed to hold
   references to arrays used by the task. Given that we adopt the policy of contiguously layout
   the regions of different LPUs of a space, we need only one reference per array per space.
   NOTE: these references need to be allocated new memories only in case the underlying arrays 
   are (1) not allocated at all, or (2) used multiple times in a space that reorder the arrays
   from its previous arrangement. Otherwise, these references will point to some earlier 
   references. 
   For determining access to a variable multiple times we need to do another static analysis of
   the intermediate represention (this should not be difficult). Then we have to mark the entries
   in spaces in the partitition hierarchy that will need allocation. These markings will help to 
   determine data duplication needs during generating code for data communication. Furthermore,
   whether or not a new space reallocates the same data or references to allocation done for another
   space determines if index reordering is needed when translating codes for computation stages.	       
*/

struct SpaceA_Content {
	int *p;
} spaceA;

struct SpaceB_Content {
	float *a;
	float *u;
	float *l;
	float *l_column;	
} spaceB;

/* Allocation for the dynamic portion of the partition hierarchy is done at runtime by the thread
   which will have some dynamic LPUs to execute. So there should be one structure instance per 
   thread that is assigned to process LPUs of a dynamic space. These threads are decided by comparing
   mapping configuration file with the PCubeS description of the hardware.
*/

typedef struct {
	float *u;
	float *l;
	float *l_column;
} SpaceC_Content;

typedef struct {
	float *u;
	float *l;
} SpaceD_Content;

/* There should be a single data structure instance to be maintained the LPU generation library to
   different space contents. For static portion of the hierarchy, this instance should have one 
   reference per space. For any dynamic portion, there should be one reference per possible dynamic 
   root threads. In case of LU factorization Space C is a dynamic root that is mapped to the two NUMA 
   nodes. So there should be two instances. 
*/

struct LUTask_Environment {
	SpaceA_Content *spaceA;
	SpaceB_Content *spaceB;
	SpaceC_Content[2] *spaceC_units;
	SpaceD_Content[2] *spaceD_units;
} taskEnvironment;

/* Definition of the library function that main function invokes at the beginning to do static 
   allocations and set up the references to different environment variables. It takes three parameters:
   first, the structure containting environment references to be initialized and if needed allocated,
   second, the environmental links of arrays loaded from files, and third, dimension information of
   all arrays been used within the task.
   To generate the implementation of this function, we have to do a sequential traversal of the compute
   flow graph generated at the end of static analysis phase. As we discover any array that is not been
   allocated anywhere yet but been read or written in current space will be  allocated. If the array 
   is marked as multiple use, then it will be allocated again regardless of a previous allocation if 
   its partition specification suggests data reordering. Otherwise, it will be assigned the reference 
   of most recent reference made to the same array.
   One exception to the aforementioned rule is the handling of dynamic spaces. Reference set-up and 
   allocation for dynamic spaces are done using separate functions at task execution time.
   NOTE: array allocations should involve zero filling of contents.	      
*/

void LU_initializeStaticReferences(LUTask_Environment *environment, 
		LUEnvironmentLinks *links, 
		LUArrayMetadata *arrayMetaData);

/* Definition of the library functions that threads responsible for initiating a dynamic region should
   invoke during task execution. There should be one function per entrance to a dynamic root. 
   In case of LU factorization we have just one dynamic root (Space C) and two entrance point to it: one 
   during Select Pivot and another during Update Lower.
   Such a function takes ppuId of the thread for that space as input to know which references to update
   in the environment. Implementation of environment update follows the same procedure described above,
   but flow traversal is restricted within the code sections that execute under the dynamic root. The 
   entranceId parameter is used to determine the code section the thread is currently going to enter. In
   case multiple entrances exist, as in LU factorization, entrance specific allocation and reference
   assignment code are separated by case conditions on the entranceId. 
   NOTE: allocations here do not need zero filling as data transfers from static spaces take place before 
   any entrance to a dynamic code block.
   NOTE: LU factorization does not involve any dynamic allocation.
*/
void spaceC_InitializeDynamicReferences(int ppuId, 
		int entranceId, 
		LUTask_Environment *environment, 
		LUArrayMetadata *arrayMetaData, List<int> *activeLPUIds);

/* As dynamic spaces are cleaned up at every exit, there should be corresponding teardown dynamic
   references methods for each dynamic space. This methods should just need the ppuId and spaceId to 
   cleanup all references and, therefore, can have a generic interface. Probably the implementation can
   also be generalized here. 	 
*/
void teardownDynamicReferences(int ppuId, char spaceId);

/* One important efficiency concern is the identification of the dynamic LPUs that get activated in any
   point of task execution. Without the aid of any helper function, this will boil down to checking 
   activating condition against each possible LPU. In spaces with a large number of LPUs such as Space C 
   in LU factorization this comparision may take considerable time to complete.
   To expedite this computation, each partition function should provide its own implementation of three
   index comparison tests of LPUs against any given index. All these comparison tests will return a LPU
   range. The function for actual activation condition will be generated by joining the results of 
   individual subconditions (here an OR will do a union of two ranges and an AND will do intersection).

   In LU factorization, we need function definition of these tests for block_size function only and the
   activation conditions in 'Select Pivot' and 'Update Lower' flow stages are direct application of one
   of these tests.

   NOTE: these function definitions and the LpuIDRange data structure will be part of compiler libraries.  
*/
typedef struct {
	int startId;
	int endId;
} LpuIdRange;

// For LPUs whose data ranges are above the compared index
inline LpuIdRange *block_size_getUpperRange(int comparedIndex, int dimensionLength, int blockSize);
// For LPUs whose data ranges are below the compared index 
inline LpuIdRange *block_size_getLowerRange(int comparedIndex, int dimensionLength, int blockSize);
// For LPUs whose data ranges include the given index (there should be at most one such LPU but we
// maintain a function signature similarity with the others to make composition easier)
inline LpuIdRange *block_size_getInclusiveRange(int comparedIndex, int dimensionLength, int blockSize);

/* Definition of the activation control function for Space C entrance in LU decomposition. Internally
   this function should invoke block_size_getInclusiveRange with appropriate parameters and generate a
   list of LPU ids (there should be just one LPU in the list at a time).
   NOTE: in the function definition, we get to know that only column dimension and block size are needed
   along with the compared index by examining the 1D partitioning in Space A and the variable dimensions
   been compared in the activation conditions.
*/
List<int> getActiveLpuIdsForSelectPivot(int k, int uDimension2Length, int blockSize);
List<int> getActiveLpuIdsForUpdateLower(int k, int lDimension2Length, int blockSize);

/* A structure is needed to group active LPUs of a dynamic space against corresponding PPUs. This is a 
   generic structure that would be a part of the compiler data structures library.
*/
typedef struct {
	int ppuGroupId;
	int *lpuIds;
} LPU_Group;

/* Definition of the method that takes a list of LPU IDs for a dynamic space and returns a new list of 
   LPU_Group. Alongside the LPU IDs it takes the name of the space as an input argument to be able to
   refer to the appropriate partitioning functions for grouping purpose.
   The grouping scheme is simple. The function refers to the partition function of the space for any
   data structure to get the total number of LPUs. The number of PPUs for the underlying space is known
   from the mapping configuration. Since each PPU handles a section of consecutive LPUs from the entire
   range of LPUs, determining what belongs to whom is easy.    
*/
List<LPU_Group*> *groupLPUsUnderPPUs(char spaceId, List<int> *activeLPUList);

/* A structure is needed to hold the PPU (parallel processing unit) Id of a thread for a space when it 
   is executing LPUs from that space. For higher level spaces a PPU group Id is also needed to determine 
   if the thread will enter any composite computation stage (such as "Update Lower" in LU factorization) 
   or not.
   NOTE: if a thread is a PPU for a space it should have nonnegative ppuId; otherwise, its ppuId will 
   be -1 but ppuGroupId will be the Id of the PPU controlling the group. For PPUs of lower-most spaces in 
   the hierarchy both ids will be equal. This structure will be a part of the compiler library.   
*/
typedef struct {
	int ppuId;
	int ppuGroupId;
} PPU_Ids;

/* A structure is needed to hold the ppu-ids of threads for all the spaces defined in the task. As we
   adopt a thread-per-core execution methods, these Ids are fixed and can be generated in the main function
   before task execution begns.
*/
typedef struct {
	PPU_Ids spaceA_ids;
	PPU_Ids spaceB_ids;
	PPU_Ids spaceC_ids;
	PPU_Ids spaceD_ids; 	
} LU_threadIds;

/* We define a constant for invalid ID
*/
#define INVALID_ID -1

/* Function definitions of the computation stages of LU factorization. The name of the functions are 
   translated into camel-case notations from their original names. Each function should take 1) a 
   reference to the LPU it is going to execute, 2) PPU ids for the executing thread, 3) the structure
   referring to the task-global scalar variables, and 4) a reference to the thread-local global global
   scalar variables. The return type is always void as all updates are done through the lpu variable.   
*/
void prepare(SpaceB_LPU *lpu, LU_threadIds *threadIds, 
		LUTaskGlobalVars *taskGlobals, LUThreadGlobalVars *threadGlobals);
void selectPivot(SpaceC_LPU *lpu, LU_threadIds *threadIds, 
		LUTaskGlobalVars *taskGlobals, LUThreadGlobalVars *threadGlobals);
void storePivot(SpaceA_LPU *lpu, LU_threadIds *threadIds, 
		LUTaskGlobalVars *taskGlobals, LUThreadGlobalVars *threadGlobals);
void interchangeRows(SpaceB_LPU *lpu, LU_threadIds *threadIds, 
		LUTaskGlobalVars *taskGlobals, LUThreadGlobalVars *threadGlobals);

/* NOTE: there is no function defition for composite stage "Update Lower"; rather there are definitions
   for its internal computation stages. This is just a matter of policy. Here we decided to keep all  
   management operations to be outside actual computation function. Therefore, composite stages are
   broken down and the run method of each thread will emulate the logic of those stages by calling their
   internal stages in appropriate places.  
*/
void calculatePivotColumn(SpaceD_LPU *lpu, LU_threadIds *threadIds, 
		LUTaskGlobalVars *taskGlobals, LUThreadGlobalVars *threadGlobals);
void updateSharedStructure(SpaceC_LPU *lpu, LU_threadIds *threadIds, 
		LUTaskGlobalVars *taskGlobals, LUThreadGlobalVars *threadGlobals);

void updateUpper(SpaceB_LPU *lpu, LU_threadIds *threadIds, 
		LUTaskGlobalVars *taskGlobals, LUThreadGlobalVars *threadGlobals);

/* TODO Interface definition to the LPU management library for retrieving the next LPU a thread going to
   execute for any space code it is responsible for. There is a lot to be thought about regarding the
   implementation of this library. Hopefully a large part of it can be made generic. For now, I do not
   have any concrete plan for this. At any rate, the following interface definition should be adequate

   The function returns a void LPU pointer that is casted to appropriate LPU type based on the context.
   To be able to determine which LPU to return it needs to know 1) the ID of the space currently under
   concern, 2) PPU group Id of the invoking thread for current space, and 3) the id of the previous LPU
   the thread just executed in this space.  	
*/
void *getNextLPUchar spaceId, int groupId, int previousLPU_id);

/* For each communication between a pair of spaces that do not have any order relationship between them,
   we need to determine if a pair of LPUs taken from opposite spaces have overlapping data regions. The
   compiler will spew one such checking method for each un-ordered space pair for each data exchange for
   which there is a dependency arc in the intermediate code. These method have the common signature of
   two LPU definitions from two spaces and the variable name as inputs and returning a boolean output
   as the verdict on overlapping.

   To generate these methods, the compiler checks the two partition configurations for the variable in 
   two concerned spaces then call the intersect method (that are known for all pair of built-in partition 
   functions) for each dimension of the variable. Then it ANDs the outcomes of those method.

   For user defined partition functions, if intersect method is not provided, an interval matching will
   be done for the index ranges of the LPUs. This would be a linear time operation on the number of 
   intervals of the LPUs.

   For LU factorization, only one such method is required to check transitions between Space C and B   	 	 	  	
*/
bool *doLPUsOverlap(SpaceB_LPU *lpu1, SpaceC_LPU *lpu2, char *variableName);

/* Above methods are utilized to determine if two PPUs have overlapping in any pair of LPUs multiplexed
   to them. So methods similar to the above should be generated for each PPU pair. These methods should 
   invoke appropriate methods for LPUs intersection given above. If any space in the pair is dynamic then
   we need the ID list of active LPUs for that space along with the ID of the PPU as an input argument.
   
   In case of LU factorization, Space C is dynamic. Therefore we have the following signature for the 
   sole method that does intersection checking.      
*/
bool *doPPUsOverlap(int SpaceB_id, int SpaceC_id, List<int> *SpaceC_LPUs, char *variableName);

/* Definition of the run method for a thread. It takes 1) the structure instace containing thread's PPU
   IDs for different spaces, 2) references of task global scalars, 3) references of its thread global 
   scalars, and 4) array metadata as input arguments. Array references will be safeguarded by library 
   routines that we yet to define. So they are not needed as inputs.
   To generate this method, we have to traverse the flow graph that been generated at the end of the 
   static analysis phase. As we visit each flow-stage in the definition, we have to layout the code for
   executing it within the run method. The difference from previous traversals to that graph from this
   one is that we have to consider multiplexing.    
*/
void run(LU_threadIds *threadIds, 
		LUTaskGlobalVars *taskGlobals, LUThreadGlobalVars *threadGlobals, 
		LUArrayMetadata *arrayMetadata) {
	
	//************************************** Prepare *******************************************/

	// check and execute the "Prepare" stage
	if (threadIds->spaceB_ids.ppuId != INVALID_ID) {
		SpaceB_LPU *spaceB_LPU = NULL;
		int spaceB_lpuId = INVALID_ID;
		// NOTE: because of the strided partitioning, this while loop should iterate only once 
		while ((spaceB_LPU = getNextLPU('B', 
				threadIds->spaceB_ids.groupId, spaceB_lpuId)) != NULL) {
			prepare(spaceB_LPU, threadIds, taskGlobals, threadGlobals);
			spaceB_lpuId = spaceB_LPU->lpuId;
		} 
	}

	//********************************* Repeat Cycle ********************************************/

	// Space A is unpartitioned so there is no need for going through the library for next LPU in
        // this case: we know all threads should enter this repeat cycle.

        // Translation of the repeat loop into a for loop can be done by checking the repeat condition.
	// The in condition's range should be translated into the loop start and end points. If there
        // is a step expression that will act as the loop increment -- default loop increment is the
        // constant 1. Any additional restriction, if exists, in the range will augment with the 
	// iteration checking through an AND. 
	int loopStart = arrayMetadata->aDims[1].range.min;
	int loopEnd = arrayMetadata->aDims[1].range.max;
 	for (threadGlobals->k = loopStart; threadGlobals->k <= loopEnd; threadGlobals->k += 1) {
		

		//************************* Dynamic Computation Block ********************************/	

		// The first stage in the loop is a conditional 'Dynamic Computation' block. Therefore,
		// we need to execute the activation condition in each PPU to determine the list of LPUs
		// that will be active for current iteration. Here all PPUs execute the same function
		// for the same parameters.
		List<int> lpuIds = getActiveLpuIdsForSelectPivot(threadGlobals->k, 
				arrayMetadata->uDims[2].length, 1);
		// The LPU Ids received in the previous step needs to be grouped under PPUs to decide 
		// there respective roles regarding handling dynamic computation in current iteration.
		List<LPU_Group*> lpuGroups = groupLPUsUnderPPUs('C', lpuIds);

		// From LPU IDs vs PPU group Id map a thread will retrieve its own LPU list. 
		List<int> myLpuIds = LPU_Groups->Nth(threadIds->spaceC_ids.groupId);
		
		// since this is a conditionally executed dynamic block, dynamic references need to 
		// updated for all threads of the  NUMA node that has its myLpuIds list non-empty.
		// NOTE: in general case this initialization should be done by only one thred, the
		// first thread in the NUMA node, and every other thread in the node should wait on
		// some barrier for the former to finish initialization.
		// However, given that this dynamic block is executed only once in per iteration and
		// there is no dynamic allocation, all threads of the NUMA node can do the same
		// initialization separately that will only store the active LPU lists in the runtime
		// library. Compiler can decide to do this optimization by traversing the computation 
		// flow nested within the repeat loop. 
		if (myLpuIds != NULL) {
			spaceC_InitializeDynamicReferences(threadIds->SpaceC_ids.ppuId, 
					1, environment, arrayMetaData, myLpuIds);		
		}	

		// The first stage within the dynamic computation block is 'Data Loader Sync.' It has
		// a unordered space transition dependency (B -> C) on variable u from 'Prepare' or
		// 'Update Upper' stage. So there is a need for detecting Space B to Space C 
		// intersections among different PPUs. 

		// each PPU will determine if it has intersection with other PPUs for u. If there is 
		// any intersection with any other PPU, it will up a binary semaphore indicating that
		// it has finished its previous step
		bool intersactionFound = false;
		for (int i = 0; i < lpuGroups->NumElements(); i++) {
			LPU_Group *currentGroup = lpuGroups->Nth(i);
			if (currentGroup->lpuIds == NULL) continue;
			intersectionFound = doPPUsOverlap(threadIds->spaceB_ids.ppuId, 
				currentGroup->ppuGroupId, currentGroup->lpuIds, "u");
			if (intersectionFound) break;
		}
		if (intersactionFound) {
			// TODO up a binary semaphore to indicate the waiting PPU can proceed 
		}
		
		// A thread decides if it is a part of the Space C group that will execute the 'Data
		// Loader Sync' and 'Select Pivot' stages for current iteration. 
		if (myLpuIds != NULL) {
			
			//************************* Data Loader Sync *********************************/
			
			// Only the PPU with valid Space C PPU Id (the first thread in the active NUMA 
			// node) should execute 'Data Loader Sync' stage. Therefore it should be the
			// one that should wait for data synchronization
			if (threadIds->spaceC_ids.ppuId != INVALID_ID) {
				// find intersections with other PPUs.
				List<int> *intersectingPPUs = new List<int>;
				bool intersactionFound = false;
				for (int i = 0; i < SPACE_B_THREADS_PER_SPACE_A; i++) {
					intersectionFound = doPPUsOverlap(i, 
							threadIds->spaceC_ids.ppuId, myLpuIds, "u");
					if (intersectionFound) {
						intersectingPPUs->Append(i);
					}
				}
				// TODO implement a wait for semantics for the currently active thread
			}	
			
			//*************************** Select Pivot ***********************************/	
			
			// there is a dependeny from 'Data Loader Sync' (executed by a single thread in 
			// the NUMA node) to 'Select Pivot' (executed by all threads in the NUMA node).
			// So there should be a barrier synchronization or counting semaphore based
			// signaling should take place before actual method call.
			
			// TODO intra-NUMA node barrier synchronization
			
			// then invoke 'Select Pivot' function for active LPUs (there should be one per 
			// iteration) 
			SpaceC_LPU *spaceC_LPU = NULL;
			while ((spaceC_LPU = getNextLPU('C', 
					threadIds->SpaceC_ids.groupId, spaceC_lpuId)) != NULL) {
				selectPivot(spaceC_LPU, threadIds, taskGlobals, threadGlobals);
			}

			// There is an outgoing dependency from 'Select Pivot' to 'Data Restorer Sync'
			// of Space A. So the controller thread of the NUMA node should signal on a 
			// semaphore that pivot selection is done		
			if (threadIds->SpaceC_ids.ppuId != INVALID_ID) {
				// TODO  signal on a semaphore 		
			}
		}

		// The movement from 'Dynamic Computation' to 'Data Restorer Sync' indicates an exit
		// from a dynamic space for all PPUs participating in the dynamic computation. So all
		// threads should call a teardown reference method to get rid of any references and 
		// free allocated memories (no allocation is done in this case).
		teardownDynamicReferences(threadIds->SpaceC_ids.ppuId, 'C');
		
		//****************************** Data Restorer Sync **********************************/	
		
		if (threadIds->SpaceA_ids.ppuId != INVALID_ID) {
			// TODO wait on the semaphore to be signaled by the thread selecting the pivot.
			// QUESTION: how does this thread know that there is only one NUMA node that 
			// updated the pivot. Or should we make both NUMA nodes to signal on a counting
			// semaphore?	
		}

		//********************************* Store Pivot **************************************/
	
		// There is a dependency from 'Data Restorer Sync' to 'Store Pivot'. The compiler skips
		// that because Space A is unpartitioned and the same thread will execute both stages.	 
		if (threadIds->SpaceA_ids.ppuId != INVALID_ID) {
			SpaceA_LPU *spaceA_LPU = getNextLPU('A', threadIds->SpaceA_Ids.groupId, NULL);
			storePivot(spaceA_LPU, threadIds, taskGlobals, threadGlobals);
		}

		//******************************* Interchange Rows ***********************************/
		
		// There is a dependency from 'Data Restorer Sync' to 'Interchange Rows' on scaler pivot
		// variable. Since all threads execute Space B code this can be implemented as a counting
		// semaphore signalled by the sole Space A thread and everyone else waiting on it.

		if (threadIds->SpaceA_ids.ppuId != INVALID_ID) {
			// TODO set the counter of the semaphore to (total number of threads - 1)	
		} else {
			// TODO wait on the semaphore.
		}

		// check and execute the "Interchange Rows" stage
		if (threadIds->spaceB_ids.ppuId != INVALID_ID) {
			SpaceB_LPU *spaceB_LPU = NULL;
			int spaceB_lpuId = INVALID_ID;
			// NOTE: because of the strided partitioning, this while loop should iterate only 
			// once 
			while ((spaceB_LPU = getNextLPU('B', 
					threadIds->spaceB_ids.groupId, spaceB_lpuId)) != NULL) {
				interchangeRows(spaceB_LPU, threadIds, taskGlobals, threadGlobals);
				spaceB_lpuId = spaceB_LPU->lpuId;
			} 
		}
		
		//************************* Dynamic Computation Block ********************************/	
		
		// Just like the 'Select Pivot' compute stage, 'Update Lower' is also nested inside a
		// dynamic conditional compute block. So again we have to determine the LPU IDs that will
		// active for current iteration (again there would be just one). Then group the LPUs 
		// under PPUs.
		lpuIds = getActiveLpuIdsForUpdateLower(threadGlobals->k, 
				arrayMetadata->uDims[2].length, 1);
		lpuGroups = groupLPUsUnderPPUs('C', lpuIds);
		myLpuIds = LPU_Groups->Nth(threadIds->spaceC_ids.groupId);
		if (myLpuIds != NULL) {
			spaceC_InitializeDynamicReferences(threadIds->SpaceC_ids.ppuId, 
					2, environment, arrayMetaData, myLpuIds);		
		}

		// Again, like in the 'Select Pivot' compute stage, the first stage in 'Update Lower' 
		// is a loader sync that has input dependencies on both 'u' and 'l' (previous time there
		// was just one dependency) to previous compute stage executed in Space B. So updaters
		// of the portion of 'u' and 'l' that would be used next need to signal on semaphore 
		// that they have finished their computation and activated LPUs for Space C (again there
		// should be just one in this case) can proceed.
		// For this to be done, similar code for intersection checking need to be put in place 	
		intersactionFound = false;
		for (int i = 0; i < lpuGroups->NumElements(); i++) {
			LPU_Group *currentGroup = lpuGroups->Nth(i);
			if (currentGroup->lpuIds == NULL) continue;
			intersectionFound = doPPUsOverlap(threadIds->spaceB_ids.ppuId, 
				currentGroup->ppuGroupId, currentGroup->lpuIds, "u");
			if (intersectionFound) break;
			intersectionFound = doPPUsOverlap(threadIds->spaceB_ids.ppuId, 
				currentGroup->ppuGroupId, currentGroup->lpuIds, "l");
			if (intersectionFound) break;
		}
		if (intersactionFound) {
			// TODO up a binary semaphore to indicate the waiting PPU can proceed 
		}
		
		// A thread decides if it is a part of the Space C group that will execute the 'Data
		// Loader Sync' and 'Update Lower' stages for current iteration. 
		if (myLpuIds != NULL) {
			
			//************************* Data Loader Sync *********************************/
			
			// Again, only the PPU with valid Space C PPU Id (the first thread in the active 
			// NUMA node) should execute 'Data Loader Sync' stage. Therefore it should be 
			// the one that should wait for data synchronization
			if (threadIds->spaceC_ids.ppuId != INVALID_ID) {
				// find intersections with other PPUs.
				List<int> *intersectingPPUs = new List<int>;
				bool intersactionFound = false;
				for (int i = 0; i < SPACE_B_THREADS_PER_SPACE_A; i++) {
					intersectionFound = doPPUsOverlap(i, 
							threadIds->spaceC_ids.ppuId, myLpuIds, "u");
					if (intersectionFound) {
						intersectingPPUs->Append(i);
					}
				}
				// TODO implement a wait for semantics for the currently active thread
			}	
			
			//****************************** Update Lower *********************************/

			// 'Update Lower' is a composite stage containing both Space C and D compute
			// stages. Hence, all threads in the active NUMA group should enter this block
			// and no additional condition checking is needed here.

			//************************ Calculate Pivot Column *****************************/
			
			// 'Calculate Pivot Column' has input dependency from 'Data Loader Sync'. Since 
			// all threads in the NUMA node participate in the former and only one in the 
			// later, an intra NUMA node couting semaphore signaling should be enough.
			
			// TODO intra-NUMA node simaphore signaling or barrier synchronization

			// check and execute the "Calculate Pivot Column" stage
			if (threadIds->spaceD_ids.ppuId != INVALID_ID) {
				SpaceD_LPU *spaceD_LPU = NULL;
				int spaceD_lpuId = INVALID_ID;
				while ((spaceD_LPU = getNextLPU('D', 
					threadIds->spaceD_ids.ppuId, spaceD_lpuId)) != NULL) {
				calculatePivotColumn(spaceD_LPU, threadIds, taskGlobals, threadGlobals);
				spaceD_lpuId = spaceD_LPU->lpuId;
			}

			//************************** Data Restorer Sync ******************************/

			// Only the controller thread of the NUMA node executes 'Data Restorer Sync' for
			// the update dependency on 'l' produced by 'Calcuate Pivot Column' executed by
			// every thread in the NUMA node. So a simple synchronization is to make the first
			// thread wait on a counting semaphore that will be signalled everyone else.

			if (threadIds->spaceC_ids.ppuId == INVALID_ID) {
				// TODO signal the counting semaphore
			} else {
				// TODO wait on the counting semaphore
			}	
			
			//************************ Update Shared Structure ****************************/

			// There is a update dependency from 'Data Restorer Sync' to 'Update Shared Str-
			// ucture' for 'l', but in the absense of replication transitions between stages
			// of same LPS need no synchronization. Therefore, this synchronization is skipped.

			if (threadIds->spaceC_ids.ppuId != INVALID_ID) {
				// then invoke 'Select Pivot' function for active LPUs (there should be 
				// one per iteration) 
				SpaceC_LPU *spaceC_LPU = NULL;
				while ((spaceC_LPU = getNextLPU('C', 
						threadIds->SpaceC_ids.groupId, spaceC_lpuId)) != NULL) {
					updateSharedStructure(spaceC_LPU, threadIds, 
							taskGlobals, threadGlobals);
				}
				// There is an outgoing dependency from 'Update Shared Structure' to 'Data 
				// Restorer Sync' of Space A. So the controller thread of the NUMA node 
				// should signal on a semaphore that update is done

				// TODO  signal on a semaphore 		
			}
		}

		// The movement from 'Dynamic Computation' to 'Data Restorer Sync' indicates an exit
		// from a dynamic space for all PPUs participating in the dynamic computation. So all
		// threads should call a teardown reference method to get rid of any references and 
		// free allocated memories (no allocation is done in this case).
		teardownDynamicReferences(threadIds->SpaceC_ids.ppuId, 'C');
		
		//****************************** Data Restorer Sync **********************************/	
		
		if (threadIds->SpaceA_ids.ppuId != INVALID_ID) {
			// TODO wait on the semaphore to be signaled by the thread updated the shared
			// l_column data structure.
		}
		
		//********************************* Update Upper *************************************/
		
		// There is a dependency from 'Data Restorer Sync' to 'Update Upper' on shared l_column
		// variable. Since all threads execute Space B code this can be implemented as a counting
		// semaphore signalled by the sole Space A thread and everyone else waiting on it.

		if (threadIds->SpaceA_ids.ppuId != INVALID_ID) {
			// TODO set the counter of the semaphore to (total number of threads - 1)	
		} else {
			// TODO wait on the semaphore.
		}

		// check and execute the "Update Upper" stage
		if (threadIds->spaceB_ids.ppuId != INVALID_ID) {
			SpaceB_LPU *spaceB_LPU = NULL;
			int spaceB_lpuId = INVALID_ID;
			// NOTE: because of the strided partitioning, this while loop should iterate only 
			// once 
			while ((spaceB_LPU = getNextLPU('B', 
					threadIds->spaceB_ids.groupId, spaceB_lpuId)) != NULL) {
				updateUpper(spaceB_LPU, threadIds, taskGlobals, threadGlobals);
				spaceB_lpuId = spaceB_LPU->lpuId;
			} 
		}

	//******************************* End of the Repeat Cycle *************************************/
	}
}



