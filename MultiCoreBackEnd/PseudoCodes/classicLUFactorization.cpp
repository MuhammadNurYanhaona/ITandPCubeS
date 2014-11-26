/* A platform specific macro represents the number of threads (equivalent to the number of
   cores) in each multicore platform.
*/
#define THREAD_COUNT 16

/* Comparing mapping configuration file with the PCubeS description of the multicore CPU, we
   should get a set of thread counts for different spaces. This is possible because the policy
   is to have one thread per core regardless of the number PCUs in individual spaces.
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
   Thess variables and their metadata will be load from external files in the main program
   based on the command line arguments.
   NOTE: we store all multi-dimensional IT arrays as uni-dimensional C arrays	 
*/

struct LUEnvironmentLinks {
	float *a;
	Dimension[2] aDims;
} links;

/* There should be a data structure holding all task-global scalar variables that are neither
   an epoch or a repeat loop index variable. Any update on element of this structure should
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
   all PCUs (parallel computational units) a particular thread handles (remember that a thread
   is a composite PPU (parallel processing unit) executing codes from possibly different 
   spaces). That is, all PCUs multiplexed in a thread first completes a single phase of 
   execution; then the thread updates its task-global scalar variables.
   During executing the initialization section n the main thread, any assignment to underlying
   variable should be repeated for each thread's version of the structure.
*/

typedef struct {
	int k;
	Epoch t;
} LUThreadGlobalVars;

/* A data structure is needed to demarcate the region of a dimension of an array that falls inside
   a single PCU. The Dimension data structure created above is not adequate in this regard as 
   sometimes we may have a single memory block allocated for a sequence of PCUs where each 
   working on a different region of the allocated structure.
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
   of the PCU it just executed (that should be -1 for the first invocation). The routine will 
   generate a new instance of the structure on the fly and return it.
   This strategy ensures that space consumption for PCU descriptions is negligeable. Furthermore,
   later on for the optimized compiler, the routine may choose a ordering of PCUs that will
   reduce data transfers.
   NOTE: the number of PartitionDimension references per array is equal to the number of 
   dimensions along with the array is been partitioned in corresponding space.    	   
*/

typedef struct {
	int pcuId;
	int *p;
} SpaceA_PCU;

typedef struct {
	int pcuId;
	float *a;
	PartitionDimension[1] aPartDims;
	float *u;
	PartitionDimension[1] uPartDims;
	float *l;
	PartitionDimension[1] lPartDims;
	float *l_column;
} SpaceB_PCU;

typedef struct {
	int pcuId;
	float *u;
	PartitionDimension[1] uPartDims;
	float *l;
	PartitionDimension[1] lPartDims;
	float *l_column;
} SpaceC_PCU;

typedef struct {
	int pcuId;
	float *u;
	PartitionDimension[1] uPartDims;
	float *l;
	PartitionDimension[1] lPartDims;
} SpaceD_PCU;

/* A set of inline index-transformation functions are needed for strided partitioning. These will
   become part of the partition function definition in the library. In the future, if we allow 
   programmer defined partition functions then similar transformation functions will be needed. 
   Before we enable such a feature, we have to finalize the set of function headers that will suffice
   accurate interpretation of indexes inside the compute block.
   NOTE: all partition function should not need to implement all transformation functions. For 
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

/* Definition for function that loads all linked environment variables from external file. The
   main function should pass all command line arguments to this function at the very beginning.
   Each file should have dimentionality information for a specific array followed by the actual
   data. File names in the command line should follow the same sequence as corresponding arrays
   are listed in the Environment section.
   NOTE: we are loading data with metadata as simplified initial implementation. Our broader
   goal is to develop a space and partition oriented IO paradigm in the future. 	  
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

/* Given that PCU descriptions are generated on the fly, a separate mechanism is needed to hold
   references to arrays used by the task. Given that we adopt the policy of contiguously layout
   the regions of different PCUs of a space, we need only one reference per array per space.
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
   which will have some dynamic PCUs to execute. So there should be one structure instance per 
   thread that is assigned to process PCUs of a dynamic space. These threads are decided by comparing
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

/* There should be a single data structure instance to be maintained the PCU generation library to
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
   any entrance to a dynamic code bock.
   NOTE: LU factorization does not involve any dynamic allocation.
*/

void spaceC_InitializeDynamicReferences(int ppuId, 
		int entranceId, 
		LUTask_Environment *environment, 
		LUArrayMetadata *arrayMetaData, List<int> *activePCUIds);

/* One important efficiency concern is the identification of the dynamic PCUs that get activated in any
   point of task execution. Without the aid of any helper function, this will boil down to checking 
   activating condition against each possible PCU. In spaces with a large number of PCUs such as Space C 
   in LU factorization this comparision may take considerable time to complete.
   To expedite this computation, each partition function should provide its own implementation of three
   index comparison tests of PCUs against any given index. All these comparison tests will return a PCU
   range. The function for actual activation condition will be generated by joining the results of 
   individual subconditions (here an OR will do a union of two ranges and an AND will do intersection).

   In LU factorization, we need function definition of these tests for block_size function only and the
   activation conditions in 'Select Pivot' and 'Update Lower' flow stages are direct application of one
   of these tests.

   NOTE: these function definitions and the PcuIDRange data structure will be part of compiler libraries.  
*/

typedef struct {
	int startId;
	int endId;
} PcuIdRange;

// For PCUs whose data ranges are above the compared index
inline PcuIdRange *block_size_getUpperRange(int comparedIndex, int dimensionLength, int blockSize);
// For PCUs whose data ranges are below the compared index 
inline PcuIdRange *block_size_getLowerRange(int comparedIndex, int dimensionLength, int blockSize);
// For PCUs whose data ranges include the given index (there should be at most one such PCU but we
// maintain a function signature similarity with the others to make composition easier)
inline PcuIdRange *block_size_getInclusiveRange(int comparedIndex, int dimensionLength, int blockSize);

/* Definition of the activation control function for Space C entrance in LU decomposition. Internally
   this function should invoke block_size_getInclusiveRange with appropriate parameters and generate a
   list of PCU ids (there should be just one PCU in the list at a time).
   NOTE: in the function definition, we get to know that only column dimension and block size are needed
   along with the compared index by examining the 1D partitioning in Space A and the variable dimensions
   been compared in the activation conditions.
   NOTE: the last parameter is needed to filter in only those PCUs the current thread/thread-group is
   responsible for processing.  
*/

List<int> getActivePcuIdsForSelectPivot(int k, int uDimension2Length, int blockSize, int ppuGroupId);
List<int> getActivePcuIdsForUpdateLower(int k, int lDimension2Length, int blockSize, int ppuGroupId);

/* A structure is needed to hold the PPU (parallel processing unit) Id of a thread for a space when it 
   is executing PCUs from that space. For higher level spaces a PPU group Id is also needed to determine 
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
   reference to the PCU it is going to execute, 2) PPU ids for the executing thread, 3) the structure
   referring to the task-global scalar variables, and 4) a reference to the thread-local global global
   scalar variables. The return type is always void as all updates are done through the pcu variable.   
*/

void prepare(SpaceB_PCU *pcu, LU_threadIds *threadIds, 
		LUTaskGlobalVars *taskGlobals, LUThreadGlobalVars *threadGlobals);
void selectPivot(SpaceC_PCU *pcu, LU_threadIds *threadIds, 
		LUTaskGlobalVars *taskGlobals, LUThreadGlobalVars *threadGlobals);
void storePivot(SpaceA_PCU *pcu, LU_threadIds *threadIds, 
		LUTaskGlobalVars *taskGlobals, LUThreadGlobalVars *threadGlobals);
void interchangeRows(SpaceB_PCU *pcu, LU_threadIds *threadIds, 
		LUTaskGlobalVars *taskGlobals, LUThreadGlobalVars *threadGlobals);

/* NOTE: there is no function defition for composite stage "Update Lower"; rather there are definitions
   for its internal computation stages. This is just a matter of policy. Here we decided to keep all  
   management operations to be outside actual computation function. Therefore, composite stages are
   broken down and the run method of each thread will emulate the logic of those stages by calling their
   internal stages in appropriate places.  
*/
void calculatePivotColumn(SpaceD_PCU *pcu, LU_threadIds *threadIds, 
		LUTaskGlobalVars *taskGlobals, LUThreadGlobalVars *threadGlobals);
void updateSharedStructure(SpaceC_PCU *pcu, LU_threadIds *threadIds, 
		LUTaskGlobalVars *taskGlobals, LUThreadGlobalVars *threadGlobals);

void updateUpper(SpaceB_PCU *pcu, LU_threadIds *threadIds, 
		LUTaskGlobalVars *taskGlobals, LUThreadGlobalVars *threadGlobals);

/* TODO Interface definition to the PCU management library for retrieving the next PCU a thread going to
   execute for any space code it is responsible for. There is a lot to be thought about regarding the
   implementation of this library. Hopefully a large part of it can be made generic. For now, I do not
   have any concrete plan for this. At any rate, the following interface definition should be adequate

   The function returns a void PCU pointer that is casted to appropriate PCU type based on the context.
   To be able to determine which PCU to return it needs to know 1) the ID of the space currently under
   concern, 2) PPU group Id of the invoking thread for current space, and 3) the id of the previous PCU
   the thread just executed in this space.  	
*/
void *getNextPCU(char spaceId, int groupId, int previousPCU_id);

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
	
	// check and execute the "Prepare" stage
	if (threadIds->spaceB_ids.ppuId != INVALID_ID) {
		SpaceB_PCU *spaceB_PCU = NULL;
		int spaceB_pcuId = INVALID_ID;
		// NOTE: because of the strided partitioning, this while loop should iterate only once 
		while ((spaceB_PCU = getNextPCU('B', 
				threadIds->spaceB_ids.groupId, spaceB_pcuId)) != NULL) {
			prepare(spaceB_PCU, threadIds, taskGlobals, threadGlobals);
			spaceB_pcuId = spaceB_PCU->pcuId;
		} 
	}

	// TODO there should be a barrier or other from of wait here that we need to decide

	// Space A is unpartitioned so there is no need for going through the library for next PCU in
        // this case: we know all threads should enter this repeat cycle.

        // Translation of the repeat loop into a for loop can be done by checking the repeat condition.
	// The in condition's range should be translated into the loop start and end points. If there
        // is a step expression that will act as the loop increment -- default loop increment is the
        // constant 1. Any additional restriction, if exists, in the range will augment with the iteration 
	// checking through an AND. 
	int loopStart = arrayMetadata->aDims[1].range.min;
	int loopEnd = arrayMetadata->aDims[1].range.max;
 	for (threadGlobals->k = loopStart; threadGlobals->k <= loopEnd; threadGlobals->k += 1) {
		
		// the first stage in the loop 'Select Pivot' executes in a dynamic region of the space
		// hierarchy. Therefore, first a call is needed to get the list of PCUs that will be 
		// activated for the current thread.
		List<int> pcuIds = getActivePcuIdsForSelectPivot(threadGlobals->k, 
				arrayMetadata->uDims[2].length, 1, threadIds->spaceC_ids->groupId);
		if (pcuIds->NumElements() != 0) {
			if (threadIds->spaceC_ids.ppuId != INVALID_ID) {
				
				// TODO IMPLEMENTATION OF DATA LOADER SYNC HERE IS A HUGE CONCERN. HOW DO
				// WE KNOW WHO IS(ARE) THE UPDATER(S) OF REGIONS IN OTHER SPACE THAT THIS
				// THREAD NEEDS FOR ITS WORK? SO FAR THIS IS THE FIRST SOURCE OF UN-INTENDED 
				// SYNCHRONIZATION. WHAT TO DO??? 

				spaceC_InitializeDynamicReferences(threadIds->spaceC_ids.ppuId, 
						1, environment, arrayMetaData, pcuIds);
				// TODO signal to rest of its group that configuration update is done
			} else {
				// TODO threads within this NUMA node should wait for configuration update
				// in the library	
			}
			// check and execute "Select Pivot" stage
			// NOTE: this time we check if the groupId is invalid. This is because we know 
			// from static analysis that there is a reduction in select pivot and all threads
			// within a Space C group should participate in it.
			if (threadIds->spaceC_ids.groupId != INVALID_ID) {
				SpaceC_PCU *spaceC_PCU = NULL;
				int spaceC_pcuId = INVALID_ID;
				while ((spaceC_PCU = getNextPCU('C', 
						threadIds->spaceC_ids.groupId, spaceC_pcuId)) != NULL) {
					selectPivot(spaceC_PCU, threadIds, taskGlobals, threadGlobals);
				}
			}
			// TODO The controller thread of the NUMA node should signal for "Data Restorer 
			// Sync" stage that follows that pivot has been updated.
		}

		// The next stage is a data restorer sync running in Space A. 
	}
}



