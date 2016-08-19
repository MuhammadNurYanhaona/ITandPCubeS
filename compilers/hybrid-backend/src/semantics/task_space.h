#ifndef _H_task_space
#define _H_task_space

#include "../utils/list.h"
#include "../utils/hashtable.h"
#include "../syntax/ast_def.h"
#include "../syntax/ast.h"
#include "../syntax/location.h"
#include "../syntax/ast_type.h"
#include "../static-analysis/usage_statistic.h"

#include <fstream>

/*	Partition Order specifies in what order individual partitions (or subpartitions) of a space will be
	generated and processed by the runtime. This parameter is mostly relevant for subpartitioned spaces
	as in normal cases partitions of a space are independent. 
*/
enum PartitionOrder { AscendingOrder, DescendingOrder, RandomOrder };

/*	Partition Link Type specifies how partitions of a single data structure in a lower space is linked to 
	its parent partition in higher space. Some-times this linking may be specified in the Space level that
	dictates the relationships of all its data structure partitions to partitions of some parent space.
	Other-times, relationships are defined data structure by data structure basis. If like type is
	
	LinkTypePartition: then lower space partition further divides upper space partitions
	LinkTypeSubpartition: then lower space partition divides each upper space subpartitions
	LinkTypeUndefined: then there is no linkage type currently defined. Some may be derived from other 
			   information
*/
enum PartitionLinkType { LinkTypePartition, LinkTypeSubpartition, LinkTypeUndefined };

class PartitionArg;
class Space;
class Symbol;
class PCubeSModel;

/*	This class stores partition funcion arguments regarding a single dimension of a task global array 
	within a single space configuration. 
*/
class DataDimensionConfig {
  protected:
	int dimensionNo;
	Node *dividingArg;
	Node *frontPaddingArg;
	Node *backPaddingArg;
  public:
	DataDimensionConfig(int dimensionNo, Node *dividingArg);
	DataDimensionConfig(int dimensionNo, Node *dividingArg, 
			Node *frontPaddingArg, Node *backPaddingArg);
	void setDividingArg(Node *dividingArg) {this->dividingArg = dividingArg; }
	void setPaddingArg(Node *paddingArg) {
		this->frontPaddingArg = this->backPaddingArg = paddingArg;
	}
	void setPaddingArg(Node *paddingArgFront, Node *paddingArgBack) {
		this->frontPaddingArg = paddingArgFront;
		this->backPaddingArg = paddingArgBack;
	}
	void setDimensionNo(int dimensionNo) { this->dimensionNo = dimensionNo; }
	int getDimensionNo() { return dimensionNo; }
	Node *getDividingArg() { return dividingArg; }
	bool hasPadding() { return (frontPaddingArg != NULL || backPaddingArg != NULL); }		
	Node *getFrontPaddingArg() { return frontPaddingArg; }
	Node *getBackPaddingArg() { return backPaddingArg; }
	static const char *getArgumentString(Node *arg, const char *prefix);
}; 

/*	This class embodies the detail of application of a single partition functions on a data structure.
	It holds the user specified partitioning arguments, name of the partition functions, and interfaces
	for determining how application of the underlying partition function affects the data structure. 
	Specific partition functions, i.e., block-count or block-size functions implements its interfaces
	to do their work. Our broad plan/goal is to allow user defined partition functions in the future.
	At that time further enrichment of this class might become necessary.
*/
class PartitionFunctionConfig {
  protected:
	yyltype *location;
	const char *functionName;
	List<DataDimensionConfig*> *arguments;
	PartitionOrder partitionOrder;
	PartitionFunctionConfig(yyltype *location, const char *functionName);		
	virtual void processArguments(List<PartitionArg*> *dividingArgs, List<PartitionArg*> *paddingArgs);
  public:
	static PartitionFunctionConfig *generateConfig(yyltype *location, 
			const char *functionName, 
			List<PartitionArg*> *dividingArgs, 
			List<PartitionArg*> *paddingArgs);
	virtual void setDimensionIds(List<int> *dimensionIds);
	virtual List<int> *getBlockedDimensions(Type *structureType) { return new List<int>; }
	virtual List<int> *getPartitionedDimensions();
	virtual int getDimensionality() { return arguments->NumElements(); }
	void setPartitionOrder(PartitionOrder partitionOrder) { this->partitionOrder = partitionOrder; }
	bool isOrdered() { return (partitionOrder == AscendingOrder || partitionOrder == DescendingOrder); }
	bool hasOverlappingsAmongPartitions();
	List<int> *getOverlappingPartitionDims();
	DataDimensionConfig *getArgsForDimension(int dimensionNo);
	const char *getName() { return functionName; }

	// function used to determine if padding arguments are applicable for the partition-function under concern
	virtual bool doesSupportGhostRegion() { return false; }
	// this function is used to determine if we need to transform/reverse-transform indexes that are generated
	// by traversing the partitions created by applying this partition function
	virtual bool doesReorderStoredData() { return false; }
	
	// following three functions are used to get index transformation or re-transformation expression and the
	// specialized index inclusion check expression for functions that reorder the indexes of the dimension they
	// partition. These expressions are used during code generation. These functions take only the name of the
	// index to be transformed/reverse-transformed and generate the desired expressions assuming that any part-
	// ition function parameter used to configure this LPS is available as an attribute of a partition object 
	// and partition count and index correspond to the LPU where the test is been made is available in a part-
	// config object. The third argument, copyMode, indicates if data copying or reference reassignment is been
	// used for the storage of underlying data. This is needed as transformation processes for data-copy and ref-
	// erence reassignment are different. 
	virtual const char *getTransformedIndex(int dimensionNo, 
			const char *origIndexName, bool copyMode) { return NULL; }
	virtual const char *getOriginalIndex(int dimensionNo, 
			const char *xformIndexName, bool copyMode) { return NULL; }
	virtual const char *getInclusionTestExpr(int dimensionNo, 
			const char *origIndexName, bool copyMode) { return NULL; }
	
	// this function is used to transform any integer unrelated to an array index of a dimension based on the 
	// partition function used in that dimension for that array. Here imprecise upper/lower bound is calculated 
	// as we do not know if the compared integer falls within the LPU boundary, nonetheless we can optimize code 
	// using an imprecise bound. For example, if the partition function is block-stride then we can use the 
	// imprecise transformation to skips some of the strides of the current LPU during loop iterations. This 
	// function, like its three predecessors, is only applicable for data reordering partitioning functions. 	
	virtual const char *getImpreciseBoundOnXformedIndex(int dimensionNo, 
			const char *origIndexName, 
			bool lowerBound, bool copyMode, int indent) { return NULL; }

	// A dimension configuration object is created for each dimension of a data structure at runtime to decide
	// about the shape of each data part. The configuration object been instantiated depends on the partition
	// function been used to divide the dimension of the structure. This function returns the name of the dim-
	// configuration object class to make this logic works. Note that this implies that each partition function
	// has a corresponding dimension configuration class. That class should be added in the part_generation.h
	// library file of memory_management module.
	virtual const char *getDimensionConfigClassName() { return NULL; }
};

class DataStructure {
  protected:
	VariableDef *definition;
	DataStructure *source;
	List<DataStructure*> *dependents;
	Space *space;

	// a variable used for static analysis to make decision about the memory allocation requirement for the data 
	// structure under concern
	LPSVarUsageStat *usageStat;

	// This variable is only useful for subpartitioned data structures. If it true, it signifies that
	// the immediate parent space of the subpartition space, i.e., the space some of whose data 
	// structures are subpartitioned, cannot store the content of this data structure due to space
	// limitation. Therefore, any update must be moved further up in the partition hierarchy if persistence
	// is desired.	
	bool nonStorable;

	// The LPS that allocates a data structure is not necessarily always the primary source. Not all LPSes 
	// allocate their data structures either. Therefore a reference is needed to point to the LPS allocation 
	// the this data structure reference should point to. These pointers are then used to set up variables in
	// LPUs properly    
	Space *allocator;

	// When true, this special flag indicates that no new data parts need to be maintained in the host for this
	// data structure for computation done on it inside GPUs -- data parts of the host can be used directly in
	// GPU CUDA kernel.	
	bool excludedFromGpuAllocation;

	// denotes the maximum number of versions to be maintained for this data structure if it is epoch dependent
	// note that version count starts from 0 not 1
	int versionCount;
  public:
	DataStructure(VariableDef *definition);
	DataStructure(DataStructure *source);
	virtual ~DataStructure() {};
	const char *getName();
	void setSpaceReference(Space *space);
	Space *getSpace() { return space; }
	DataStructure *getSource() { return source; }
	Type *getType();
	void flagAsNonStorable() { nonStorable = true; }
	bool isNonStorable() { return nonStorable; }
	virtual bool hasOverlappingsAmongPartitions() { return false; }
	LPSVarUsageStat *getUsageStat() { return usageStat; }
	void setAllocator(Space *allocator) { this->allocator = allocator; }
	Space *getAllocator() { return allocator; }
	void excludeFromGpuAllocation() { this->excludedFromGpuAllocation = true; }
	bool isExcludedFromGpuAllocation() { return excludedFromGpuAllocation; }
	
	// returns current structure if it has been marked allocated or gets the closest source reference that is
	// marked allocated  
	DataStructure *getClosestAllocation();

	// This function is used to determine if configurations of a single data structure in two different LPSes
	// have been indictated to use the same memory allocation. If the answer is YES then no communication is 
	// needed to synchronize the LPSes regarding the concerned data structure; otherwise, there need to be a 
	// data transfer of some form when there is a transition between these two LPSes and in between the structure 
	// has been updated by some computation stage.
	bool useSameAllocation(DataStructure *other);
	
	// functions for determining epoch dependencies
	void updateVersionCount(int version);
	int getLocalVersionCount() { return versionCount; }
	int getVersionCount();

	// returns the root reference of this data structure
	DataStructure *getPrimarySource();
};

class ArrayDataStructure : public DataStructure {
  protected:
	List<int> *sourceDimensions; 		// indicates the dimensions of the data structure available for 
				     		// partitioning within the space under concern
	List<PartitionFunctionConfig*> *partitionSpecs;
	List<int> *afterPartitionDimensions; 	// indicates the dimension remain available for partitioning
					     	// by subsequent spaces from each partition of the data structure
					     	// created by current space
  public:
	ArrayDataStructure(VariableDef *definition);
	ArrayDataStructure(ArrayDataStructure *source);
	bool isOrderDependent();
	// note that dimension IDs start from 1 instead of from 0
	void setSourceDimensions(List<int> *sourceDimensions) { this->sourceDimensions = sourceDimensions; }
	void setAfterPartitionDimensions(List<int> *afterPartitionDimensions) {
		this->afterPartitionDimensions = afterPartitionDimensions;
	}
	List<int> *getRemainingDimensions() { return afterPartitionDimensions; }
	void addPartitionSpec(PartitionFunctionConfig *partitionConfig);	
	bool hasOverlappingsAmongPartitions();
	List<int> *getOverlappingPartitionDims();
	PartitionFunctionConfig *getPartitionSpecForDimension(int dimensionNo);	
	bool isPartitioned() { return partitionSpecs != NULL && partitionSpecs->NumElements() > 0; }
	bool isPartitionedAlongDimension(int dimensionNo);
	bool isPartitionedEarlier();
	bool isPartitionedAlongDimensionEarlier(int dimensionNo);
	int getDimensionality();

	// This function is primarily needed for file I/O to determines if data parts generated for the array
	// have overlappings of regions due to padding in anywhere in their partition hierarchy. The padding
	// regions are generally skipped during file writes.
	bool doesGenerateOverlappingParts();

	// Since some partition functions results in reordering array dimensions if data is been copied down
	// to ensure that each partition work over a sequential block of memory, during code generation we
	// need to know if any indexing on an array needs to be transformed from original to reordered index
	// and vice versa. Therefore this two methods have been provided to aid such decision making. The
	// former check for potential reordering starting from current LPS to any upper LPS up to the bound
	// specified. The latter check for only local reordering.
	bool isDimensionReordered(int dimensionNo, Space *comparisonBound);
	bool isDimensionLocallyReordered(int dimensionNo);
	// Following two functions do the same thing as the preceeding two do -- only these functions take in
	// into account all dimensions of the array
	bool isReordered(Space *comparisonBound);
	bool isLocallyReordered();
	// This function is needed to determine if an array needs to be allocated again. The logic is if there
	// is a reordering of the array after last allocation then the variable should be allocated again. 
	bool isReorderedAfter(Space *allocatorSpace);
	// This is used to determine if we need to have a for loop to iterate over the entries along a given
	// dimension of the array
	bool isSingleEntryInDimension(int dimensionNo);
	// This is a print function to display how different dimensions of the array are partitioned in this 
	// LPS under concern. This function is purely for diagnostic purpose and has not been perfected.
	void print();

	//---------------------------------------------------------------------- Host code generation helper routines
	
	// These are four functions used during code generation correspond to array dimensions that have been
	// reordered by underlying partition functions. The transformation codes for reordered data been copied 
	// into a new memory location for current LPS and for only LPU definition generating reordered indexes 
	// for unordered data are different.  
	const char *getIndexXfromExpr(int dimensionNo, const char *indexName);
	const char *getReorderedInclusionCheckExpr(int dimensionNo, const char *indexName);
	const char *getReverseXformExpr(int dimensionNo, const char *xformIndex);
	const char *getImpreciseBoundOnXformedIndex(int dimensionNo, 
			const char *indexName, bool lowerBound, int indent);
};

/*	Token, Coordinate, and CoordinateSystem classes implement a mechanism for associating dimensions
	of data structures to the dimensions of corresponding spaces. Comparing to old Fortran-D terminology,
	this is the mechanism of storing data structure alignment information in IT. We view individual 
	spaces as a coordinate system of dimensionality as specified in the partition block. Then dimensions
	of individual data structures within it somehow must be mapped to the coordinate dimensions. If some
	data structure is replicated, it should have a wildcard token in along some dimensions.

	This representations gives a good way to visualize and think about the partitioning scheme. It also
	makes validation of partition specification easy. That boils down to enuring that all data structures
	have one-and-only-one token per coordinate dimension. 
*/
class Token {
  protected:
	int dimensionId;
	DataStructure *data;
  public:
 	static int wildcardTokenId;
	Token(DataStructure *data, int dimensionId) {
		this->data = data; 
		this->dimensionId = dimensionId; 
	}
	bool isWildcard() { return dimensionId == wildcardTokenId; }
	DataStructure *getData() { return data; }
	int getDimensionId() { return dimensionId; }
};

class Coordinate {
  protected:
	int id;	
	List<Token*> *tokenList;
  public:
	Coordinate(int id) {
		this->id = id; 
		tokenList = new List<Token*>; 
	}
	void storeToken(Token *token) { tokenList->Append(token); }
	int getTokenCount() { return tokenList->NumElements(); }
	Token *getTokenForDataStructure(const char *dataStructureName);
	List<Token*> *getTokenList() { return tokenList; }	
};

class CoordinateSystem {
  protected:
	int dimensionCount;
	List<Coordinate*> *dimensions;
  public:
	CoordinateSystem(int dimensionCount);
	bool isBalanced();
	Coordinate *getCoordinate(int dimensionNo);		
};

/* 	the class correspond to a logical processing space (LPS) 
*/
class Space {
  protected:
	const char *id;
	int dimensions;
	bool dynamic;
	bool subpartitionSpace;
	CoordinateSystem *coordSys;
	Space *parent;
	Space *subpartition;
	Hashtable<DataStructure*> *dataStructureList;
	List<Space*> *children;
	
	// This is a variable used for the backend compiler. It represents the position in the PCubeS hierarchy
	// the current LPS has been mapped to. PPSes are labeled from 1 to up from the lowest physical space to
	// the highest 
	int ppsId;
	
	// This is, again, used by the backend compiler. It is needed to determine what data dependency should 
	// be resolved by moving data around (across segments or within a single segment) and what should be 
	// resolved by signaling-and-waiting on some synchronization primitives. 
	int segmentedPPS;

	// a flag indicating if the LPS has some compution stages executing in it; this information is needed to
	// determine whether or not to generate LPUs for this LPS
	bool executesCode;
  public:
	static const char *RootSpaceName;
	static const char *SubSpaceSuffix;
	Space(const char *name, int dimensions, bool dynamic, bool subpartitionSpace);
	void setStructureList(Hashtable<DataStructure*> *dataStructureList);
	void initEmptyStructureList();
	void addDataStructure(DataStructure *structure);
	DataStructure *getStructure(const char *name);
	bool isInSpace(const char *structName);
	DataStructure *getLocalStructure(const char *name);
	void setParent(Space *parent) { this->parent = parent; }
	Space *getParent() { return parent; }
	void addChildSpace(Space *child) { children->Append(child); }
	List<Space*> *getChildrenSpaces() { return children; }
	int getDimensionCount() { return dimensions; }
	const char *getName() { return id; }
	void storeToken(int coordinate, Token *token);
	bool isValidCoordinateSystem() { return (coordSys == NULL || coordSys->isBalanced()); }	
	bool isParentSpace(Space *suspectedParent);
	Space *getClosestSubpartitionRoot();
	bool isDynamic() { return dynamic; }
	List<const char*> *getLocallyUsedArrayNames();
	List<const char*> *getLocalDataStructureNames();
	bool isReplicatedInCurrentSpace(const char *dataStructureName);
	bool isReplicated(const char *dataStructureName);
	bool isSubpartitionSpace() { return subpartitionSpace; }
	static List<Space*> *getConnetingSpaceSequenceForSpacePair(Space *first, Space *last);
	List<const char*> *getLocalDataStructuresWithOverlappedPartitions();
	List<const char*> *getNonStorableDataStructures();
	CoordinateSystem *getCoordinateSystem() { return coordSys; }
	void setSubpartition(Space *subpartition) { this->subpartition = subpartition; }
	Space *getSubpartition() { return subpartition; }
	bool isRoot() { return parent == NULL; }
	Space *getRoot() { return (parent == NULL) ? this : parent->getRoot(); }
	Symbol *getLpuIdSymbol();
	void setPpsId(int ppsId) { this->ppsId = ppsId; }
	int getPpsId() { return ppsId; }
	void setSegmentedPPS(int segmentedPPS) { this->segmentedPPS = segmentedPPS; }
	int getSegmentedPPS() { return segmentedPPS; }
	void flagToExecuteCode() { executesCode = true; }
	bool doesExecuteCode() { return executesCode; }

	// given an array dimension, this function returns the coordinate dimension the array dimension has been
	// aligned to
	int getAlignmentDimensionForArray(int arrayDimension, const char *arrayName);

	//---------------------------------------------------------------------- Host code generation helper routines
	
	// a helper routine for code generation that determines if any of the structures listed in the partition
	// configuration of the LPS referred by this instance needs to be allocated a memory
	bool allocateStructures();
	// does the checking above for a specific data structure
	bool allocateStructure(const char *structureName);

	//----------------------------------------------------------------------- GPU code generation helper routines

	// This helper routine is used to calculate the LPU counts along different dimension when an LPS entry
	// happens inside a GPU kernel. The second argument tells what array variables can be used to calculate
	// the LPU counts. This is important as not all kernel uses all arrays of an LPS. 
	void genLpuCountInfoForGpuKernelExpansion(std::ofstream &stream, 
			const char *indentStr,
			List<const char*> *accessibleArrays, 
			bool perWarpCountInfo);

	// This helper routine is used to construct updated partition dimension information for relevant arrays 
	// when the GPU kernel code enters into a new LPS as part of its compution 
	void genArrayDimInfoForGpuKernelExpansion(std::ofstream &stream,
			const char *indentStr,
			List<const char*> *arrayNames,
			int topmostGpuPps,
			bool perWarpCountInfo,
			bool perWarpDimensionInfo);
};

/*	The entire partition block is seen as a hierarchy of coordinate systems of spaces. The hierarchy is 
	rooted in the RootSpace that holds all task-global data structures. Below it a tree is formed based on 
	the partial ordering visible in the user's specification for the partition block.
*/
class PartitionHierarchy {
  protected:
	PCubeSModel *pcubesModel;
	Hashtable<Space*> *spaceHierarchy;
  public:
	PartitionHierarchy();
	void setPCubeSModel(PCubeSModel *pcubesModel);
	PCubeSModel *getPCubeSModel();
	bool isMappedToHybridModel();
	Space *getSpace(char spaceId);
	Space *getSubspace(char spaceId);
	Space *getRootSpace();
	bool addNewSpace(Space *space);
	Space *getCommonAncestor(Space *space1, Space *space2);	
	
	// This routine sets up allocator LPS references to all data structures of different LPSes in the 
	// partition hierarchy so that memory allocations can be done appropriately and also structure references 
	// can be set accordingly during LPU generations. The analysis done here is dependent on the LPS-to-PPS 
	// mapping. Therefore, it should be done after ppsIds are set to LPSes properly. 	
	void performAllocationAnalysis(int segmentedPPS);
  private:
	// We try to keep the environment inside the GPU cards to look more like the multicore memory model as 
	// opposed to the segmented memory model. So the data parts are allocated for the the top-most LPS in the
	// partition hierarchy that is mapped to a GPU PPS as opposed to for the LPS that hosts the computation
	// stages. Furthermore, we ignore reordering of array indices in the LPS hierarchy after the allocation for
	// the top-most GPU LPS. This choice has been made to avoid developing intra-GPU communicator for data part
	// synchronization which was a major work for the segmented memory backend. We could use the same logic for
	// communicators for the GPU as we have for the host and cluster level communicator, but our initial assump-
	// tion is that the overhead of communicator operations will be significant. Furthermore, we wanted to 
	// maximize the amount of time we spent on GPU computation in-between data stage in-out from the host and
	// minimize the number of times we do the transfaer. In that regard, having larger data part allocations 
	// for higher level LPSes often helps. 
	// The following function, implements the deviant behavior for GPU LPSes regarding memory allocation as a
	// supporting function to the performAllocationAnalysis() routine above.   
	void performAllocAnalysisOnLpsMappedToGpu(Space *lps, int gpuPpsId);
};

#endif
