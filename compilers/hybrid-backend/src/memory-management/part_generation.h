#ifndef _H_part_generation
#define _H_part_generation

/* This header file embodies classes that are needed to generate data part descriptions for LPUs of different
   LPSes multiplexed to the current PPU. There are overlappings in functionalities of some of the libraries 
   of this header with some in the ../partition-lib directory. This is because we already had those libraries
   for multicore backends and that were not sufficient for segmented memory environments.

   The appropriate use of the library interface presented by this header file is to generate code that will
   instantiate appropriate partition configuration classes, based on the partition section of the source code, 
   at the beginning of a task launch. After the task determines the sizes of different data structure it and
   know what LPUs have been allocated to it then it invokes library methods in configuration instances to 
   generate all the unique data parts corresponding to its LPUs.  
*/

#include <vector>	

#include "../utils/list.h"
#include "../utils/utility.h"
#include "../runtime/structure.h"
#include "../partition-lib/partition.h"

class DimConfig;
class PartMetadata;
class ListMetadata;
class DataPartsList;
class DataItemConfig;

/* This is just a helper class to hold the results of different configuration functions of Dim Parition Config 
   class as needed for data parts construction. 
*/
class DimensionMetadata {
  public:
	DimensionMetadata() {
		paddings[0] = 0;
		paddings[1] = 0;
	}	
	Dimension partDimension;
	int paddings[2];
};

/* This is a helper class used in determining if two data partition configurations generate the same data parts. 
   It stands for an interval description pattern different parts genenated by the partition instruction for a 
   single dimension adhere to. The goal is to estimate if the parts are the same by comparing just the pattern
   instead of comparing individual data parts. Note that a partition instruction may cause a dimension to be 
   partitionned into parts with different patterns. In that case, all those patterns need to be considered during
   partition configurations comparison. Further note that the pattern comparison process must be hierarchical 
   due to the presence     
*/
class PartIntervalPattern {
  public:
	// since each part starts at a different index, the beginning should be specified as an expression; the
	// suggested pattern is 'partId * multiplier + offset' A partition instruction is supposed to evaluate its 
	// runtime arguments for the partition and padding parameters and setup the numerical values for the
	// multiplier and the offset so that patterns comparison can be done effectively   
	const char *beginExpr;
	int count;
	int period;
	int length;
	// Dometimes a partition instruction may not divide the dimension evenly among the generated parts. Then
	// it may create some parts having some overflow indices from the trailing end of the dimension. The 
	// following field retains the overflow amount. Note that a new pattern should be generated for each
	// distinct overflow amount  
	int overflow;
	// the number of times the pattern occurs in the generated parts
	int instances;
  public:
	PartIntervalPattern();
	// a second constructor for convenience that sets every field execpt the begin expression
	PartIntervalPattern(int c, int p, int l, int o, int i);
	bool isEqual(PartIntervalPattern *other);
	bool isEqualIgnoringInstanceCount(PartIntervalPattern *other);
	int getPartLength();	
};

/* Superclass to represent the partition configuration applied for a particular dimension of an array for an LPS. 
   Note that this class has a parent reference. This is needed as partitioning in IT is hierarchical and the 
   properties of a lower LPS data part often cannot be found without investigating its higher LPS ancestor parts. 
   For the same reasons most of the public method of this class takes a list of part Ids as input as oppose to a 
   single part id. Here each item in the list identifies a part in some LPS that is in the path that leads to the 
   data part of the LPS under concern.  
*/
class DimPartitionConfig {
  protected:
	// span of the dimension of the underlying array
	Dimension dataDimension;
	// any arguments needed by the partition function applied on this dimension
	int *partitionArgs;
	// for any partition there may be a padding at the beginning and a padding at the end 
	int paddings[2];
	// the number of physical processors that are responsible for managing LPUs of the concerned LPS 
	int ppuCount;
	// represents which dimension of the LPS current dimension of the array is aligned to 
	int lpsAlignment;
	// represent the ancestor config if the current instance is dividing an already divided dimension
	DimPartitionConfig *parentConfig;

	// a recursive helper routine to get the part of the dimension that been subject to partitioning by the 
	// current dimension configuration instance 
	Dimension getDimensionFromParent(List<int> *partIdList, int position);
	
	// these two functions determine the actual amount of padding been applied to a particular part; the 
	// effective padding may differ from the configuration for parts at the boundary 
	virtual int getEffectiveFrontPadding(int partId, Dimension parentDimension) { return paddings[0]; }		
	virtual int getEffectiveRearPadding(int partId, Dimension parentDimension) {  return paddings[1]; }

  public:
	DimPartitionConfig(Dimension dimension, int *partitionArgs, 
			int paddings[2], int ppuCount, int lpsAlignment);
	// constructor to be overriden by functions that do not support padding
	DimPartitionConfig(Dimension dimension, int *partitionArgs, int ppuCount, int lpsAlignment);
	virtual ~DimPartitionConfig() {}

	bool hasPadding() { return paddings[0] > 0 || paddings[1] > 0; }
	Dimension getDataDimension() { return dataDimension; }
	void setParentConfig(DimPartitionConfig *parentConfig) { this->parentConfig = parentConfig; }
	int getLpsAlignment() { return lpsAlignment; }

	// retrieves the dimension index, or part-Id, from the lpuId for current LPS
	virtual int pickPartId(int *lpuId) { return lpuId[lpsAlignment]; }
	// similar to the above, this function retrieves the parts count along a dimension from lpu counts 
	virtual int pickPartCount(int *lpuCount) { return lpuCount[lpsAlignment]; }

	// this function makes a conservative estimate of what may be the largest part dimension length given the
	// length of the parent dimension 
	int getLargestPartDimLength(int parentDimLength);

	// an accumulator function that returns results of different dimension configuration utilies above 
	DimensionMetadata *generateDimMetadata(List<int> *partIdList);

	// function to be used by data partition config to generate dimension metadata about a structure from just 
	// its hierarchical part Id
	Dimension getPartDimension(List<int> *partIdList);

	// This is a recursive process for determining how original data dimension has been repeatedly divided
	// into smaller dimensions along the partition hierarchy of a task to reach a particular data part 
	// identified by the last argument. Along the way this process also calculate the number of partitions
	// in each point of divisioning. This function has been added to support file IO operation that needs a 
	// data part index to file location transformation.
	void getHierarchicalDimensionAndPartCountInfo(List<Dimension*> *dimensionList, 
			List<int> *partCountsList, 
			int position, List<int> *partIdList);

	// Similar to the above, this function has been added to aid file I/O. It returns the original index of
	// a data point along a dimension that might be reordered one or more time due to the use of reordering
	// partition functions. The process of getting back the original index from a transformed one works 
	// recursively backward from lower parts to upper parts.
	virtual int getOriginalIndex(int partIndex, int position, List<int> *partIdList, 
			List<int> *partCountList, 
			List<Dimension*> *partDimensionList);
	
	// determines the part dimension given a part Id; for reordering partition function it is the part 
	// dimension after the indexes have been shuffled to have each part occupying a contiguous chunk along 
	// the dimension line
	virtual Dimension getPartDimension(int partId, Dimension parentDimension) = 0;
	
	// determines how many unique parts of the data structure can be found along this dimension by
	// dividing the parent dimension 
	virtual int getPartsCount(Dimension parentDimension) = 0;

	// The DimPartitionConfig and its subclasses are designed state-free. So is the DataPartitionConfig 
	// class that holds instances of these classes to specify the partition construction of a data structure
	// for a particular LPS. For calculations related to communication, however, we need stateful versions
	// of these classes and for the hierarchical agregate DataPartitionConfig class. This function is 
	// provided to aid in the stateful DataItemConfig, the counterpart of DataPartitionConfig, creation. 
	virtual PartitionInstr *getPartitionInstr() = 0;

	// This function tells if a partition configuration does not divide the dimension it is specified to
	// operate on. For example, replication-config is by default like that. Sometimes, some other partition
	// functions may be configured in a similar way. In particular, if there are multiple tasks in the 
	// program and the user wants to match partition configurations for different tasks for a shared data
	// structure to reduce data movements due to task transitions
	virtual bool isDegenerativeCase() = 0;

	// this tells if a partition configuration reorder the indices of the dimension it operate over
	virtual bool doesReorderIndices() = 0;
	
	// this tells if two instances of dimension partition configurations are the same in all respects
	virtual bool isEqual(DimPartitionConfig *otherConfig) = 0;

	// given a dimension to partition, the subclasses should return all the different interval patterns
	// for the dimension parts they generate from it; the default base-class implementation should be used
	// when the number of parts is just 1 to ensure that there is no discrepancy in the terminal case pattern 
	virtual List<PartIntervalPattern*> *getPartIntervalPatterns(Dimension origDimension);
};

/* configuration subclass to be instantiated when a data dimension has not been divided within the LPS */
class ReplicationConfig : public DimPartitionConfig {
  protected:
	int getEffectiveFrontPadding(int partId, Dimension parentDimension) { return 0; } 		
	int getEffectiveRearPadding(int partId, Dimension parentDimension) { return 0;  }		
  public:
	ReplicationConfig(Dimension dimension) : DimPartitionConfig(dimension, NULL, 0, -1) {}
	
	int pickPartId(int *lpsId) { return 0; } 		
	int pickPartCount(int *lpuCount) { return 1; }
	int getPartsCount(Dimension parentDimension) { return 1; }
	Dimension getPartDimension(int partId, Dimension parentDimension) { return parentDimension; }		
	PartitionInstr *getPartitionInstr() {
        	PartitionInstr *instr = new VoidInstr();
        	Assert(instr != NULL);
        	return instr;
	}
	bool isDegenerativeCase() { return true; }
	bool doesReorderIndices() { return false; }
	bool isEqual(DimPartitionConfig *otherConfig) {
		return dynamic_cast<ReplicationConfig*>(otherConfig) != NULL;
	}
};

/* configuration subclass corresponding to 'block_size' partition function; it takes a 'size' parameter */
class BlockSizeConfig : public DimPartitionConfig {
  protected:
	int getEffectiveFrontPadding(int partId, Dimension parentDimension); 		
	int getEffectiveRearPadding(int partId,  Dimension parentDimension);
  public:
	BlockSizeConfig(Dimension dimension, int *partitionArgs, int paddings[2], 
			int ppuCount, int lpsAlignment) : DimPartitionConfig(dimension, 
			partitionArgs, paddings, ppuCount, lpsAlignment) {}
	
	int getPartsCount(Dimension parentDimension);
	int getLargestPartDimLength(int parentDimLength);
	Dimension getPartDimension(int partId, Dimension parentDimension);
	PartitionInstr *getPartitionInstr();
	bool isDegenerativeCase() { return false; }
	bool doesReorderIndices() { return false; }
	bool isEqual(DimPartitionConfig *otherConfig);
	List<PartIntervalPattern*> *getPartIntervalPatterns(Dimension origDimension);
};

/* configuration subclass corresponding to 'block_count' partition function; it takes a 'count' parameter */
class BlockCountConfig : public DimPartitionConfig {
  protected:
	int getEffectiveFrontPadding(int partId, Dimension parentDimension); 		
	int getEffectiveRearPadding(int partId,  Dimension parentDImension);
  public:
	BlockCountConfig(Dimension dimension, int *partitionArgs, int paddings[2], 
			int ppuCount, int lpsAlignment) : DimPartitionConfig(dimension, 
			partitionArgs, paddings, ppuCount, lpsAlignment) {}
	
	int getPartsCount(Dimension parentDimension);
	int getLargestPartDimLength(int parentDimLength);
	Dimension getPartDimension(int partId, Dimension parentDimension);
	PartitionInstr *getPartitionInstr();
	bool isDegenerativeCase() { return partitionArgs[0] == 1; }
	bool doesReorderIndices() { return false; }
	bool isEqual(DimPartitionConfig *otherConfig);
	List<PartIntervalPattern*> *getPartIntervalPatterns(Dimension origDimension);
};

/* configuration subclass for parameter-less 'stride' partition function */
class StrideConfig : public DimPartitionConfig {
  public:
	StrideConfig(Dimension dimension, int ppuCount, int lpsAlignment) 
			: DimPartitionConfig(dimension, NULL, ppuCount, lpsAlignment) {}
	
	int getPartsCount(Dimension parentDimension);
	int getLargestPartDimLength(int parentDimLength);
	Dimension getPartDimension(int partId, Dimension parentDimension);
	int getOriginalIndex(int partIndex, int position, List<int> *partIdList, 
			List<int> *partCountList, 
			List<Dimension*> *partDimensionList);
	PartitionInstr *getPartitionInstr() { 
		PartitionInstr *instr = new StrideInstr(ppuCount);
		Assert(instr != NULL);
		return instr; 
	}
	bool isDegenerativeCase() { return ppuCount == 1; }
	bool doesReorderIndices() { return true; }
	bool isEqual(DimPartitionConfig *otherConfig);
	List<PartIntervalPattern*> *getPartIntervalPatterns(Dimension origDimension);
};

/* configuration subclass for 'block_stride' partition function that takes a 'block_size' parameter */
class BlockStrideConfig : public DimPartitionConfig {
  public:
	BlockStrideConfig(Dimension dimension, int *partitionArgs, 
			int ppuCount, int lpsAlignment) : DimPartitionConfig(dimension, 
			partitionArgs, ppuCount, lpsAlignment) {}
	
	int getPartsCount(Dimension parentDimension);
	int getLargestPartDimLength(int parentDimLength);
	Dimension getPartDimension(int partId, Dimension parentDimension);
	int getOriginalIndex(int partIndex, int position, List<int> *partIdList, 
			List<int> *partCountList, 
			List<Dimension*> *partDimensionList);
	PartitionInstr *getPartitionInstr();
	bool isDegenerativeCase() { return ppuCount == 1; }
	bool doesReorderIndices() { return true; }
	bool isEqual(DimPartitionConfig *otherConfig);
	List<PartIntervalPattern*> *getPartIntervalPatterns(Dimension origDimension);
};

/* This is the class that holds the partition configuration for different dimensions of a single data structure 
   within an LPS. An instance of this class needs to be instantiated based on the instructions about partitions 
   and once the length of different dimensions are known. Know that will be one dimPartitionConfig instance per 
   dimension of the structure. That is the configuration is independent of the dimensionality of the LPS itself. 
   Any dimension not been partitioned by the LPS should have a replication-config for it. This strategy enables 
   us to readily compare parts of one LPS with parts of other LPSes as the dimensionality of the parts remain 
   the same regardless of the dimensionality of the LPSes they are mean for. 
*/
class DataPartitionConfig {
  protected:
	// the number of dimension in the data structure
	int dimensionCount;
	// partition configuration along each of those dimensions
	List<DimPartitionConfig*> *dimensionConfigs;
	// A reference to the parition configuration of immediate ancestor LPS cantaining the underlying data 
	// structure
	DataPartitionConfig *parent;
	// In the partition hierarchy the parent part specification for a data structure not necessarily recides 
	// in the parent of the current LPS. Rather the parent part specification may lies further up in the 
	// ancestry. This need to be taken into account when generating part Ids from LPU ids. The following 
	// property keeps track of the number of parent links for an LPU needs to be skipped to get to the next 
	// level. TODO we should try to make the part Id calculation process more robust in the future.
	int parentJump;
	// a configuration instance that is needed to construct the part-container (check part_tracking.h)
	// instance for the underlying data structure
	std::vector<DimConfig> *dimensionOrder;
	// an integer identifier for the LPS the data structure partition is specified for
	int lpsId;
  public:
	DataPartitionConfig(int dimensionCount, List<DimPartitionConfig*> *dimensionConfigs);
	~DataPartitionConfig();
	void setParent(DataPartitionConfig *parent, int parentJump);
	DimPartitionConfig *getDimensionConfig(int dimNo) { return dimensionConfigs->Nth(dimNo); }
	void configureDimensionOrder();
	std::vector<DimConfig> *getDimensionOrder();
	void setLpsId(int lpsId) { this->lpsId = lpsId; }
	int getLpsId() { return lpsId; }
	
	// the function is to be used for generating metadata for subsequent use in generating a data part
	PartMetadata *generatePartMetadata(List<int*> *partIdList);
	// generates a data part Id for an LPU from the LPU Id
	List<int*> *generatePartId(List<int*> *lpuIds);
	// updates fields of an existing part Id to create the new part Id
	void generatePartId(List<int*> *lpuIds, List<int*> *partId);
	// returns the number of elements there should be in a valid part Id constructed for a data part
	int getPartIdLevels();
	// generate a blank part-Id template to be used over and over where applicable
	List<int*> *generatePartIdTemplate();

	// function to set up the metadata regarding the partition dimensions of the underlying object 
	// within an LPU
	void updatePartDimensionInfo(List<int*> *partIdList, int *lpuCounts, PartDimension *partDimension);
	// an alternative version for calculating part dimension information of a data part by drawing out
	// information from a parent part; this should be more efficient than the function above as it will avoid 
	// some recursions
	void updatePartDimensionInfo(int *lpuId, int *lpuCounts, 
			PartDimension *partDims, PartDimension *parentPartDims);

	// This function has been added to aid GPU SM memory allocation decision. Given the lengths along different
	// dimensions of a parent data part, it determines the dimension lengths for the largest subpart. Note that
	// this function makes a conservative estimate -- not the exact calculation -- of the subpart's dimensions.
	// Further, it is an in-place update mechanism where parent part's information is replaced by the subpart's
	// information.  
	void getLargestPartDimLengths(int *parentDimLengths);

	// function to be used to determine how many LPUs should be there within an LPS at runtime; when
	// the second argument is NULL, data-dimension should be used to determine the parts count
	int getPartsCountAlongDimension(int dimensionNo, Dimension *parentDimension = NULL);

	// According to the chosen strategy the data part for an LPU for an LPS may refer to a portion of a larger 
	// part from an ancestor LPS. Therefore, at runtime the identity of that ancestor part needs to be 
	// calculated as opposed to the smaller subpart the LPU refers to. This method provides that functionality. 
	// Here the second parameter indicates how many lpuIds should be skipped from the rear to reach the desired 
	// ancestor. The third parameter is used when an already allocated list for the part ID should be updated. 
	// If it is null then the function creates a new part ID list and returns it. 
	List<int*> *generateSuperPartId(List<int*> *lpuIds, int backsteps, List<int*> *idTemplate = NULL);  

	// function to generate the list of data parts (see allocation.h) from the partition configuration; note 
	// that the data parts list returned by this function is unusable until memory allocations has been done 
	DataPartsList *generatePartList(int epochCount);

	// this function is used to determine the data-parts content of PPUs other than the current one so that 
	// decision about the nature and content of communication for shared data can be made.
	ListMetadata *generatePartListMetadata(List<List<int*>*> *partIds);

	// Stateful versions of data-partition-configuration are needed by the communication libraries; To
	// give an example why is that, consider a synchronization of ghost boundary regions of data parts.
	// Then the sender parts will have their paddings disabled but not the receiver parts, despite the whole 
	// synchronization being taken place within the sphere/confinement of a single LPS. In that case, we 
	// will have two DataItemConfig instances, one for the sender and the other for the receiver, and 
	// activate paddings on the second before we use them to calculate data movement requirements.
	DataItemConfig *generateStateFulVersion();

	// When running a multi-tasked program in a segmented-memory architecture, at the point of transition
	// from one task to a subsequent task, we need to determine what kind of data parts rearrangements
	// and data communications may be needed to prepare the program environment for the upcoming task.
	// The ideal situation is the data parts arrangement left by completed tasks is readily usable in the
	// future task, but that might not be the case at a particular situation. Furthermore, even if the
	// already available data parts can be used to serve the needs of future tasks, determining if they
	// are adequate needs an elaborate machanism for comparting the actual parts description. The process
	// may be time consuming and our investigation suggests that it may also have signification memory 
	// footprint. We choose to implement a much faster decision making process instead that may miss some 
	// cases and result in not-strictly-necessary communications and/or data-rearrangements. The chosen
	// process compares the partition hierarchies of interacting tasks sharing the same data structure.
	// The process goes far beyond comparing just the sameness of the partition configurations when	it tries
	// deduce if the parts generated by the two configurations may be equivalent. 
	bool isEquivalent(DataPartitionConfig *other);
  private:
	// a recursive helper routine for the generatePartId(List<int*> lpuIds) function
	void generatePartId(List<int*> *lpuIds, int position, 
		List<int*> *partIdUnderConstr, 
		bool updateExistingPartId = false, int updatePoint = 0);

	// this is a helper function for comparing equivalence of data partition configurations; it provides 
	// a compact hierarchy of vectors (elements of the vectors are the dimension partition instructions)
	// for a configuration by eliminating degenarative instructions
	void preparePartitionHierarchy(List<std::vector<DimPartitionConfig*>*> *hierarchy);

	// this is again a helper function for comparing if two data-partition-configuration are equivalent; it
	// recursively deduces if the way a particular dimension of is divided by two configurations is the same
	bool generateSimilarParts(List<std::vector<DimPartitionConfig*>*> *first, 
			List<std::vector<DimPartitionConfig*>*> *second, 
			int currentDimNo, 
			int currentLevel, 
			Dimension dimToDivide);
};

#endif
