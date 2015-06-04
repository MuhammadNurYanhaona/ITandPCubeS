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

#include "allocation.h"
#include "../utils/list.h"
#include "../utils/interval_utils.h"
#include "../codegen/structure.h"

/* This is just a helper class to hold the results of different configuration functions of Dim Parition 
   Config class as needed for data parts construction. 
*/
class DimensionMetadata {
  public:
	LineInterval *coreInterval;
	LineInterval *interval;
	Dimension partDimension;
	int paddings[2];
};

/* Superclass to represent the partition configuration applied for a particular dimension of an array for
   an LPS. Note that this class has a parent reference. This is needed as partitioning in IT is hierarchi-
   cal and the properties of a lower LPS data part often cannot be found without investigating its higher
   LPS ancestor parts. For the same reasons most of the public method of this class takes a list of part
   Ids as input as oppose to a single part id. Here each item in the list identifies a part in some LPS
   that is in the path that leads to the data part of the LPS under concern.  
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

	// a recursive helper routine to get the part of the dimension that been subject to partitioning
	// by the current dimension configuration instance 
	Dimension getDimensionFromParent(List<int> *partIdList, int position);
	
	// determines the part dimension given a part Id; for reordering partition function it is the
	// part dimension after the indexes have been shuffled to have each part occupying a contiguous
	// chunk along the dimension line
	virtual Dimension getPartDimension(int partId, Dimension parentDimension) = 0;
	
	// these two functions determine the actual amount of padding been applied to a particular part;
	// the effective padding may differ from the configuration for parts at the boundary 
	virtual int getEffectiveFrontPadding(int partId, Dimension parentDimension) { return paddings[0]; }		
	virtual int getEffectiveRearPadding(int partId, Dimension parentDimension) {  return paddings[1]; }

  public:
	DimPartitionConfig(Dimension dimension, int *partitionArgs, 
			int paddings[2], int ppuCount, int lpsAlignment);
	// constructor to be overriden by functions that do not support padding
	DimPartitionConfig(Dimension dimension, int *partitionArgs, int ppuCount, int lpsAlignment);

	bool hasPadding() { return paddings[0] > 0 || paddings[1] > 0; }
	Dimension getDataDimension() { return dataDimension; }
	void setParentConfig(DimPartitionConfig *parentConfig) { this->parentConfig = parentConfig; }

	// Four functions to determine the interval configuration for a data part. The first and third 
	// ignores padding and the second and fourth do not. The last two are needed for index reordering 
	// partition functions. As each data part is kept separate, data parts for functions like stride 
	// and block-stride get their allocation holding non-consecutive indexes in shuffled into a 
	// consecutive order. Maintaining an interval configuration for the storage format may sometimes 
	// allow us to streamline the calculation of communication need.    
	virtual LineInterval *getCoreInterval(List<int> *partIdList) = 0;
	virtual LineInterval *getInterval(List<int> *partIdList) = 0;
	virtual LineInterval *getXformedCoreInterval(List<int> *partIdList);
	virtual LineInterval *getXformedInterval(List<int> *partIdList);

	// retrieves the dimension index, or part-Id, from the lpuId for current LPS
	virtual int pickPartId(int *lpuId) { return lpuId[lpsAlignment]; }

	// an accumulator function that returns results of different dimension configuration utilies above 
	DimensionMetadata *generateDimMetadata(List<int> *partIdList);

	// function to be used by data partition config to generate dimension metadata about the structure 
	Dimension getPartDimension(List<int> *partIdList);
	
	// determines how many unique parts of the data structure can be found along this dimension by
	// dividing the parent dimension 
	virtual int getPartsCount(Dimension parentDimension) = 0;

};

/* configuration subclass to be instantiated when a data dimension has not been divided within the LPS */
class ReplicationConfig : public DimPartitionConfig {
  protected:
	int getEffectiveFrontPadding(int partId, Dimension parentDimension) { return 0; } 		
	int getEffectiveRearPadding(int partId, Dimension parentDimension) { return 0;  }		
	Dimension getPartDimension(int partId, Dimension parentDimension) { return parentDimension; }		
  public:
	ReplicationConfig(Dimension dimension) : DimPartitionConfig(dimension, NULL, 0, 0) {}
	
	LineInterval *getCoreInterval(List<int> *partIdList);
	LineInterval *getInterval(List<int> *partIdList);
	LineInterval *getXformedCoreInterval(List<int> *partIdList);
	LineInterval *getXformedInterval(List<int> *partIdList);		
	int pickPartId(int *lpsId) { return 0; } 		
	int getPartsCount(Dimension parentDimension) { return 1; }
};

/* configuration subclass corresponding to 'block_size' partition function; it takes a 'size' parameter */
class BlockSizeConfig : public DimPartitionConfig {
  protected:
	int getEffectiveFrontPadding(int partId, Dimension parentDimension); 		
	int getEffectiveRearPadding(int partId,  Dimension parentDimension);
	Dimension getPartDimension(int partId,   Dimension parentDimension);
  public:
	BlockSizeConfig(Dimension dimension, int *partitionArgs, int paddings[2], 
			int ppuCount, int lpsAlignment) : DimPartitionConfig(dimension, 
			partitionArgs, paddings, ppuCount, lpsAlignment) {}
	
	int getPartsCount(Dimension parentDimension);
	LineInterval *getCoreInterval(List<int> *partIdList);
	LineInterval *getInterval(List<int> *partIdIdList);
};

/* configuration subclass corresponding to 'block_count' partition function; it takes a 'count' parameter */
class BlockCountConfig : public DimPartitionConfig {
  protected:
	int getEffectiveFrontPadding(int partId, Dimension parentDimension); 		
	int getEffectiveRearPadding(int partId,  Dimension parentDImension);
	Dimension getPartDimension(int partId,   Dimension parentDimension);
  public:
	BlockCountConfig(Dimension dimension, int *partitionArgs, int paddings[2], 
			int ppuCount, int lpsAlignment) : DimPartitionConfig(dimension, 
			partitionArgs, paddings, ppuCount, lpsAlignment) {}
	
	int getPartsCount(Dimension parentDimension);
	LineInterval *getCoreInterval(List<int> *partIdList);
	LineInterval *getInterval(List<int> *partIdList);
};

/* configuration subclass for parameter-less 'stride' partition function */
class StrideConfig : public DimPartitionConfig {
  protected:
	Dimension getPartDimension(int partId, Dimension parentDimension);
  public:
	StrideConfig(Dimension dimension, int ppuCount, int lpsAlignment) 
			: DimPartitionConfig(dimension, NULL, ppuCount, lpsAlignment) {}
	
	int getPartsCount(Dimension parentDimension);
	LineInterval *getCoreInterval(List<int> *partIdList);
	LineInterval *getInterval(List<int> *partIdList) { return getCoreInterval(partIdList); }
	LineInterval *getXformedCoreInterval(List<int> *partIdList);
	LineInterval *getXformedInterval(List<int> *partIdList) { return getXformedCoreInterval(partIdList); }
};

/* configuration subclass for 'block_stride' partition function that takes a 'block_size' parameter */
class BlockStrideConfig : public DimPartitionConfig {
  protected:
	Dimension getPartDimension(int partId, Dimension parentDimension);
  public:
	BlockStrideConfig(Dimension dimension, int *partitionArgs, 
			int ppuCount, int lpsAlignment) : DimPartitionConfig(dimension, 
			partitionArgs, ppuCount, lpsAlignment) {}
	
	int getPartsCount(Dimension parentDimension);
	LineInterval *getCoreInterval(List<int> *partIdList);
	LineInterval *getInterval(List<int> *partIdList) { return getCoreInterval(partIdList); }
	LineInterval *getXformedCoreInterval(List<int> *partIdList);
	LineInterval *getXformedInterval(List<int> *partIdList) { return getXformedCoreInterval(partIdList); }
};

/* This is the class that holds the partition configuration for different dimensions of a single data 
   structure within an LPS. An instance of this class needs to be instantiated based on the instructions
   about partitions and once the length of different dimensions are known. Know that will be one dim-
   PartitionConfig instance per dimension of the structure. That is the configuration is independent of
   the dimensionality of the LPS itself. Any dimension not been partitioned by the LPS should have a 
   replication-config for it. This strategy enables us to readily compare parts of one LPS with parts of
   other LPSes as the dimensionality of the parts remain the same regardless of the dimensionality of the
   LPSes they are mean for. 
*/
class DataPartitionConfig {
  protected:
	// the number of dimension in the data structure
	int dimensionCount;
	// partition configuration along each of those dimensions
	List<DimPartitionConfig*> *dimensionConfigs;
	// A reference to the parition configuration of immediate ancestor LPS cantaining the underlying 
	// data structure
	DataPartitionConfig *parent;
  public:
	DataPartitionConfig(int dimensionCount, List<DimPartitionConfig*> *dimensionConfig) {
		this->dimensionCount = dimensionCount;
		this->dimensionConfigs = dimensionConfigs;
		this->parent = NULL;
	}
	void setParent(DataPartitionConfig *parent);
	
	// the function is to be used for generating metadata for subsequent use in generating a data part
	PartMetadata *generatePartMetadata(List<int*> *partIdList);

	// generates a data part Id for an LPU from the LPU Id
	List<int*> *generatePartId(List<int*> *lpuIds);
	// function to generate a list of unique part Ids from LPU Ids list
	List<List<int*>*> *generatePartIdList(List<List<int*>*> *lpuIdList);

	// function to set up the metadata regarding the partition dimensions of the underlying object 
	// within an LPU
	void updatePartDimensionInfo(List<int*> *partIdList, PartDimension *partDimension);

	// function to be used to determine how many LPUs should be there within an LPS at runtime; when
	// the second argument is NULL, data-dimension should be used to determine the parts count
	int getPartsCountAlongDimension(int dimensionNo, Dimension *parentDimension = NULL);

	// According to the chosen strategy the data part for an LPU for an LPS may refer to a portion of
	// a larger part from an ancestor LPS. Therefore, at runtime the identity of that ancestor part 
	// needs to be calculated as opposed to the smaller subpart the LPU refers to. This method provides
	// that functionality. Here the second parameter indicates how many lpuIds should be skipped from 
	// the rear to reach the desired ancestor.
	List<int*> *generateSuperPartIdList(List<int*> *lpuIds, int backsteps);  

	// function to generate the list of data parts (see allocation.h) from the partition configuration
	template <class type> static DataPartsList *generatePartList(DataPartitionConfig *config, 
			List<List<int*>*> *localPartIds, int epochCount) {
		List<PartMetadata*> *partMetadataList = new List<PartMetadata*>;
		for (int i = 0; i < localPartIds->NumElements(); i++) {
			List<int*> *partIdList = localPartIds->Nth(i);
			partMetadataList->Append(config->generatePartMetadata(partIdList));
		}
		Dimension *dataDimensions = new Dimension[config->dimensionCount];
		bool hasPadding = false;
		for (int d = 0; d < config->dimensionCount; d++) {
			DimPartitionConfig *dimConfig = config->dimensionConfigs->Nth(d);
			dataDimensions[d] = dimConfig->getDataDimension();
			hasPadding = hasPadding || dimConfig->hasPadding();
		}
		ListMetadata *listMetadata = new ListMetadata(config->dimensionCount, dataDimensions);
		listMetadata->setPadding(hasPadding);
		listMetadata->generateIntervalSpec(partMetadataList);
		DataPartsList *dataPartsList = new DataPartsList(listMetadata, epochCount);
		DataPartsList::allocate<type>(dataPartsList, partMetadataList);
        	return dataPartsList;
	}

	// this function is used to determine the data-parts content of PPUs other than the current one
	// so that decision about the nature and content of communication for shared data can be made.
	ListMetadata *generatePartListMetadata(List<List<int*>*> *partIds);

  private:
	// a recursive helper routine for the generatePartId(List<int*> lpuIds) function
	void generatePartId(List<int*> *lpuIds, int position, List<int*> *partIdUnderConstr);
};

#endif
