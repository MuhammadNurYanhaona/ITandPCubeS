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

/* Superclass to represent the partition configuration applied for a particular dimension of an array for
   an LPS. 
*/
class DimPartitionConfig {
  protected:
	// span of the dimension of the underlying array
	Dimension dimension;
	// any arguments needed by the partition function applied on this dimension
	int *partitionArgs;
	// for any partition there may be a padding at the beginning and a padding at the end 
	int paddings[2];
	// the number of physical processors that are responsible for managing LPUs of the concerned LPS 
	int ppuCount;
	// represents which dimension of the LPS current dimension of the array is aligned to 
	int lpsAlignment;
  public:
	DimPartitionConfig(Dimension dimension, int *partitionArgs, 
			int paddings[2], int ppuCount, int lpsAlignment) {
		this->dimension = dimension;
		this->partitionArgs = partitionArgs;
		this->paddings[0] = paddings[0];
		this->paddings[1] = paddings[1];
		this->ppuCount = ppuCount;
		this->lpsAlignment = lpsAlignment;
	}
	// constructor to be overriden by functions that do not support padding
	DimPartitionConfig(Dimension dimension, int *partitionArgs, int ppuCount, int lpsAlignment) {
		this->dimension = dimension;
		this->partitionArgs = partitionArgs;
		this->paddings[0] = 0;
		this->paddings[1] = 0;
		this->ppuCount = ppuCount;
		this->lpsAlignment = lpsAlignment;
	}
	bool hasPadding() { return paddings[0] > 0 || paddings[1] > 0; }
	Dimension getDataDimension() { return dimension; }

	// determines how many unique parts of the data structure can be found along this dimension 
	virtual int getPartsCount() = 0;

	// Four functions to determine the interval configuration for a data part. The first and third 
	// ignores padding and the second and fourth do not. The last two are needed for index reordering 
	// partition functions. As each data part is kept separate, data parts for functions like stride 
	// and block-stride get their allocation holding non-consecutive indexes in shuffled into a 
	// consecutive order. Maintaining an interval configuration for the storage format may sometimes 
	// allow us to streamline the calculation of communication need.    
	virtual LineInterval *getCoreInterval(int partId) = 0;
	virtual LineInterval *getInterval(int partId) = 0;
	virtual LineInterval *getXformedCoreInterval(int partId) { return getCoreInterval(partId); }
	virtual LineInterval *getXformedInterval(int partId) { return getInterval(partId); }

	// determines the part dimension given a part Id; for reordering partition function it is the
	// part dimension after the indexes have been shuffled to have each part occupying a contiguous
	// chunk along the dimension line
	virtual Dimension getPartDimension(int partId) = 0;
	// these two functions determine the actual amount of padding been applied to a particular part;
	// the effective padding may differ from the configuration for parts at the boundary 
	virtual int getEffectiveFrontPadding(int partId) { return paddings[0]; }		
	virtual int getEffectiveRearPadding(int partId) {  return paddings[1]; }
	// retrieves the dimension index, or part-Id, from the lpuId
	virtual int pickPartId(int *lpuId) { return lpuId[lpsAlignment]; } 		
};

/* configuration subclass to be instantiated when a data dimension has not been divided within the LPS */
class ReplicationConfig : public DimPartitionConfig {
  public:
	ReplicationConfig(Dimension dimension) : DimPartitionConfig(dimension, NULL, 0, 0) {}
	int getPartsCount() { return 1; }
	LineInterval *getCoreInterval(int partId) {
		Line *line = new Line(dimension.range.min, dimension.range.max);
		return LineInterval::getFullLineInterval(line);
	}
	LineInterval *getInterval(int partId) { return getCoreInterval(partId); }
	LineInterval *getXformedCoreInterval(int partId) { return getCoreInterval(partId); }
	LineInterval *getXformedInterval(int partId) { return getCoreInterval(partId); }		
	Dimension getPartDimension(int partId) { return dimension; }		
	int getEffectiveFrontPadding(int partId) { return 0; } 		
	int getEffectiveRearPadding(int partId) { return 0;  }		
	int pickPartId(int *lpsId) { return 0; } 		
};

/* configuration subclass corresponding to 'block_size' partition function; it takes a 'size' parameter */
class BlockSizeConfig : public DimPartitionConfig {
  public:
	BlockSizeConfig(Dimension dimension, int *partitionArgs, int paddings[2], 
			int ppuCount, int lpsAlignment) : DimPartitionConfig(dimension, 
			partitionArgs, paddings, ppuCount, lpsAlignment) {}
	int getPartsCount();
	LineInterval *getCoreInterval(int partId);
	LineInterval *getInterval(int partId);
	Dimension getPartDimension(int partId);
	int getEffectiveFrontPadding(int partId); 		
	int getEffectiveRearPadding(int partId);
};

/* configuration subclass corresponding to 'block_count' partition function; it takes a 'count' parameter */
class BlockCountConfig : public DimPartitionConfig {
  public:
	BlockCountConfig(Dimension dimension, int *partitionArgs, int paddings[2], 
			int ppuCount, int lpsAlignment) : DimPartitionConfig(dimension, 
			partitionArgs, paddings, ppuCount, lpsAlignment) {}
	int getPartsCount();
	LineInterval *getCoreInterval(int partId);
	LineInterval *getInterval(int partId);
	Dimension getPartDimension(int partId);
	int getEffectiveFrontPadding(int partId); 		
	int getEffectiveRearPadding(int partId);
};

/* configuration subclass for parameter-less 'stride' partition function */
class StrideConfig : public DimPartitionConfig {
  public:
	StrideConfig(Dimension dimension, int ppuCount, int lpsAlignment) 
			: DimPartitionConfig(dimension, NULL, ppuCount, lpsAlignment) {}
	int getPartsCount();
	LineInterval *getCoreInterval(int partId);
	LineInterval *getInterval(int partId) { return getCoreInterval(partId); }
	LineInterval *getXformedCoreInterval(int partId);
	LineInterval *getXformedInterval(int partId) { return getXformedCoreInterval(partId); }
	Dimension getPartDimension(int partId);
};

/* configuration subclass for 'block_stride' partition function that takes a 'block_size' parameter */
class BlockStrideConfig : public DimPartitionConfig {
  public:
	BlockStrideConfig(Dimension dimension, int *partitionArgs, 
			int ppuCount, int lpsAlignment) : DimPartitionConfig(dimension, 
			partitionArgs, ppuCount, lpsAlignment) {}
	int getPartsCount();
	LineInterval *getCoreInterval(int partId);
	LineInterval *getInterval(int partId) { return getCoreInterval(partId); }
	LineInterval *getXformedCoreInterval(int partId);
	LineInterval *getXformedInterval(int partId) { return getXformedCoreInterval(partId); }
	Dimension getPartDimension(int partId);
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
  public:
	DataPartitionConfig(int dimensionCount, List<DimPartitionConfig*> *dimensionConfig) {
		this->dimensionCount = dimensionCount;
		this->dimensionConfigs = dimensionConfigs;
	}
	// the function is to be used for generating metadata for subsequent use in generating a data part
	PartMetadata *generatePartMetadata(int *partId);
	// utility function to retrieve multi-dimensional LPU ids from a translated linear id
	int *getMultidimensionalLpuId(int lpsDimensions, int *lpuCount, int linearId);
	
	// A recursive function to determine lpu ids from the multiplex range configuration that specifies 
	// a section of the LPS been assigned to the current PPU. This will be useful when we will allow
	// programming control of the nature of LPU-to-PPU multiplexing. Currently the strategy we are 
	// using, which is the same default been imported from the multicore backend, is to linearize the
	// lpu ids then give each PPU a contiguous segment from the id line. This strategy seems to not be 
	// the best choice when we want better control over the nature of commmunications and memory load
	// balancing.  
	List<int*> *getLpuIdsFromRange(int lpsDimensions, int currentDimension, Range *localLpus);
	
	// two functions for determining part-ids that are unique to current PPU from the lpus allocated
	// to it for execution
	List<int*> *getLocalPartIds(int lpsDimensions, int *lpuCount, Range localRange);
	List<int*> *getLocalPartIds(List<int*> *localLpuIds);

	// generate a data part Id for an LPU from the LPU Id
	int *generatePartId(int *lpuId);

	// function to generate the list of data parts (see allocation.h) from the partition configuration
	template <class type> DataPartsList *generatePartList(List<int*> *localPartIds, int epochCount) {
		List<PartMetadata*> *partMetadataList = new List<PartMetadata*>;
		for (int i = 0; i < localPartIds->NumElements(); i++) {
			int *partId = localPartIds->Nth(i);
			partMetadataList->Append(generatePartMetadata(partId));
		}
		Dimension *dataDimensions = new Dimension[dimensionCount];
		bool hasPadding = false;
		for (int d = 0; d < dimensionCount; d++) {
			DimPartitionConfig *dimConfig = dimensionConfigs->Nth(d);
			dataDimensions[d] = dimConfig->getDataDimension();
			hasPadding = hasPadding || dimConfig->hasPadding();
		}
		ListMetadata *listMetadata = new ListMetadata(dimensionCount, dataDimensions);
		listMetadata->setPadding(hasPadding);
		listMetadata->generateIntervalSpec(partMetadataList);
		DataPartsList *dataPartsList = new DataPartsList(listMetadata, epochCount);
		dataPartsList->allocate <type> (partMetadataList);
        	return dataPartsList;
	}

	// this function is used to determine the data-parts content of PPUs other than the current one
	// so that decision about the nature and content of communication for shared data can be made.
	ListMetadata *generatePartsMetadata(List<int*> *partIds);
};

#endif
