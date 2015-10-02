#ifndef PART_CONFIG_H_
#define PART_CONFIG_H_

/* The partition and data configuration classes of this header are similar to that in the part_generation.h library. 
 * The difference is that these classes are state dependent, and created and destroyed as needed as opposed their
 * counter-parts in the part_generation.h library. We have these, seemingly duplicate classes to expedite communication
 * related computations and also to reduce the integration time of communication related features, that we developed
 * in a separate project, into the compiler.     
 */


#include <vector>
#include "../partition-lib/partition.h"
#include "../codegen/structure.h"

class DimConfig;
class LpsDimConfig;

// the class that represents the partition configuration for a data structure on a single LPS
class PartitionConfig {
  private:
	int dimensions;
	std::vector<PartitionInstr*> instructions;
	int lpsId;
  public:
	PartitionConfig(int dimensions);
	void setInstruction(int dimNo, PartitionInstr *instruction);
	PartitionInstr *getInstruction(int dimNo);
	void setLpsId(int lpsId) { this->lpsId = lpsId; }
	int getLpsId() { return lpsId; }
};

// the class that represents the hierarchical partition configuration that leads to the data parts for any LPS
class DataItemConfig {
  private:
	int dimensions;
	int levels;
	std::vector<Dimension> dataDimensions;
	std::vector<PartitionConfig*> partitionConfigs;
  public:
	DataItemConfig(int dimensions, int levels);
	void setDimension(int dimNo, Dimension dimension);
	Dimension getDimension(int dimNo);
	void setPartitionInstr(int levelNo, int dimNo, PartitionInstr *instruction);
	PartitionInstr *getInstruction(int levelNo, int dimNo);
	int getDimensionality() { return dimensions; }
	void setLpsIdOfLevel(int levelNo, int lpsId) { partitionConfigs[levelNo]->setLpsId(lpsId); }
	int getLevels() { return levels; }
	PartitionConfig *getConfigForLevel(int levelNo) { return partitionConfigs.at(levelNo); }

	// parent links should be set appropriately in all partition instructions to be able to generate proper interval
	// descriptions for data parts; this is also important to determine the dimension boundaries at any level in the
	// partition hierarchy for a data part
	void updateParentLinksOnPartitionConfigs();

	// While creating an interval description for the data content or any upper level part containers that embody
	// a set of data parts, we need to know how dimensions have been partitioned in the layers above to determine
	// the dimension boundary at the level the interval calculation is ongoing. The following two functions are
	// provided to set the dimension and parts count properties in the upper levels so that interval calculation can
	// be done correctly in any descendant level.
	void adjustDimensionAndPartsCountAtLevel(int levelNo, int dimNo);
	void setPartIdAtLevel(int levelNo, int dimNo, int partId);

	// To be able to traverse part-distribution-tree (a construct used for communication) we need to know how the
	// partitioned dimensions of a data structure have been ordered in that tree. This function generates the order
	// vector that is followed when constructing that tree.
	std::vector<LpsDimConfig> *generateDimOrderVector();
	// this version is needed to traverse/use the part-container-tree which is specific to a particular LPS chain
	std::vector<DimConfig> *generateDimOrderVectorWithoutLps();
};

#endif /* PART_CONFIG_H_ */
