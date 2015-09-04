#ifndef PART_CONFIG_H_
#define PART_CONFIG_H_

#include <vector>
#include "../utils/partition.h"
#include "../structure.h"

// the class that represents the partition configuration for a data structure on a single LPS
class PartitionConfig {
private:
	int dimensions;
	std::vector<PartitionInstr*> instructions;
public:
	PartitionConfig(int dimensions);
	void setInstruction(int dimNo, PartitionInstr *instruction);
	PartitionInstr *getInstruction(int dimNo);
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
};

#endif /* PART_CONFIG_H_ */
