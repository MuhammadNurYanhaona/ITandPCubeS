#include "part_config.h"
#include "../utils/partition.h"
#include "../structure.h"
#include <iostream>
#include <vector>
#include <cstdlib>

using namespace std;

//---------------------------------------------------- LPS Partition Configuration ------------------------------------------------------/

PartitionConfig::PartitionConfig(int dimensions) {
	this->dimensions = dimensions;
	instructions.reserve(dimensions);
	for (int i = 0; i < dimensions; i++) {
		instructions.push_back(NULL);
	}
}

void PartitionConfig::setInstruction(int dimNo, PartitionInstr *instruction) {
	instructions[dimNo] = instruction;
}

PartitionInstr *PartitionConfig::getInstruction(int dimNo) {
	return instructions[dimNo];
}

//------------------------------------------------------ Data Item Configuration --------------------------------------------------------/

DataItemConfig::DataItemConfig(int dimensions, int levels) {
	this->dimensions = dimensions;
	this->levels = levels;
	dataDimensions.reserve(dimensions);
	for (int i = 0; i < dimensions; i++) {
		dataDimensions[i] = Dimension();
	}
	partitionConfigs.reserve(levels);
	for (int i = 0; i < levels; i++) {
		partitionConfigs[i] = new PartitionConfig(dimensions);
	}
}

void DataItemConfig::setDimension(int dimNo, Dimension dimension) {
	dataDimensions[dimNo] = dimension;
}

Dimension DataItemConfig::getDimension(int dimNo) {
	return dataDimensions[dimNo];
}

void DataItemConfig::setPartitionInstr(int levelNo, int dimNo, PartitionInstr *instruction) {
	PartitionConfig *partitionConfig = partitionConfigs[levelNo];
	partitionConfig->setInstruction(dimNo, instruction);
}

PartitionInstr *DataItemConfig::getInstruction(int levelNo, int dimNo) {
	PartitionConfig *partitionConfig = partitionConfigs[levelNo];
	return partitionConfig->getInstruction(dimNo);
}

void DataItemConfig::updateParentLinksOnPartitionConfigs() {
	for (int i = 1; i < levels; i++) {
		PartitionConfig *currentConfig = partitionConfigs[i];
		PartitionConfig *previousConfig = partitionConfigs[i - 1];
		for (int j = 0; j < dimensions; j++) {
			PartitionInstr *prevInstr = previousConfig->getInstruction(j);
			PartitionInstr *currInstr = currentConfig->getInstruction(j);
			currInstr->setPrevInstr(prevInstr);
		}
	}
}

void DataItemConfig::adjustDimensionAndPartsCountAtLevel(int levelNo, int dimNo) {
	PartitionConfig *currentConfig = partitionConfigs[levelNo];
	PartitionInstr *instr = currentConfig->getInstruction(dimNo);
	Dimension dimension;
	if (levelNo == 0) {
		dimension = dataDimensions[dimNo];
	} else {
		dimension = instr->getPreviousInstr()->getDimension();
	}
	instr->calculatePartsCount(dimension, true);
}

void DataItemConfig::setPartIdAtLevel(int levelNo, int dimNo, int partId) {
	PartitionConfig *currentConfig = partitionConfigs[levelNo];
	PartitionInstr *instr = currentConfig->getInstruction(dimNo);
	instr->setPartId(partId);
}
