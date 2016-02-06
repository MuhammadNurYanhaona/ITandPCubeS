#include "part_config.h"
#include "../partition-lib/partition.h"
#include "../utils/binary_search.h"
#include "../communication/part_distribution.h"
#include "../runtime/structure.h"
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
	lpsId = -1;
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
		dataDimensions.insert(dataDimensions.begin() + i, Dimension());
	}
	partitionConfigs.reserve(levels);
	for (int i = 0; i < levels; i++) {
		partitionConfigs.insert(partitionConfigs.begin() + i, new PartitionConfig(dimensions));
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

vector<LpsDimConfig> *DataItemConfig::generateDimOrderVector() {
	vector<LpsDimConfig> *dimOrderVector = new vector<LpsDimConfig>;
	for (int levelNo = 0; levelNo < levels; levelNo++) {
		PartitionConfig *currentConfig = partitionConfigs[levelNo];
		vector<int> instrOrder;
		int dimensionality = dataDimensions.size();
		for (int dimNo = 0; dimNo < dimensionality; dimNo++) {
			PartitionInstr *instr = currentConfig->getInstruction(dimNo);
			int priority = instr->getPriorityOrder();
			int location = binsearch::locatePointOfInsert(instrOrder, priority);
			instrOrder.insert(instrOrder.begin() + location, priority);
			int configLocation = levelNo * dimensionality + location;
			dimOrderVector->insert(dimOrderVector->begin() + configLocation,
					LpsDimConfig(levelNo, dimNo, currentConfig->getLpsId()));
		}
	}
	return dimOrderVector;
}

vector<DimConfig> *DataItemConfig::generateDimOrderVectorWithoutLps() {
	vector<DimConfig> *dimOrderVector = new vector<DimConfig>;
	for (int levelNo = 0; levelNo < levels; levelNo++) {
		PartitionConfig *currentConfig = partitionConfigs[levelNo];
		vector<int> instrOrder;
		int dimensionality = dataDimensions.size();
		for (int dimNo = 0; dimNo < dimensionality; dimNo++) {
			PartitionInstr *instr = currentConfig->getInstruction(dimNo);
			int priority = instr->getPriorityOrder();
			int location = binsearch::locatePointOfInsert(instrOrder, priority);
			instrOrder.insert(instrOrder.begin() + location, priority);
			int configLocation = levelNo * dimensionality + location;
			dimOrderVector->insert(dimOrderVector->begin() + configLocation, DimConfig(levelNo, dimNo));
		}
	}
	return dimOrderVector;
}

void DataItemConfig::disablePaddingInAllLevels() {
	int dimensionality = dataDimensions.size();
	for (int levelNo = 0; levelNo < levels; levelNo++) {
		PartitionConfig *currentConfig = partitionConfigs[levelNo];
		for (int dimNo = 0; dimNo < dimensionality; dimNo++) {
			PartitionInstr *instr = currentConfig->getInstruction(dimNo);
			instr->setExcludePaddingFlag(true);
		}
	}
}
