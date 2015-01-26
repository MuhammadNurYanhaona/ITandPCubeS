#include "structure.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>

//------------------------------------------- Dimension -------------------------------------------------/

int Dimension::getLength() {
	return abs(range.max - range.min) + 1;
}

void Dimension::setLength(int length) {
	range.min = 0;
	range.max = length - 1;	
}

bool Dimension::isIncreasing() {
	return (range.min <= range.max);
}

Range Dimension::getPositiveRange() {
	if (isIncreasing()) return range;
	Range positiveRange;
	positiveRange.min = range.max;
	positiveRange.max = range.min;
	return positiveRange;
}

Range Dimension::adjustPositiveSubRange(Range positiveSubRange) {
	if (isIncreasing()) return positiveSubRange;
	Range subRange;
	subRange.min = range.min - positiveSubRange.min;
	subRange.max = range.min - positiveSubRange.max;
	return subRange;
}

Dimension Dimension::getNormalizedDimension() {
	int length = getLength();
	Range normalRange;
	normalRange.min = 0;
	normalRange.max = length - 1;
	Dimension normalDimension;
	normalDimension.range = normalRange;
	return normalDimension;	
}

void Dimension::print(std::ofstream &stream) {
	stream << "min " << range.min;
	stream << " max " << range.max;
}

//----------------------------------------- Part Dimension  ---------------------------------------------/

void PartDimension::print(std::ofstream &stream) {
	stream << "storage: ";
	storage.print(stream);
	stream << std::endl;
	stream << "partition: ";
	partition.print(stream);
	stream << std::endl;
}

//--------------------------------------------- PPU ID --------------------------------------------------/

void PPU_Ids::print(std::ofstream &stream) {
	stream << "\tLPS Name: " << lpsName << std::endl;
	stream << "\t\tGroup Id: " << groupId << std::endl;
	stream << "\t\tGroup Size: " << groupSize << std::endl;
	stream << "\t\tPPU Count: " << ppuCount << std::endl;
	if (id != INVALID_ID) {
		stream << "\t\tId: " << id << std::endl;
	}
}

//------------------------------------------- Thread IDs ------------------------------------------------/

void ThreadIds::print(std::ofstream &stream) {
	stream << "Thread No: " << threadNo << std::endl;
	for (int i = 0; i < lpsCount; i++) {
		ppuIds[i].print(stream);
	}
	stream.flush();
}