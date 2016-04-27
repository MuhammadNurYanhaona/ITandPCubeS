#include "structure.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <algorithm>

//------------------------------------------- Dimension -------------------------------------------------/

int Dimension::getLength() {
	return abs(range.max - range.min) + 1;
}

void Dimension::setLength(int length) {
	range.min = 0;
	range.max = length - 1;	
	this->length = length;
}

void Dimension::setLength() {
	this->length = getLength();
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
	normalDimension.length = length;
	return normalDimension;	
}

bool Dimension::isEqual(Dimension other) {
	return (this->range.min == other.range.min) && (this->range.max == other.range.max);
}

void Dimension::print(std::ostream &stream) {
	stream << range.min << "--" << range.max;
}

//----------------------------------------- Part Dimension  ---------------------------------------------/

void PartDimension::print(std::ofstream &stream, int indentLevel) {
	
	for (int i = 0; i < indentLevel; i++) stream << "\t";
	stream << "count: " << count << " ";
	stream << "index: " << index << " ";
	stream << "storage: ";
	storage.print(stream);
	stream << " partition: ";
	partition.print(stream);
	stream << std::endl;
}

bool PartDimension::isIncluded(int index) {
	if (partition.range.min > partition.range.max) {
		return index >= partition.range.max 
				&& index <= partition.range.min;
	} else { 
		return index >= partition.range.min 
				&& index <= partition.range.max;
	}
}

int PartDimension::adjustIndex(int index) {
	if (partition.range.min > partition.range.max)
		return partition.range.min - index;
	else return index + partition.range.min;
}

int PartDimension::safeNormalizeIndex(int index, bool matchToMin) {
	int normalIndex = index - partition.range.min;
	int length = partition.getLength();	
	if (normalIndex >= 0 && normalIndex < length) return normalIndex;
	return (matchToMin) ? 0 : length - 1;	
}

PartDimension PartDimension::getSubrange(int begin, int end) {
	PartDimension subDimension = PartDimension();
	subDimension.storage = this->storage;
	subDimension.partition.range.min = begin;
	subDimension.partition.range.max = end;
	subDimension.partition.setLength();
	return subDimension;
}

int PartDimension::getDepth() {
	if (parent == NULL) return 1;
	else return parent->getDepth() + 1;
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

int *ThreadIds::getAllPpuCounts() {
	int *counts = new int[lpsCount];
	for (int i = 0; i < lpsCount; i++) counts[i] = ppuIds[i].ppuCount;
	return counts;
}
