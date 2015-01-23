#include "structure.h"
#include <iostream>
#include <fstream>

void PPU_Ids::print(std::ofstream &stream) {
	stream << "\tLPS Name: " << lpsName << std::endl;
	stream << "\t\tGroup Id: " << groupId << std::endl;
	stream << "\t\tGroup Size: " << groupSize << std::endl;
	stream << "\t\tPPU Count: " << ppuCount << std::endl;
	if (id != INVALID_ID) {
		stream << "\t\tId: " << id << std::endl;
	}
}

void ThreadIds::print(std::ofstream &stream) {
	stream << "Thread No: " << threadNo << std::endl;
	for (int i = 0; i < lpsCount; i++) {
		ppuIds[i].print(stream);
	}
	stream.flush();
}
