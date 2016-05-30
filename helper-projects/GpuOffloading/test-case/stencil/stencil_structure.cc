#include "stencil_structure.h"
#include "../../runtime/structure.h"
#include "../../utils/list.h"
#include "../../utils/hashtable.h"
#include "../../utils/partition.h"
#include "../../utils/binary_search.h"

#include <vector>
#include <iostream>
#include <cstdlib>
#include <time.h>

using namespace stencil;

//---------------------------------------------------------------- ID Generator --------------------------------------------------------------/

IdGenerator::IdGenerator(int lpuCount) {
        this->lpuCount = lpuCount;
}

List<int*> *IdGenerator::getPartId(int linearLpuId) {
	List<int*> *partIdList = new List<int*>;
        int *partId = new int[1];
	partId[0] = linearLpuId;
	partIdList->Append(partId);
	return partIdList;
}

//----------------------------------------------------------------- Plate Part ---------------------------------------------------------------/

void PlatePart::duplicate(PlatePart *copy) {
	copy->storageDims[0] = this->storageDims[0];
        copy->storageDims[1] = this->storageDims[1];
        copy->partId = this->partId;
        int partSize = storageDims[0].getLength() * storageDims[1].getLength();
        copy->data = new double[partSize];
        memcpy(copy->data, this->data, partSize * sizeof(double));
        memcpy(copy->data_lag_1, this->data_lag_1, partSize * sizeof(double));
}
                
bool PlatePart::sameContent(PlatePart *other) {
	int partSize = storageDims[0].getLength() * storageDims[1].getLength();
        double *myData = data;
        double *otherData = other->data;
        for (int i = 0; i < partSize; i++) {
                if ((myData[i] - otherData[i]) > 0.01
                        || (otherData[i] - myData[i] > 0.01)) return false;
        }
        return true;
}

//------------------------------------------------------------- Plate Part Generator ---------------------------------------------------------/

PlatePartGenerator::PlatePartGenerator(int lpuCount, int padding, Dimension *plateDims) {
	this->lpuCount = lpuCount;
	this->padding = padding;
	this->plateDims = plateDims;
}

PlatePart *PlatePartGenerator::generatePart(List<int*> *partId) {
	int dimId = partId->Nth(0)[0];
        Dimension parentDim1 = plateDims[0];
        Dimension parentDim2 = plateDims[1];
        Dimension dimension1 = block_count_getRange(parentDim1, lpuCount, dimId, padding, padding);
        Dimension dimension2 = parentDim2;

        int partSize = dimension1.getLength() * dimension2.getLength();

        double *data = new double[partSize];
	double *data_lag_1 = new double[partSize];
        for (int i = 0; i < partSize; i++) {
                data[i] = ((rand() % 100) / 75.00);
        }
        memcpy(data_lag_1, data, partSize * sizeof(double));

        PlatePart *part = new PlatePart();
        part->storageDims[0] = dimension1;
        part->storageDims[1] = dimension2;
        part->data = data;	
	part->data_lag_1 = data_lag_1;
        part->partId = partId;

        return part;
}

//-------------------------------------------------------------- Part Id Container ----------------------------------------------------------/

void PartIdContainer::addPartId(List<int> *partId) {
	int id = partId->Nth(0);
        int location = binsearch::locateKey(partArray, id);
        if (location == KEY_NOT_FOUND) {
                int insertIndex = binsearch::locatePointOfInsert(partArray, id);
                partArray.insert(partArray.begin() + insertIndex, id);
        }
}
                
bool PartIdContainer::doesIdExist(List<int> *partId) {
	int id = partId->Nth(0);
        int location = binsearch::locateKey(partArray, id);
        return (location != KEY_NOT_FOUND);
}

//---------------------------------------------------------------- Part Part Map ------------------------------------------------------------/

PlatePartMap::PlatePartMap() {
	partList = new List<PlatePart*>;
        idContainer = new PartIdContainer();
}
                
bool PlatePartMap::partExists(List<int*> *partId) {
	List<int> idList;
	idList.Append(partId->Nth(0)[0]);
	return idContainer->doesIdExist(&idList);
}
                
void PlatePartMap::addPart(PlatePart *part) {
	partList->Append(part);
        List<int> idList;
	idList.Append(part->partId->Nth(0)[0]);
        idContainer->addPartId(&idList);
}
                
PlatePart *PlatePartMap::getPart(List<int*> *partId) {
	int id = partId->Nth(0)[0];
	for (int i = 0; i < partList->NumElements(); i++) {
		PlatePart *part = partList->Nth(i);
		if (part->partId->Nth(0)[0] == id) return part;
	}
	return NULL;
}
                
PlatePartMap *PlatePartMap::duplicate() {
	PlatePartMap *otherMap = new PlatePartMap();
	for (int i = 0; i < partList->NumElements(); i++) {
                PlatePart *part = partList->Nth(i);
                PlatePart *copy = new PlatePart();
                part->duplicate(copy);
                otherMap->addPart(copy);
        }
	return otherMap;
}
                
void PlatePartMap::matchParts(PlatePartMap *otherMap, std::ofstream &logFile) {
	for (int i = 0; i < partList->NumElements(); i++) {
                PlatePart *part1 = partList->Nth(i);
                PlatePart *part2 = otherMap->partList->Nth(i);
                if (!part1->sameContent(part2)) {
                        logFile << "plate part mismatch\n";
                        logFile.flush();
                }
        }
}

//------------------------------------------------------------- Get Next LPU Routine ---------------------------------------------------------/

void getNextLpu(int linearLpuId,
		stencil::StencilLpu *lpuInstance,
		stencil::IdGenerator *idGenerator,
		stencil::PlatePartMap *partMap) {

	List<int*> *partId =  idGenerator->getPartId(linearLpuId);
        PlatePart *part = partMap->getPart(partId);
        int *id = partId->Nth(0);
        delete[] id;
        delete partId;

	lpuInstance->plate = part->data;
	lpuInstance->plate_lag_1 = part->data_lag_1;
        lpuInstance->platePartId = part->partId;
        lpuInstance->platePartDims[0].partition = lpuInstance->platePartDims[0].storage = part->storageDims[0];
        lpuInstance->platePartDims[1].partition = lpuInstance->platePartDims[1].storage = part->storageDims[1];
}

