#include "mmm_structure.h"
#include "../runtime/structure.h"
#include "../utils/list.h"
#include "../utils/hashtable.h"
#include "../utils/partition.h"

#include <iostream>
#include <cstdlib>
#include <time.h>

//---------------------------------------------------------------- ID Generator --------------------------------------------------------------/

IdGenerator::IdGenerator(int *lpuCount) {
	this->lpuCount = lpuCount;
}

List<int*> *IdGenerator::getAPartId(int linearLpuId) {
	int idDim1 = linearLpuId / lpuCount[1];
	List<int*> *partIdList = new List<int*>;
	int *partId = new int[2];
	partId[0] = idDim1;
	partId[1] = 0;
	partIdList->Append(partId);
	return partIdList;	
}

List<int*> *IdGenerator::getBPartId(int linearLpuId) {
	int idDim2 = linearLpuId % lpuCount[1];
	List<int*> *partIdList = new List<int*>;
	int *partId = new int[2];
	partId[0] = 0;
	partId[1] = idDim2;
	partIdList->Append(partId);
	return partIdList;	
}

List<int*> *IdGenerator::getCPartId(int linearLpuId) {
	int idDim1 = linearLpuId / lpuCount[1];
	int idDim2 = linearLpuId % lpuCount[1];
	List<int*> *partIdList = new List<int*>;
	int *partId = new int[2];
	partId[0] = idDim1;
	partId[1] = idDim2;
	partIdList->Append(partId);
	return partIdList;	
}

//----------------------------------------------------------------- Matrix Part --------------------------------------------------------------/

void MatrixPart::duplicate(MatrixPart *copy) {
	copy->storageDims[0] = this->storageDims[0];
	copy->storageDims[1] = this->storageDims[1];
	copy->partId = this->partId;
	int partSize = storageDims[0].getLength() * storageDims[1].getLength();
	copy->data = new double[partSize];
	memcpy(copy->data, this->data, partSize * sizeof(double));
}

bool MatrixPart::sameContent(MatrixPart *other) {
	int partSize = storageDims[0].getLength() * storageDims[1].getLength();
	double *myData = data;
	double *otherData = other->data;
	for (int i = 0; i < partSize; i++) {
		if ((myData[i] - otherData[i]) > 0.01 
			|| (otherData[i] - myData[i] > 0.01)) return false;
	}
	return true;
}

//------------------------------------------------------------ Matrix Part Generator ---------------------------------------------------------/

MatrixPartGenerator::MatrixPartGenerator(int *lpuCount, 
		int blockSize,
		Dimension *aDims,
		Dimension *bDims,
		Dimension *cDims) {
	this->lpuCount = lpuCount;
	this->blockSize = blockSize;
	this->aDims = aDims;
	this->bDims = bDims;
	this->cDims = cDims;			
}

MatrixPart *MatrixPartGenerator::generateAPart(List<int*> *partId) {
	
	int dim1Id = partId->Nth(0)[0];
	Dimension parentDim1 = aDims[0];
	Dimension parentDim2 = aDims[1];
	Dimension dimension1 = block_size_getRange(parentDim1, lpuCount[0], dim1Id, blockSize, 0, 0);
	Dimension dimension2 = parentDim2;
	
	int partSize = dimension1.getLength() * dimension2.getLength();
	
	double *data = new double[partSize];
	for (int i = 0; i < partSize; i++) {
		data[i] = ((rand() % 100) / 75.00);
	}
	
	MatrixPart *part = new MatrixPart();
	part->storageDims[0] = dimension1;
	part->storageDims[1] = dimension2;
	part->data = data;
	part->partId = partId;

	return part;
}

MatrixPart *MatrixPartGenerator::generateBPart(List<int*> *partId) {
	
	int dim2Id = partId->Nth(0)[1];
	Dimension parentDim1 = bDims[0];
	Dimension parentDim2 = bDims[1];
	Dimension dimension1 = parentDim1;
	Dimension dimension2 = block_size_getRange(parentDim2, lpuCount[1], dim2Id, blockSize, 0, 0);
	
	int partSize = dimension1.getLength() * dimension2.getLength();
	
	double *data = new double[partSize];
	for (int i = 0; i < partSize; i++) {
		data[i] = ((rand() % 100) / 75.00);
	}

	MatrixPart *part = new MatrixPart();
	part->storageDims[0] = dimension1;
	part->storageDims[1] = dimension2;
	part->data = data;
	part->partId = partId;

	return part;
}

MatrixPart *MatrixPartGenerator::generateCPart(List<int*> *partId) {
	
	int dim1Id = partId->Nth(0)[0];
	int dim2Id = partId->Nth(0)[1];
	Dimension parentDim1 = cDims[0];
	Dimension parentDim2 = cDims[1];
	Dimension dimension1 = block_size_getRange(parentDim1, lpuCount[0], dim1Id, blockSize, 0, 0);
	Dimension dimension2 = block_size_getRange(parentDim2, lpuCount[1], dim2Id, blockSize, 0, 0);
	
	int partSize = dimension1.getLength() * dimension2.getLength();
	double *data = new double[partSize];
	for (int i = 0; i < partSize; i++) {
		data[i] = 0.0;
	}

	MatrixPart *part = new MatrixPart();
	part->storageDims[0] = dimension1;
	part->storageDims[1] = dimension2;
	part->data = data;
	part->partId = partId;

	return part;
}

//--------------------------------------------------------------- Matrix Part Map ------------------------------------------------------------/

MatrixPartMap::MatrixPartMap() {
	aPartList = new List<MatrixPart*>;
	aSearchIndex = 0;
	bPartList = new List<MatrixPart*>;
	bSearchIndex = 0;
	cPartList = new List<MatrixPart*>;
	cSearchIndex = 0;
}

bool MatrixPartMap::aPartExists(List<int*> *partId) {
	int location = getIdLocation(partId, aPartList, aSearchIndex);
	if (location != -1) {
		aSearchIndex = location;
	}
	return aSearchIndex;
}

MatrixPart *MatrixPartMap::getAPart(List<int*> *partId) {
	int location = getIdLocation(partId, aPartList, aSearchIndex);
	if (location != -1) {
		aSearchIndex = location;
		return aPartList->Nth(aSearchIndex);
	}
	return NULL;
}

bool MatrixPartMap::bPartExists(List<int*> *partId) {
	int location = getIdLocation(partId, bPartList, bSearchIndex);
	if (location != -1) {
		bSearchIndex = location;
	}
	return bSearchIndex;
}

MatrixPart *MatrixPartMap::getBPart(List<int*> *partId) {
	int location = getIdLocation(partId, bPartList, bSearchIndex);
	if (location != -1) {
		bSearchIndex = location;
		return bPartList->Nth(bSearchIndex);
	}
	return NULL;
}

bool MatrixPartMap::cPartExists(List<int*> *partId) {
	int location = getIdLocation(partId, cPartList, cSearchIndex);
	if (location != -1) {
		cSearchIndex = location;
	}
	return cSearchIndex;
}

MatrixPart *MatrixPartMap::getCPart(List<int*> *partId) {
	int location = getIdLocation(partId, cPartList, cSearchIndex);
	if (location != -1) {
		cSearchIndex = location;
		return cPartList->Nth(cSearchIndex);
	}
	return NULL;
}

int MatrixPartMap::getIdLocation(List<int*> *partId, List<MatrixPart*> *partList, int lastSearchedPlace) {
	int partCount = partList->NumElements();
	int searchedParts = 0;
	while (searchedParts < partCount) {
		int searchIndex = (lastSearchedPlace + searchedParts) % partCount;
		MatrixPart *part = partList->Nth(searchIndex);
		int *queryId = partId->Nth(0);
		int *foundId = part->partId->Nth(0);
		if (queryId[0] == foundId[0] && queryId[1] == foundId[1]) return searchIndex;
		searchedParts++;
	}
	return -1;
}

MatrixPartMap *MatrixPartMap::duplicate() {
	
	MatrixPartMap *otherMap = new MatrixPartMap();

	for (int i = 0; i < aPartList->NumElements(); i++) {
		MatrixPart *aPart = aPartList->Nth(i);
		MatrixPart *copy = new MatrixPart();
		aPart->duplicate(copy);
		otherMap->aPartList->Append(copy);
	}
	for (int i = 0; i < bPartList->NumElements(); i++) {
		MatrixPart *bPart = bPartList->Nth(i);
		MatrixPart *copy = new MatrixPart();
		bPart->duplicate(copy);
		otherMap->bPartList->Append(copy);
	}
	for (int i = 0; i < cPartList->NumElements(); i++) {
		MatrixPart *cPart = cPartList->Nth(i);
		MatrixPart *copy = new MatrixPart();
		cPart->duplicate(copy);
		otherMap->cPartList->Append(copy);
	}

	return otherMap;
}

void MatrixPartMap::matchParts(MatrixPartMap *otherMap, std::ofstream &logFile) {
	
	for (int i = 0; i < aPartList->NumElements(); i++) {
		MatrixPart *aPart1 = aPartList->Nth(i);
		MatrixPart *aPart2 = otherMap->aPartList->Nth(i);
		if (!aPart1->sameContent(aPart2)) {
			logFile << "A part mismatch\n";
			logFile.flush();
		}
	}
	for (int i = 0; i < bPartList->NumElements(); i++) {
		MatrixPart *bPart1 = bPartList->Nth(i);
		MatrixPart *bPart2 = otherMap->bPartList->Nth(i);
		if (!bPart1->sameContent(bPart2)) {
			logFile << "B part mismatch\n";
			logFile.flush();
		}
	}
	for (int i = 0; i < cPartList->NumElements(); i++) {
		MatrixPart *cPart1 = cPartList->Nth(i);
		MatrixPart *cPart2 = otherMap->cPartList->Nth(i);
		if (!cPart1->sameContent(cPart2)) {
			logFile << "C part mismatch\n";
			logFile.flush();
		}
	}
}

//------------------------------------------------------------- Get Next LPU Routine ---------------------------------------------------------/

void getNextLpu(int linearLpuId,
		MMMLpu *lpuInstance,
		IdGenerator *idGenerator,
		MatrixPartMap *partMap) {

	List<int*> *aPartId =  idGenerator->getAPartId(linearLpuId);
	MatrixPart *aPart = partMap->getAPart(aPartId);
	int *aId = aPartId->Nth(0);
	delete[] aId;
	delete aPartId;
	
	lpuInstance->a = aPart->data;
	lpuInstance->aPartId = aPart->partId;
	lpuInstance->aPartDims[0].partition = lpuInstance->aPartDims[0].storage = aPart->storageDims[0];	
	lpuInstance->aPartDims[1].partition = lpuInstance->aPartDims[1].storage = aPart->storageDims[1];	

	List<int*> *bPartId =  idGenerator->getBPartId(linearLpuId);
	MatrixPart *bPart = partMap->getBPart(bPartId);
	int *bId = bPartId->Nth(0);
	delete[] bId;
	delete bPartId;
	
	lpuInstance->b = bPart->data;
	lpuInstance->bPartId = bPart->partId;
	lpuInstance->bPartDims[0].partition = lpuInstance->bPartDims[0].storage = bPart->storageDims[0];	
	lpuInstance->bPartDims[1].partition = lpuInstance->bPartDims[1].storage = bPart->storageDims[1];	

	List<int*> *cPartId =  idGenerator->getCPartId(linearLpuId);
	MatrixPart *cPart = partMap->getCPart(cPartId);
	int *cId = cPartId->Nth(0);
	delete[] cId;
	delete cPartId;

	lpuInstance->c = cPart->data;
	lpuInstance->cPartId = cPart->partId;
	lpuInstance->cPartDims[0].partition = lpuInstance->cPartDims[0].storage = cPart->storageDims[0];	
	lpuInstance->cPartDims[1].partition = lpuInstance->cPartDims[1].storage = cPart->storageDims[1];

	lpuInstance->id = linearLpuId;	
}
