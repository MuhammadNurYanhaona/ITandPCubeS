#include "mmm_structure.h"
#include "../runtime/structure.h"
#include "../utils/list.h"
#include "../utils/hashtable.h"
#include "../utils/partition.h"

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
	bPartList = new List<MatrixPart*>;
	cPartList = new List<MatrixPart*>;
}

bool MatrixPartMap::isIdInList(List<int*> *partId, List<MatrixPart*> *partList) {
	for (int i = 0; i < partList->NumElements(); i++) {
		MatrixPart *part = partList->Nth(i);
		int *queryId = partId->Nth(0);
		int *foundId = part->partId->Nth(0);
		if (queryId[0] == foundId[0] && queryId[1] == foundId[1]) return true;
	}
	return false;
}

MatrixPart *MatrixPartMap::getPart(List<int*> *partId, List<MatrixPart*> *partList) {
	for (int i = 0; i < partList->NumElements(); i++) {
		MatrixPart *part = partList->Nth(i);
		int *queryId = partId->Nth(0);
		int *foundId = part->partId->Nth(0);
		if (queryId[0] == foundId[0] && queryId[1] == foundId[1]) return part;
	}
	return NULL;
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
