#ifndef _H_mmm_structure
#define _H_mmm_structure

/* This header file has classes that serve as a replacement for the LPU and data part generation mechanism we have in the
 * compilers. This matrix-matrix multiplication and fixed partition configuration specific classes are written as porting
 * the existing LPU and part generation libraries in this project would be to much hassle.
 */

#include "../runtime/structure.h"
#include "../utils/list.h"
#include "../utils/hashtable.h"

#include <vector>
#include <fstream>

// This namespace contains the classes that we dynamically generate for individual task to track scalar variables' state,
// dimensionality of the originial data structures used inside a task, and task partition information. All of these may be
// needed during the execution of a compute stage.
namespace mmm {

	class Partition {
	  public:
		int blockSize;
	};

	class ArrayMetadata {
	  public:
		Dimension aDims[2];
		Dimension bDims[2];
		Dimension cDims[2];
	};

	class TaskGlobals {
	  public:
	};

	class ThreadLocals {
	  public:
	};
}

class MMMLpu : public LPU {
  public:
        double *a;
        List<int*> *aPartId;
        PartDimension aPartDims[2];
        double *b;
        List<int*> *bPartId;
        PartDimension bPartDims[2];
        double *c;
        List<int*> *cPartId;
        PartDimension cPartDims[2];
  public:
        MMMLpu() : LPU() {
                a = NULL;
                aPartId = NULL;
                b = NULL;
                bPartId = NULL;
                c = NULL;
                cPartId = NULL;
        }
};

class IdGenerator {
  private:
        int *lpuCount;
  public:
	IdGenerator(int *lpuCount);
        List<int*> *getAPartId(int linearLpuId);
        List<int*> *getBPartId(int linearLpuId);
        List<int*> *getCPartId(int linearLpuId);
};

class MatrixPart {
  public:
        double *data;
        Dimension storageDims[2];
        List<int*> *partId;
  public:
	MatrixPart() {
		data = NULL;
		partId = NULL;
	}
	void duplicate(MatrixPart *copy);
	bool sameContent(MatrixPart *other);
};

class MatrixPartGenerator {
  private:
        int *lpuCount;
	int blockSize;
        Dimension *aDims;
        Dimension *bDims;
        Dimension *cDims;
  public:
        MatrixPartGenerator(int *lpuCount,
			int blockSize, 
			Dimension *aDims, 
			Dimension *bDims, 
			Dimension *cDims);
        MatrixPart *generateAPart(List<int*> *partId);
        MatrixPart *generateBPart(List<int*> *partId);
        MatrixPart *generateCPart(List<int*> *partId);
};

class PartIdContainer {
  protected:
	std::vector<int> partArray;
	std::vector<PartIdContainer*> nextContainers;
  public:
	PartIdContainer() {}
	bool doesIdExist(List<int> *partId) { return  doesIdExist(partId, 0); }
	void addPartId(List<int> *partId) { addPartId(partId, 0); }
  private:
	void addPartId(List<int> *partId, int position);	
	bool doesIdExist(List<int> *partId, int position);
};

class MatrixPartMap {
  private:
        List<MatrixPart*> *aPartList;
	PartIdContainer *aIdContainer;
	int aSearchIndex;
        List<MatrixPart*> *bPartList;
	PartIdContainer *bIdContainer;
	int bSearchIndex;
        List<MatrixPart*> *cPartList;
	int cSearchIndex;
  public:
        MatrixPartMap();
        bool aPartExists(List<int*> *partId);
        void addAPart(MatrixPart *part);
	MatrixPart *getAPart(List<int*> *partId);
        bool bPartExists(List<int*> *partId);
        void addBPart(MatrixPart *part);
	MatrixPart *getBPart(List<int*> *partId);
        bool cPartExists(List<int*> *partId);
        void addCPart(MatrixPart *part) { cPartList->Append(part); }
	MatrixPart *getCPart(List<int*> *partId);
	MatrixPartMap *duplicate();
	void matchParts(MatrixPartMap *otherMap, std::ofstream &logFile);
  private:
	int getIdLocation(List<int*> *partId, List<MatrixPart*> *partList, int lastSearchedPlace);
};

void getNextLpu(int linearLpuId, 
		MMMLpu *lpuInstance, 
		IdGenerator *idGenerator, 
		MatrixPartMap *partMap);

#endif
