#ifndef _H_mmm_structure
#define _H_mmm_structure

/* This header file has classes that serve as a replacement for the LPU and data part generation mechanism we have in the
 * compilers. This matrix-matrix multiplication and fixed partition configuration specific classes are written as porting
 * the existing LPU and part generation libraries in this project would be to much hassle.
 */

#include "../runtime/structure.h"
#include "../utils/list.h"
#include "../utils/hashtable.h"

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

class MatrixPartMap {
  private:
        List<MatrixPart*> *aPartList;
        List<MatrixPart*> *bPartList;
        List<MatrixPart*> *cPartList;
  public:
        MatrixPartMap();
        bool aPartExists(List<int*> *partId) { return isIdInList(partId, aPartList); }
        void addAPart(MatrixPart *part) { aPartList->Append(part); }
	MatrixPart *getAPart(List<int*> *partId) { return getPart(partId, aPartList); }
        bool bPartExists(List<int*> *partId) { return isIdInList(partId, bPartList); }
        void addBPart(MatrixPart *part) { bPartList->Append(part); }
	MatrixPart *getBPart(List<int*> *partId) { return getPart(partId, bPartList); }
        bool cPartExists(List<int*> *partId) { return isIdInList(partId, cPartList); }
        void addCPart(MatrixPart *part) { cPartList->Append(part); }
	MatrixPart *getCPart(List<int*> *partId) { return getPart(partId, cPartList); }
	MatrixPartMap *duplicate();
	void matchParts(MatrixPartMap *otherMap, std::ofstream &logFile);
  private:
	bool isIdInList(List<int*> *partId, List<MatrixPart*> *partList);
	MatrixPart *getPart(List<int*> *partId, List<MatrixPart*> *partList);
};

void getNextLpu(int linearLpuId, 
		MMMLpu *lpuInstance, 
		IdGenerator *idGenerator, 
		MatrixPartMap *partMap);

#endif
