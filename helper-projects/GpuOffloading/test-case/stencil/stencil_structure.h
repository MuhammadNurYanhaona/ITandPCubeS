#ifndef _H_stencil_structure
#define _H_stencil_structure

#include "../../runtime/structure.h"
#include "../../utils/list.h"
#include "../../utils/hashtable.h"

#include <vector>
#include <fstream>

// This namespace contains the classes that we dynamically generate for individual task to track scalar variables' state,
// dimensionality of the originial data structures used inside a task, and task partition information. All of these may be
// needed during the execution of a compute stage.
namespace stencil {

        class Partition {
          public:
                int blockCount;
		int blockSize;
		int padding1;
		int padding2;
        };

        class ArrayMetadata {
          public:
                Dimension plateDims[2];
        };

        class TaskGlobals {
          public:
		int iterations;
        };

        class ThreadLocals {
          public:
		int currIteration;
        };

	// this generator is for the scenario where the GPU can hold an entire Space A LPU	
	class IdGenerator {
          private:
                int lpuCount;
          public:
                IdGenerator(int lpuCount);
                List<int*> *getPartId(int linearLpuId);
        };

	class PlatePart {
          private:
                double *data;
		double *data_lag_1;
		int versionHead;
	  public:
                Dimension storageDims[2];
                List<int*> *partId;
          public:
                PlatePart() {
                        data = NULL;
			data_lag_1 = NULL;
                        partId = NULL;
			versionHead = 0;
                }
		void setData(double *data, int version);
		double *getData(int version);
		void advanceEpoch();	
                void duplicate(PlatePart *copy);
                bool sameContent(PlatePart *other);
        };
	
	class StencilLpu : public LPU {
	  public:
		PlatePart *partReference;
		double *plate;
		double *plate_lag_1;
                List<int*> *platePartId;
                PartDimension platePartDims[2];
          public:
                StencilLpu() : LPU() {
			partReference = NULL;
                        plate = NULL;
			plate_lag_1 = NULL;
                        platePartId = NULL;
                }
	};

	class PlatePartGenerator {
          private:
                int lpuCount;
                int padding;
                Dimension *plateDims;
          public:
                PlatePartGenerator(int lpuCount, int padding, Dimension *plateDims);
                PlatePart *generatePart(List<int*> *partId);
        };

	class PartIdContainer {
          protected:
                std::vector<int> partArray;
          public:
                PartIdContainer() {}
                bool doesIdExist(List<int> *partId);
                void addPartId(List<int> *partId);
        };
	
	class PlatePartMap {
          private:
                List<PlatePart*> *partList;
                PartIdContainer *idContainer;
          public:
                PlatePartMap();
                bool partExists(List<int*> *partId);
                void addPart(PlatePart *part);
                PlatePart *getPart(List<int*> *partId);
		List<PlatePart*> *getPartList() { return partList; }
                PlatePartMap *duplicate();
                void matchParts(PlatePartMap *otherMap, std::ofstream &logFile);
        };

	void getNextLpu(int linearLpuId,
                        stencil::StencilLpu *lpuInstance,
                        stencil::IdGenerator *idGenerator,
                        stencil::PlatePartMap *partMap);
}

#endif
