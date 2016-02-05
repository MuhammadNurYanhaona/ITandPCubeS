#ifndef ALLOCATION_H_
#define ALLOCATION_H_

/* This is a subset of the allocation library of the segmented-memory compiler within the memory management module.
 * We included it here to develop the feature for communication buffers to operating memory data-parts interaction.
 * */

#include <iostream>
#include <cstdlib>
#include <vector>
#include "../utils/list.h"
#include "../structure.h"
#include "part_tracking.h"

/* This is an extension to the SuperPart construction within part-tracking library for linking list of data part
   allocations with their faster searching mechanism. This class can be avoided if the data part-list directly
   operates on the part container instead of using it as a mechanism to determine what part to return. We adopt
   this current strategy as the list based part allocation mechanism was developed earlier and the efficient search
   mechanism has been developed after we faced severe performance and memory problems in straightforward process.
   We did not want to change all the dependencies of the data-part-list construct; rather we only wanted to
   eliminate those problems.
*/
class PartLocator : public SuperPart {
  protected:
        int partListIndex;
  public:
        PartLocator(List<int*> *partId, int dataDimensions, int partListIndex)
                        : SuperPart(partId, dataDimensions) {
                this->partListIndex = partListIndex;
        }
        inline int getPartListIndex() { return partListIndex; }
};

/* This class holds all information to identify a part of a data structure configured for a particular LPS and
   to determine how to access/manipulate its content appropriately
*/
class PartMetadata {
  protected:
        // the number of dimensions in the data structure
        int dimensionality;
        // possibly multidimensional part Id specifying the index of the part within the whole; note that we
        // have a list here as partitioning in IT is hierarchical and for identification of a part in an LPS we
        // may need to identify its ancestor parts in higher LPSes
        List<int*> *idList;
        // spread of the part along different dimensions
        // note that even for data reordering partition functions we should have a contiguous spread for a data
        // part as then we will consider all indexes within the data has been transformed in a way to conform
        // with the reordering
        Dimension *boundary;
  public:
        PartMetadata(int dimensionality, List<int*> *idList, Dimension *boundary);
        int getSize();
        inline int getDimensions() { return dimensionality; }
        inline List<int*> *getIdList() { return idList; }
        inline Dimension *getBoundary() { return boundary; }
};

/* This class holds the meta-data and actual memory allocation for a part of a data structure
*/
class DataPart {
  protected:
        PartMetadata *metadata;
        void *data;
  public:
        DataPart(PartMetadata *metadata) {
                this->metadata = metadata;
                this->data = NULL;
        }
        template <class type> static void allocate(DataPart *dataPart) {
                int size = dataPart->metadata->getSize();
                dataPart->data = new type[size];
                char *charData = reinterpret_cast<char*>(dataPart->data);
                int charSize = size * sizeof(type) / sizeof(char);
                for (int i = 0; i < charSize; i++) {
                        charData[i] = 0;
                }
        }
        inline PartMetadata *getMetadata() { return metadata; }
        void *getData() { return data; }
        template <class type> static void print(DataPart *dataPart, std::ostream &stream) {
        	int size = dataPart->metadata->getSize();
        	type *typedData = reinterpret_cast<type*>(dataPart->getData());
        	for (int i = 0; i < size; i++) {
        		stream << typedData[i] << " ";
        	}
        	stream << "\n";
        }
};

#endif /* ALLOCATION_H_ */
