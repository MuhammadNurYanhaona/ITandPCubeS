#include <iostream>
#include <cstdlib>
#include <vector>
#include "../utils/list.h"
#include "../structure.h"
#include "part_tracking.h"
#include "allocation.h"

PartMetadata::PartMetadata(int dimensionality, List<int*> *idList, Dimension *boundary) {
        this->dimensionality = dimensionality;
        this->idList = idList;
        this->boundary = boundary;
}

int PartMetadata::getSize() {
        int size = 1;
        for (int i = 0; i < dimensionality; i++) {
                size *= boundary[i].length;
        }
        return size;
}
