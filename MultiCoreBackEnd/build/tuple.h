#ifndef _H_tuple
#define _H_tuple

#include <iostream>
#include <vector>

class ValueCoordinatePair;
class COOMVMEnvironment;
class COOMVMPartition;

class ValueCoordinatePair {
  public:
	float value;
	int row;
	int column;
};

class COOMVMEnvironment {
  public:
	ValueCoordinatePair* m;
	float* v;
	float* w;
};

class COOMVMPartition {
  public:
	int p;
	int r;
};

#endif
