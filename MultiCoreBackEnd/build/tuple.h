#ifndef _H_tuple
#define _H_tuple

#include <iostream>
#include <vector>

class LUFEnvironment;
class LUFPartition;

class LUFEnvironment {
  public:
	float* a;
	float* u;
	float* l;
	int* p;
};

class LUFPartition {
  public:
};

#endif
