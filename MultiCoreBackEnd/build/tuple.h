#ifndef _H_tuple
#define _H_tuple

#include <iostream>
#include <vector>

class MMEnvironment;
class MMPartition;

class MMEnvironment {
  public:
	float* a;
	float* b;
	float* c;
};

class MMPartition {
  public:
	int k;
	int l;
	int q;
};

#endif
