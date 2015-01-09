#ifndef _H_tuple
#define _H_tuple

#include <iostream>
#include <vector>

class PEnvironment;
class PPartition;
class SEnvironment;
class SPartition;
class LUFEnvironment;
class LUFPartition;

class PEnvironment {
  public:
	int* p;
	float* u;
	float* v;
};

class PPartition {
  public:
	int n;
};

class SEnvironment {
  public:
	float* m;
	float* v;
};

class LUFEnvironment {
  public:
	float* a;
	float* u;
	float* l;
	int* p;
};

class LUFPartition {
  public:
	int s;
};

#endif
