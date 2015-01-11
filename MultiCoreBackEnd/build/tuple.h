#ifndef _H_tuple
#define _H_tuple

#include <iostream>
#include <vector>

class Point;
class Rectangle;
class Coefficients;
class MCAEEnvironment;
class MCAEPartition;

class Point {
  public:
	float x;
	float y;
};

class Rectangle {
  public:
	int top;
	int right;
	int bottom;
	int left;
};

class Coefficients {
  public:
	int order;
	float values[2];
};

class MCAEEnvironment {
  public:
	Rectangle* grid;
	std::vector<Coefficients> shape;
	float area;
};

class MCAEPartition {
  public:
	int p;
};

#endif
