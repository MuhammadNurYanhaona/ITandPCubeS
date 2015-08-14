#ifndef _H_structure
#define _H_structure

class Range {
public:
        int min;
        int max;
        Range() {
        	min = 0;
        	max = 0;
        }
        Range(int index) {
        	min = index;
        	max = index;
        }
        Range(int min, int max) {
        	this->min = min;
        	this->max = max;
        }
        bool isEqual(Range other) {
        	return (this->min == other.min && this->max == other.max);
        }
};

class Dimension {
public:
        int length;
        Range range;
        Dimension() {
        	length = 0;
        	range = Range();
        }
        void setLength() {
        	length = range.max - range.min + 1;
        }
};

#endif



