#ifndef _H_common_utils
#define _H_common_utils

#include <math.h>
#include <cstdlib>

#include "list.h"

inline int min(int x, int y) { return x < y ? x : y; }

inline int max(int x, int y) { return x > y ? x : y; }

inline int gcd(int a, int b) {
	while (a != b) {
		if (a > b) a = a - b;
		else b = b - a;
	}
	return a;
}

inline int lcm(int a, int b) {
	return (a * b) / gcd(a, b);
}

inline int countDigits (int n) {
    if (n == 0) return 1;
    return floor (log10(abs (n))) + 1;
}

namespace common_utils {

	bool isStringInList(const char *str, List<const char*> *list);
}

#endif

