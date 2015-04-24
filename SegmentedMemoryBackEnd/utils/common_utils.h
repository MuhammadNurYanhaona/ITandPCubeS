#ifndef _H_common_utils
#define _H_common_utils

#include "../utils/list.h"

inline int min(int x, int y) { return x < y ? x : y; }

inline int max(int x, int y) { return x > y ? x : y; }

namespace common_utils {

	bool isStringInList(const char *str, List<const char*> *list);
}

#endif

