#include "common_utils.h"
#include "list.h"

#include <iostream>
#include <cstdlib>
#include <string.h>

bool common_utils::isStringInList(const char *str, List<const char*> *list) {
	if (str == NULL || list == NULL) return false;
	for (int i = 0; i < list->NumElements(); i++) {
		if (strcmp(str, list->Nth(i)) == 0) return true;
	}
	return false;
}
