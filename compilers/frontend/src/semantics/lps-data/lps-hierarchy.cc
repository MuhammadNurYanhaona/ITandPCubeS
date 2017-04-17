#include "../task_space.h"
#include "../symbol.h"
#include "../partition_function.h"
#include "../../common/constant.h"
#include "../../common/location.h"
#include "../../common/errors.h"
#include "../../syntax/ast.h"
#include "../../syntax/ast_def.h"
#include "../../static-analysis/usage_statistic.h"
#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/utils/hashtable.h"

#include <deque>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <cstdlib>

//-------------------------------------------- Partition Hierarchy -------------------------------------------------/

PartitionHierarchy::PartitionHierarchy() {
	spaceHierarchy = new Hashtable<Space*>;
}

Space *PartitionHierarchy::getSpace(char spaceId) {
	char *key = (char *) malloc(sizeof(char) * 2);
	key[0] = spaceId;
	key[1] = '\0';
	Space *space = spaceHierarchy->Lookup(key);
	free(key);
	return space;
}

Space *PartitionHierarchy::getSubspace(char spaceId) {
	int suffixLength = strlen(Space::SubSpaceSuffix);
	char *key = (char *) malloc(sizeof(char) * (suffixLength + 2));
	key[0] = spaceId;
	key[1] = '\0';
	strcat(key, Space::SubSpaceSuffix);
	Space *space = spaceHierarchy->Lookup(key);
	free(key);
	return space;
}

Space *PartitionHierarchy::getRootSpace() {
	return spaceHierarchy->Lookup(Space::RootSpaceName);
}

bool PartitionHierarchy::addNewSpace(Space *space) {
	Space *duplicateSpace = spaceHierarchy->Lookup(space->getName());
	bool duplicateFound = false;
	if (duplicateSpace != NULL) duplicateFound = true;
	spaceHierarchy->Enter(space->getName(), space, false);
	return !duplicateFound;
}

Space *PartitionHierarchy::getCommonAncestor(Space *space1, Space *space2) {
	if (space1 == space2) return space1;
	if (space1->isParentSpace(space2)) return space2;
	if (space2->isParentSpace(space1)) return space1;
	Space *nextSpace = space1;
	while ((nextSpace = nextSpace->getParent()) != NULL) {
		if (space2->isParentSpace(nextSpace)) return nextSpace;
	}
	return getRootSpace();
}
