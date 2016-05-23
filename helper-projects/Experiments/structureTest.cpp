/*
 * structureTest.cpp
 *
 *  Created on: Dec 26, 2014
 *      Author: yan
 */
#include <iostream>

typedef struct {
	bool good;
	int count;
} Eggs;

Eggs **getSomeEggs(int eggsCount);

int mainSt() {
	Eggs **eggs = getSomeEggs(10);
	for (int i = 0; i < 10; i++) {
		std::cout << eggs[i]->count << std::endl;
	}
	return 0;
}

Eggs **getSomeEggs(int eggsCount) {
	Eggs **eggs = new Eggs*[eggsCount];
	for (int i = 0; i < eggsCount; i++) {
		eggs[i] = new Eggs;
		eggs[i]->good = true;
		eggs[i]->count = i + 1;
	}
	return eggs;
}


