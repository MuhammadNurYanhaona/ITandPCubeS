#include <iostream>

void printEntries(int *array, int count) {
	for (int i = 0; i < count; i++) {
		std::cout << array[i] << '\t';
	}
}

int mainPFSA() {
	int myArray[3];
	myArray[0] = 1;
	myArray[1] = 10;
	myArray[2] = 99;
	printEntries(myArray, 3);
	return 0;
}



