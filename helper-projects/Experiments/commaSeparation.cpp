#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cctype>
#include "utils.h"

int mainComma() {
	int b = 0;
	int a;
	if ((a = 10, a > 5) && (b = a, b > 9)) {
		std::cout << "B - " << b;
	}

	int array[10];
	for (int i = 0; i < 10; i++) array[i] = i;
	int index = 0;
	std::cout << "\nArray access: ";
	std::cout << array[(index = index + 1, index++, index)];
	std::cout << " index " << index;
	return 0;
}



