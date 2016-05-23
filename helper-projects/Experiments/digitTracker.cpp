#include <math.h>
#include <iostream>
#include <cstdlib>

using namespace std;

int countDigits (int n) {
    if (n == 0) return 1;
    return floor (log10(abs (n))) + 1;
}

int mainDT() {
	cout << "Digits in 10434 is " << countDigits(10434) << "\n";
	cout << "Digits in 2 is " << countDigits(2) << "\n";
	cout << "Digits in 434 is " << countDigits(434) << "\n";
	return 0;
}



