#include <iostream>

using namespace std;

void assignPointer(char **destination, double *sourceArray, int position) {
	*destination = reinterpret_cast<char*>(sourceArray + position);
}

int mainPS() {

	double *array1 = new double[10];
	double *array2 = new double[10];
	double *array3 = new double[10];

	for (int i = 0; i < 10; i++) array1[i] = i + 1 + 0.0;
	for (int i = 0; i < 10; i++) array2[i] = i * i;
	for (int i = 0; i < 10; i++) array3[i] = i * i * i;

	cout << "Array 1: ";
	for (int i = 0; i < 10; i++) cout << array1[i] << " ";
	cout << "\nArray 2: ";
	for (int i = 0; i < 10; i++) cout << array2[i] << " ";
	cout << "\nArray 3: ";
	for (int i = 0; i < 10; i++) cout << array3[i] << " ";

	char **transferArray = new char*[3];
	assignPointer(&transferArray[0], array1, 5);
	assignPointer(&transferArray[1], array2, 5);
	assignPointer(&transferArray[2], array3, 5);

	cout << "\nTransfer Array Content: ";
	for (int i = 0; i < 3; i++) {
		double *element = reinterpret_cast<double*>(transferArray[i]);
		cout << *element << " ";
	}

	for (int i = 0; i < 3; i++) {
		double *element = reinterpret_cast<double*>(transferArray[i]);
		*element += 5;
	}

	cout << "\nAfter update through the transfer buffer\n";
	cout << "Array 1: ";
	for (int i = 0; i < 10; i++) cout << array1[i] << " ";
	cout << "\nArray 2: ";
	for (int i = 0; i < 10; i++) cout << array2[i] << " ";
	cout << "\nArray 3: ";
	for (int i = 0; i < 10; i++) cout << array3[i] << " ";

	return 0;
}
