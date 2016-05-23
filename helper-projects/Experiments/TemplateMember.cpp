#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cctype>
#include <stdlib.h>
#include <string.h>
#include <deque>
#include "utils.h"
#include "structures.h"
#include <iostream>
#include <stdlib.h>
#include <time.h>

class Student {
private:
	void *grades;
public:
	Student() { grades = NULL; }
	template <class type> static void allocate(Student *student, int count) {
		type *t = new type[count];
		student->grades = (void *) t;
	}
	void *getGrades() {
		return grades;
	}
	template <class type> static void printGrades(Student *student, int count) {
		type *tGrades = (type *) student->grades;
		for (int i = 0; i < count; i++) {
			std::cout << tGrades[i] << std::endl;
		}
	}
};

int mainTM() {
	Student *student = new Student;
	Student::allocate<float>(student, 5);
	float *grades = (float *) student->getGrades();
	for (int i = 0; i < 5; i++) {
		grades[i] = i;
	}
	Student::printGrades<float>(student, 5);
	return 0;
}



