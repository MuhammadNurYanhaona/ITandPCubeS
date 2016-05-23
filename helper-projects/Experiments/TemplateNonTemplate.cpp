#include <iostream>
using namespace std;

template <class type> class Student {
	type t;
 public:
	Student(type t) { this->t = t; }
	void print() {
		cout << "I am " << t << '\n';
	}
};

int mainTNT() {
	Student<float> st1 = Student<float>(1.1);
	Student<int> st2 = Student<int>(5);
	st1.print();
	st2.print();
//	Student *arr = &st2;
//	arr->print();
	return 1;
}



