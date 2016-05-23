#include<iostream>
#include<string>

using namespace std;

class Person {
	public:
		string name;
		int age;
		void print() {
			cout << "\nName: " << name;
			cout << "\nAge: " << age;
		}
};

class Student {
	public:
		Person person;
		string school;
		int year;
		void print() {
			person.print();
			cout << "\nSchool: " << school;
			cout << "\nYear: " << year;
		}
};

void createAndAssignStudent(Student &student) {
	Person person;
	person.name = "Yan";
	person.age = 31;
	Student grad;
	grad.person = person;
	grad.school = "UVa";
	grad.year = 5;
	student = grad;
	cout << "\nInside the local function\n";
	student.print();
	person.age = 55;
	person.name = "Andrew";
	cout << "\nAgain inside the local function\n";
	grad.print();
}

int mainOb() {
	Student student;
	createAndAssignStudent(student);
	cout << "\nInside the main function\n";
	student.print();
	return 1;
}



