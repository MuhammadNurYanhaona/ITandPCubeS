#include <iostream>
#include <vector>

using namespace std;

class Employee {
private:
	int id;
	float salary;
public:
	Employee(int id, float salary) {
		this->id = id;
		this->salary = salary;
	}
	void describe() {
		cout << "Employee " << id << " earns " << salary << " per month\n";
	}
};

int mainVD() {

	vector<Employee*> *employeeVector = new vector<Employee*>;
	employeeVector->reserve(2);
	employeeVector->push_back(NULL);
	employeeVector->push_back(NULL);

	Employee *employee1 = new Employee(1, 10.105);
	(*employeeVector)[0] = employee1;
	Employee *employee2 = new Employee(2, 140.10);
	(*employeeVector)[1] = employee2;

	employeeVector->erase(employeeVector->begin());
	delete employeeVector;

	employee1->describe();
	employee2->describe();

	return 0;
}


