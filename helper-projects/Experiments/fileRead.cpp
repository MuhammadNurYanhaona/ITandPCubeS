#include <iostream>
#include <fstream>
#include <string>
using namespace std;

int main0 () {
	string line;
	std::ifstream myfile("/home/yan/pcubes.ml");

	if (myfile.is_open()) {
		while ( getline (myfile,line) ) {
			cout << line << '\n';
		}
		myfile.close();
	}

	else cout << "Unable to open file";

	return 0;
}
