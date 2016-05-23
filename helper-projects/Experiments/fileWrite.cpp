#include <fstream>
#include <iostream>
using namespace std;

int main2 () {
  ofstream myfile ("/home/yan/example.txt");
  if (myfile.is_open()) {
    myfile << "This is a line.\n";
    myfile << "This is another line.\n";
    myfile.close();
  }
  else cout << "Unable to open file";
  return 0;
}



