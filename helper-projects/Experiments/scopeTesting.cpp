#include <iostream>

int mainScopeTesting() {

	{ int iterationNo = 0;
	  for (int i = 0; i < 3; i++) {
		 iterationNo++;
		 { int iterationNo = 0;
			 for (int i = 0; i < 5; i++) {
				 iterationNo++;
				 std::cout << "Iteration No Inside: " << iterationNo << "\n";
			 }
		 }
		 std::cout << "Iteration No Outside: " << iterationNo << "\n";
	  }
	}
	return 0;
}



