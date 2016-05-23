#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>

using namespace std;

int getTag(int communicatorId,
		int senderId,
		int receiverId, int digitsInId) {

	ostringstream ostream;
	ostream << communicatorId;
	ostream << std::setfill('0') << std::setw(digitsInId) << senderId;
	ostream << std::setfill('0') << std::setw(digitsInId) << receiverId;
	istringstream istream(ostream.str());
	int tag;
	istream >> tag;
	return tag;
}

int mainTA() {
	int segmentCount = 100;
	ostringstream digitStr;
	digitStr << segmentCount;
	int digitsForId = digitStr.str().length();
	int tag = getTag(5, 1, 12, digitsForId);
	std::cout << "Tag: " << tag << "\n";
}
