#include<iostream>
#include<string>

using namespace std;

class Region {
public:
	string name;
	long area;
	void print() {
		cout << "\nRegion: " << name;
		cout << "\nArea: " << area;
	}
};

void assignArray(Region regions[2]) {
	Region localRegions[2];
	localRegions[0].name = "Dhaka";
	localRegions[0].area = 1000l;
	localRegions[1].name = "Charlottesville";
	localRegions[1].area = 90l;
	regions[0] = localRegions[0];
	regions[1] = localRegions[1];
}

int mainArrA() {
	Region twoRegions[2];
	assignArray(twoRegions);
	twoRegions[0].print();
	twoRegions[1].print();
	return 1;
}



