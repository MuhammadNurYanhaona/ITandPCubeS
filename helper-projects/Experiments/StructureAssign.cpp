#include <iostream>

class Flower {
public:
	std::string name;
	int size;
	std::string color;

	Flower(std::string name, int size, std::string color) {
		this->name = name;
		this->size = size;
		this->color = color;
	}

	void print() {
		std::cout << "Name: " << name << std::endl;
		std::cout << "Size: " << size << std::endl;
		std::cout << "Color: " << color << std::endl;
	}
};

Flower getAFlower(bool readOrWhite) {
	Flower tulip = Flower("Tulip", 10, "Red");
	Flower rhododendron = Flower("Rhododendron", 5, "White");
	if (readOrWhite) return tulip;
	return rhododendron;
}

int mainStructAssign() {
	Flower flower = getAFlower(true);
	flower.print();
	return 0;
}



