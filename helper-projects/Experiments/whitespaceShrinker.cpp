#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cctype>
#include "utils.h"

void shrinkWhitespaces(std::string &str) {
        for (int i = 0; i < str.length(); i++) {
                if (!std::isspace(str[i])) continue;
                str.replace(i, 1, " ");
                int j = i + 1;
                while (j < str.length()) {
                        if (std::isspace(str[j])) {
                                str.erase(j, 1);
                        } else break;
                }
        }
}

int main5() {

	std::string str = "  hello Mr   Jack. 	Can I 		Help you in any 	way? ";
	shrinkWhitespaces(str);
	std::cout << "|" << str << "|";
}



