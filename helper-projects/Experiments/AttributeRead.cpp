#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cctype>
#include <string.h>
#include <stdio.h>

int mainAttrRead()  {
		std::string str = "Hello<yan> this is me <hamid><two><";

        int attrStart = -1;
        int position = 0;
        while ((attrStart = str.find('<', position)) != std::string::npos) {
                int attrEnd = str.find('>', attrStart);
                if (attrEnd != std::string::npos) {
                        position = attrEnd;
                        int attrLength = attrEnd - attrStart - 1;
                        std::string attr = str.substr(attrStart + 1, attrLength);
                        std::cout << attr << "\n";
                } else break;
        }
        return -1;
}


