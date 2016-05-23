#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cctype>
#include "utils.h"
#include <stdio.h>
#include <string.h>

int getLastIndexOf(const char *str, char ch) {
	int length = strlen(str);
	int lastIndex = -1;
	for (int i = 0; i < length; i++) {
		if (str[i] == ch) lastIndex = i;
	}
	return lastIndex;
}

char *substr(const char *str, int begin, int end) {
	int length = end - begin + 1;
	char *buffer = new char[length + 1];
	const char *source = str + begin;
	strncpy(buffer, source, length);
	return buffer;
}

int mainLIOS() {
	const char *path = "/home/yan/file.txt";
	int lastSlash = getLastIndexOf(path, '/');
	char *subs = substr(path, lastSlash + 1, strlen(path));
	std::cout << subs;
	return 0;
}



