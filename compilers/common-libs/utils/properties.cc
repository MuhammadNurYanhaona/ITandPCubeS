#include "properties.h"
#include "hashtable.h"
#include "list.h"
#include "string_utils.h"

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string.h>

using namespace std;

//--------------------------------------------------------- Properties ---------------------------------------------------------/

Properties::Properties() {
	propertyMap = new Hashtable<const char*>;
}

Properties::~Properties() {
	delete propertyMap;
}

const char *Properties::getProperty(const char *key) { 
	return propertyMap->Lookup(key); 
}

void Properties::putProperty(const char *key, const char *value) {
	propertyMap->Enter(key, value);
}

//------------------------------------------------------- Properties Reader ----------------------------------------------------/

Hashtable<Properties*> *PropertyReader::propertiesGroups = new Hashtable<Properties*>;

void PropertyReader::readPropertiesFile(const char *filePath, const char *groupKey) {
	
	string propertyLine;
	ifstream propertiesFile(filePath);
	string delimiter = "=";

	Properties *propertyGroup = new Properties();

	if (!propertiesFile.is_open()) {
		cout << "Unable to open properties file: " << filePath << "\n";
		cout << "Compilation will be done based on default properties\n";
	} else {
		while ( getline (propertiesFile,propertyLine) ) {
			List<string> *tokenList = string_utils::tokenizeString(propertyLine, delimiter);
			string key = tokenList->Nth(0);
			string_utils::trim(key);
			string value = tokenList->Nth(1);
			string_utils::trim(value);
			propertyGroup->putProperty(strdup(key.c_str()), strdup(value.c_str()));
			delete tokenList;
		}
		propertiesFile.close();
	}
		
	PropertyReader::propertiesGroups->Enter(groupKey, propertyGroup);
}
