#ifndef _H_properties
#define _H_properties

#include "hashtable.h"

// this class holds all the properties read from a single property file
class Properties {
  protected:
	Hashtable<const char*> *propertyMap;
  public:
	Properties();
	~Properties();
	const char *getProperty(const char *key);
	void putProperty(const char *key, const char *value); 	
};

// a reader class that reads and groups properties from different sources
class PropertyReader {
  public:
	// the static reference to be used to retrieve the properties found in a single file
	static Hashtable<Properties*> *propertiesGroups;
  public:
	static void readPropertiesFile(const char *filePath, const char *groupKey);	   	
};

#endif
