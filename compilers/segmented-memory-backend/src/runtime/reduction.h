#ifndef _H_reduction
#define _H_reduction

namespace reduction {

	typedef union {
		bool boolValue;
		char charValue;	
		int intValue;
		float floatValue;
		double doubleValue;
		long longValue;
	} Data;

	typedef struct {
		reduction::Data data; 
		unsigned int index;
	} Result;

}

#endif
