#ifndef _H_semantic_helper
#define _H_semantic_helper

/* This header file contains the definitions of all auxiliary classes that are needed at various
   sub-phases of semantic analysis but aren't required after the underlying sub-phases are done
*/

#include "../lex/scanner.h"

namespace semantic_helper {

	class ArrayDimConfig {
	  private:
		const char *name;
		int dimensions;
	  public:
		ArrayDimConfig(const char *name, int dimensions) {
			this->name = name;
			this->dimensions = dimensions;
		}
		const char *getName() { return name; }
		int getDimensions() { return dimensions; }	
	};

};

#endif
