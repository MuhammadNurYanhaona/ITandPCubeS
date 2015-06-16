#ifndef _H_environment_base
#define _H_environment_base

/* This header file content the environment base class that holds information regarding I/O operations of various
   arrays that are part of a task environment. Task specific environment classes should extend this base class */

#include "../utils/hashtable.h"
#include "../codegen/structure.h"
#include "stream.h" 	

class EnvironmentBase {
  protected:
	Hashtable<const char*> *inputBindings;
	Hashtable<const char*> *outputBindings;

  public:
	EnvironmentBase() {
		inputBindings = new Hashtable<const char*>;
		outputBindings = new Hashtable<const char*>;
	}
	
	inline void bindInput(const char *varName, const char *fileName) {
		inputBindings->Enter(varName, fileName);
	}
	inline void bindOutput(const char *varName, const char *fileName) {
		outputBindings->Enter(varName, fileName);
	}

	inline bool hasInputBinding(const char *varName) { 
		return inputBindings->Lookup(varName) != NULL; 
	}
	inline bool hasOutputBinding(const char *varName) { 
		return outputBindings->Lookup(varName) != NULL; 
	}

	// a helper method to initialize array dimensions information when the data for the array come from a file	
	inline void readDimensionInfo(const char *varName, Dimension *dimension) {
		const char *inputFile = inputBindings->Lookup(varName);
		TypedInputStream<char> *stream = new TypedInputStream<char>(inputFile);
		stream->copyDimensionInfo(dimension);
		delete stream;
	}

	inline const char *getInputFileForStructure(const char *varName) {
		return inputBindings->Lookup(varName);
	}
	inline const char *getOutputFileForStructure(const char *varName) {
		return outputBindings->Lookup(varName);
	}

	// these are inefficient implementations; we should rather clear the content of existing maps as opposed
	// to creating new maps to avoid wasting space 
	inline void resetInputBindings() { 
		inputBindings = new Hashtable<const char*>; 
	}
	inline void resetOutputBindings() { 
		outputBindings = new Hashtable<const char*>; 
	}
	void resetBindings() {
		resetInputBindings();
		resetOutputBindings();
	}
};

#endif
