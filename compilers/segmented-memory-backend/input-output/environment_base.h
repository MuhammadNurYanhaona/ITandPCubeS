#ifndef _H_environment_base
#define _H_environment_base

/* This header file content the environment base class that holds information regarding I/O operations of various
   arrays that are part of a task environment. Task specific environment classes should extend this base class */

#include "../utils/hashtable.h"
#include "../codegen/structure.h"
#include "stream.h" 

#include <mpi.h>
#include <fstream>	

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

	// We have not yet discussed/solved how to do parallel file outputs. As an impromptu solution we are doing
	// serialized writes. The segments will write their portions of data one after another in their original
	// ordering. For that to work properly, there need to be a signaling from the earlier segment to the later
	// segment for the latter to begin its writing. This function does the signal and wait interaction for that.   
	void getReadyForOutput(int segmentId, 
			int segmentCount, 
			MPI_Comm communicator, std::ofstream &logFile);
	
	// This is the counter-part of the above to indicate that the next segment can start writing its structures. 
	void signalOutputCompletion(int segmentId, 
			int segmentCount, 
			MPI_Comm communicator, std::ofstream &logFile);

	// a helper method to initialize array dimensions information when the data for the array come from a file	
	inline void readDimensionInfo(const char *varName, PartDimension *partDims) {
		const char *inputFile = inputBindings->Lookup(varName);
		TypedInputStream<char> *stream = new TypedInputStream<char>(inputFile);
		stream->copyDimensionInfo(partDims);
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
