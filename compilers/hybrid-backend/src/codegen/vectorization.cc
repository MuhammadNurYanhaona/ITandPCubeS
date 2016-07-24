#include "vectorization.h"
#include "name_transformer.h"
#include "../utils/list.h"
#include "../utils/string_utils.h"

#include <iostream>
#include <sstream>
#include <string.h>
#include <cstdlib>

using namespace std;

const char *vectorizeLoops(const char *originalLoopHeaders, int indentLevel) {

	// currently we have the loop vectorization logic implemented for GPU platform only
	ntransform::NameTransformer *transformer 
			= ntransform::NameTransformer::transformer;
	ntransform::HybridNameTransformer *hybridTransformer 
			= dynamic_cast<ntransform::HybridNameTransformer*>(transformer);
	if (hybridTransformer == NULL 
			|| !hybridTransformer->isGpuMode()) {
		return originalLoopHeaders;
	}

	std::ostringstream newLoopHeaders;

	// add all but the last loop as they are in the result stream; currently we are only
	// vectorizing the last loop
	string oldLoopStr = string(originalLoopHeaders);
	int lastForLoopIndex = oldLoopStr.rfind("for (");
	string upperLoops = oldLoopStr.substr(0, lastForLoopIndex);
	newLoopHeaders << upperLoops;

	// extract the loop index iteration configuration from the last loop
	string targetLoop = oldLoopStr.substr(lastForLoopIndex);
	int targetHeaderEndIndex = targetLoop.find(") {");
	string targetHeader = targetLoop.substr(0, targetHeaderEndIndex);
	string targetBody = targetLoop.substr(targetHeaderEndIndex);

	// separate the three parts in the for loop header
	int indexInitIndex = targetHeader.find(";");
	string indexInitializer = targetHeader.substr(0, indexInitIndex);
	string remaining = targetHeader.substr(indexInitIndex + 1);
	int indexBoundIndex = remaining.find(";");
	string indexBoundChecker = remaining.substr(0, indexBoundIndex);
	string indexStepper = remaining.substr(indexBoundIndex + 1);

	// add a constant step increase to the initializer expression so that different CUDA
	// threads may start at different indexes
	newLoopHeaders << indexInitializer << " + threadId;";
	// keep the bound checking intact
	newLoopHeaders << indexBoundChecker << ";";
	// multiply the stepper increment by the number of threads per warp so that each thread
	// has a distinct index to operate on in each iteration
	newLoopHeaders << indexStepper << " * WARP_SIZE";
	
	// finally add the body of the last loop in the new stream
	newLoopHeaders << targetBody;

	return strdup(newLoopHeaders.str().c_str());
}
