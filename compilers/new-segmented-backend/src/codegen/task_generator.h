#ifndef _H_task_generator
#define _H_task_generator

#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include <stdio.h>

#include "sync_mgmt.h"

#include "../../../common-libs/utils/list.h"
#include "../../../frontend/src/syntax/ast_task.h"
#include "../../../frontend/src/syntax/ast_type.h"

class PPS_Definition;
class MappingNode;

// This is basically a coordinator class that generates data structures, constants, and functions
// corresponding to a single IT task in matching header and program files. It is a coordinator as 
// most of these functionalities are implemented by calling other code generation utility libraries 
// with appropriate parameters.
class TaskGenerator {
  private:
	TaskDef *taskDef;
	const char *processorFile;
	const char *mappingFile;
	const char *headerFile;
	const char *programFile;
	const char *initials;
	MappingNode *mappingRoot;
	SyncManager *syncManager;
	int segmentedPPS;
	bool involveReduction;
  public:
	TaskGenerator(TaskDef *taskDef, 
		const char *outputDirectory, 
		const char *mappingFile,
		const char *processorFile);

	const char *getHeaderFile() { return headerFile; }
	const char *getProgramFile() { return programFile; }
	TaskDef *getTaskDef() { return taskDef; }
	const char* getTaskName() { return taskDef->getName(); }
	const char* getInitials() { return initials; }
	static const char *getHeaderFileName(TaskDef *taskDef);
	static const char *getNamespace(TaskDef *taskDef);
	SyncManager *getSyncManager() { return syncManager; }
	bool hasCommunicators();
	bool hasReductions() { return involveReduction; }

	// function to generate all data structures and methods that are relevant to this task 
	// including a thread run function to run the task as a parallel program in multiple threads
	void generate(List<PPS_Definition*> *pcubesConfig);

	// a supporting function for task executor that initiates the environment links; it returns 
	// the list of external links to aid the calculation of other helper functions
	List<const char*> *initiateEnvLinks(std::ofstream &programFile);
	// a helper function for I/O that check and display messages if some object type is currently
	// unsupported by the I/O routine
	bool isUnsupportedInputType(Type *type, const char *varName);
	// a function for initializing partition related data structures
	void readPartitionParameters(std::ofstream &stream);
	// a function to be used by task-invocator library to copy parition parameters from partition
	// object to an array of integer, the form used within thread-state; TODO we should change
	// the thread-state implementation in the future to avoid this
	void copyPartitionParameters(std::ofstream &stream);
	// a supporting function for task main that gets input initialization parameters and invokes
	// task initialization function
	void inovokeTaskInitializer(std::ofstream &stream, 
			List<const char*> *externalEnvLinks, 
			bool skipArgInitialization = false);
	// a supporting function for generating an array of thread-state objects, one for each thread,
	// then initializing them	
	void initiateThreadStates(std::ofstream &stream);
	// a supporting function to group threads into segments; a process will run only the threads
	// of its own segment but it will need to investigate other groups to determine where and 
	// what data to communicate	
	void performSegmentGrouping(std::ofstream &stream, bool segmentIdPassed = false);
	// a supporting function to initialize all data parts of different structures that will be 
	// needed in computations carried by the threads of the segment managed by the current process 
	void initializeSegmentMemory(std::ofstream &stream);
	// a supporting function to create a map of data communicators that will be used to resolve
	// data dependencies within the task that requires communications between segments
	bool generateCommunicators(std::ofstream &stream);
	// a supporting function that starts threads once initialization is done for all necessary 
	// data	structures
	void startThreads(std::ofstream &stream);
	// a supporting function that generates prompts and codes for writing results of computations
	// to external files 
	void writeResults(std::ofstream &stream); 		
};

#endif

