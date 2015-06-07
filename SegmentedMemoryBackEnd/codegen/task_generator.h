#ifndef _H_task_generator
#define _H_task_generator

#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include <stdio.h>

#include "sync_mgmt.h"
#include "../syntax/ast_task.h"
#include "../syntax/ast_type.h"
#include "../utils/list.h"

class PPS_Definition;
class MappingNode;

// This is basically a coordinator class that generates data structures, constansts, and functions
// corresponding to a single IT task in matching header and program file. It is a coordinator as
// most of these functionalities are implemented by calling other code generation utility libraries
// with appropriate parameters. The only function that it generates on its own is a main function
// correspond to the task.

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
	
	// parameter that indicates a main function needs to be generated for this task automatically
	// as there are no controller program invoking and controlling tasks in the source code
	bool isolatedTask;
  public:
	TaskGenerator(TaskDef *taskDef, 
		const char *outputDirectory, 
		const char *mappingFile,
		const char *processorFile,bool isolatedTask);
	const char *getHeaderFile() { return headerFile; }
	const char *getProgramFile() { return programFile; }
	TaskDef *getTaskDef() { return taskDef; }
	const char* getTaskName() { return taskDef->getName(); }
	const char* getInitials() { return initials; }
	static const char *getHeaderFileName(TaskDef *taskDef);
	static const char *getNamespace(TaskDef *taskDef);
	SyncManager *getSyncManager() { return syncManager; }

	// function to generate all data structures and methods that are relevant to this task 
	// including a thread run function to run the task as a parallel program in multiple threads
	void generate(List<PPS_Definition*> *pcubesConfig);
	// function to generate a main function for the task in the absense of any coordinator 
	// program in the IT source file
	void generateTaskMain();

	// a supporting function for task main that initiates the environment links based on input
	// from console; it returns the list of external links to aid the calculation of other helper
	// functions
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
	// a supporting function for task main that get input initialization parameters and invoke
	// tasks initialization function
	void inovokeTaskInitializer(std::ofstream &stream, 
			List<const char*> *externalEnvLinks, 
			bool skipArgInitialization = false);
	// a supporting function for generating an array of thread-state objects, one for each thread,
	// then initializing them	
	void initiateThreadStates(std::ofstream &stream);
	// a supporting function to groups threads into segments; a process will run only the threads
	// of its own segment but it will needs to investigate other groups to determine where and 
	// what data to communicate	
	void performSegmentGrouping(std::ofstream &stream);
	// a supporting function that starts threads once initialization is done for all necessary 
	// data	structures
	void startThreads(std::ofstream &stream);
	// a supporting function that generates prompts and codes for writing results of computations
	// to external files 
	void writeResults(std::ofstream &stream); 		
};

#endif

