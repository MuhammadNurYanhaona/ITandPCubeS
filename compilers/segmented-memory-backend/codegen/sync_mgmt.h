#ifndef _H_sync_mgmt
#define _H_sync_mgmt

// This header file is for the class that generates all synchronization variables and structures,
// and generates the initialization functions for those variables and structures too for a single
// task found in the IT source code.

#include "../syntax/ast_task.h"
#include "../static-analysis/sync_stat.h"
#include "../utils/list.h"

class SyncManager {
  protected:
	TaskDef *taskDef;
	const char *headerFile;
	const char *programFile;
	const char *initials;
	List<SyncRequirement*> *taskSyncList;
  public:
	SyncManager(TaskDef *taskDef, 
			const char *headerFile, 
			const char *programFile, 
			const char *initials);
	
	// this function investigates the computation block of a task and extracts the list of
	// synchronization primitives
	void processSyncList();

	// tells if the task under concern requires synchronization or not
	bool involvesSynchronization();

	// this function generates and writes the array of sync primitives in the header file 
	// for the task
	void generateSyncPrimitives();

	// this functions generates the method that the task::main function uses to initialze
	// sync primitive arrays and each primitive inside those arrays
	void generateSyncInitializerFn();

	// this function generates a data structure that holds pointers to different sync 
	// primitives that a thread, which works as a composite PPU, needs to wait/signal on 
	// different phases of its execution of thread::run function
	void generateSyncStructureForThreads();

	// this function generates the routine that the thread::run function invokes at the 
	// beginning to get in SyncDataStructure described above with all pointers references set
	// appropriately for the calling thread
	void generateFnForThreadsSyncStructureInit();  		
};

#endif
