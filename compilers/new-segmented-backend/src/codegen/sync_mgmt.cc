#include "sync_mgmt.h"
#include "code_constant.h"

#include "../../../frontend/src/syntax/ast_task.h"
#include "../../../frontend/src/semantics/data_access.h"
#include "../../../frontend/src/static-analysis/sync_stat.h"

#include "../../../common-libs/utils/list.h"
#include "../../../common-libs/utils/string_utils.h"
#include "../../../common-libs/utils/decorator_utils.h"

#include <cstdlib>
#include <deque>
#include <sstream>
#include <fstream>
#include <iostream>

SyncManager::SyncManager(TaskDef *taskDef,
                        const char *headerFile,
                        const char *programFile,
                        const char *initials) {

	this->taskDef = taskDef;
	this->headerFile = headerFile;
	this->programFile = programFile;
	this->initials = initials;
	this->taskSyncList = NULL;
}

void SyncManager::processSyncList() {

	std::cout << "Processing task specific synchronization requirements\n";
	taskSyncList = new List<SyncRequirement*>;
	CompositeStage *computation = taskDef->getComputation();
	FlowStage *stage = computation;
	
	std::deque<FlowStage*> queue;
	queue.push_back(stage);
	while (!queue.empty()) {
		FlowStage *stage = queue.front();
		queue.pop_front();
		CompositeStage *compositeStage = dynamic_cast<CompositeStage*>(stage);
		if (compositeStage != NULL) {
			List<FlowStage*> *stageList = compositeStage->getStageList();
			for (int i = 0; i < stageList->NumElements(); i++) {
				queue.push_back(stageList->Nth(i));
			}
		}
		StageSyncReqs *stageSyncReqs = stage->getAllSyncRequirements();
		List<SyncRequirement*> *stageSyncList = stageSyncReqs->getAllSyncRequirements();
		for (int i = 0; i < stageSyncList->NumElements(); i++) {
			SyncRequirement *sync = stageSyncList->Nth(i);
			DependencyArc *arc = sync->getDependencyArc();
			if (arc->getSignalSrc() == stage) {
				taskSyncList->Append(sync);
			}
		}
	}
}

bool SyncManager::involvesSynchronization() {
	return taskSyncList != NULL && taskSyncList->NumElements() > 0;
}

void SyncManager::generateSyncPrimitives() {

	if (taskSyncList->NumElements() > 0) {

		std::cout << "\tGenerating global primitives for synchronization\n";
		std::string stmtSeparator = ";\n";
		std::ofstream stream;
        	stream.open (headerFile, std::ofstream::out | std::ofstream::app);
        	if (!stream.is_open()) {
                	std::cout << "Unable to open output header file to write sync primitives";
                	std::exit(EXIT_FAILURE);
        	}
		const char *message = "global synchronization primitives";
		decorator::writeSectionHeader(stream, message);

		stream << std::endl;	
		for (int i = 0; i < taskSyncList->NumElements() ; i++) {
			
			SyncRequirement *sync = taskSyncList->Nth(i);
			Space *syncOwner = sync->getSyncOwner();

			// initialize the sync variable array and array of barriers to reader-to-writer has_read signals
			// we mentioned elsewhere that we need two primitives per update as current implementation of sync
			// primitives does not take into account reader-to-writer okay-to-update-again signals
			stream << "static RS *" << sync->getSyncName() << "s["; 
			stream << "Space_" << syncOwner->getName() << "_Threads_Per_Segment]";
			stream << stmtSeparator;
			stream << "static Barrier *" << sync->getReverseSyncName() << "s[";
			stream << "Space_" << syncOwner->getName() << "_Threads_Per_Segment]";
			stream << stmtSeparator;
		}
		stream.close();
	}
}

void SyncManager::generateSyncInitializerFn() {
	
	if (taskSyncList->NumElements() > 0) {
		
		std::cout << "\tGenerating function for initializing global sync primitives\n";
		std::string stmtSeparator = ";\n";
		std::string indent = "\t";
		std::string doubleIndent = "\t\t";
		std::ofstream hfStream, pfStream;

        	hfStream.open (headerFile, std::ofstream::out | std::ofstream::app);
        	pfStream.open (programFile, std::ofstream::out | std::ofstream::app);
        	if (!(hfStream.is_open() && pfStream.is_open())) {
                	std::cout << "Unable to open output file(s) to write global sync primitive initializer function";
                	std::exit(EXIT_FAILURE);
		}

		const char *message = "Initializer function for global synchronization primitives";
		decorator::writeSectionHeader(hfStream, message);
		decorator::writeSectionHeader(pfStream, message);
		
		hfStream << std::endl << "void initializeSyncPrimitives()" << stmtSeparator << std::endl;
		hfStream.close();

		pfStream << std::endl;
		pfStream << "void " << initials << "::initializeSyncPrimitives() {\n\n";
		for (int i = 0; i < taskSyncList->NumElements() ; i++) {
			SyncRequirement *sync = taskSyncList->Nth(i);
			Space *syncOwner = sync->getSyncOwner();
			Space *syncSpan = sync->getSyncSpan();
			pfStream << indent;
			pfStream << "for (int i = 0; i < Space_" << syncOwner->getName() << "_Threads_Per_Segment; i++) {\n";
			pfStream << doubleIndent << "int participants = ";
			pfStream << "Space_" << syncSpan->getName() << "_Threads_Per_Segment";
			pfStream << " / Space_" << syncOwner->getName() << "_Threads_Per_Segment";
			pfStream << stmtSeparator << doubleIndent;
			pfStream << sync->getSyncName() << "s[i] = new RS(participants)";
			pfStream << stmtSeparator << doubleIndent;
			pfStream << sync->getReverseSyncName() << "s[i] = new Barrier(participants)";
			pfStream << stmtSeparator; 
			pfStream << indent << "}\n";
		}	
		pfStream << "}\n";
		pfStream.close();
	}
}

void SyncManager::generateSyncStructureForThreads() {
	if (taskSyncList->NumElements() > 0) {
		std::cout << "\tGenerating data structure holding sync primitives for each thread\n";
		std::string stmtSeparator = ";\n";
		std::string indent = "\t";
		std::ofstream stream;
        	stream.open (headerFile, std::ofstream::out | std::ofstream::app);
        	if (!stream.is_open()) {
                	std::cout << "Unable to open output header file to write sync data structure for threads";
                	std::exit(EXIT_FAILURE);
        	}

                const char *message = "data structure and function for initializing thread's sync primitives";
		decorator::writeSectionHeader(stream, message);		

		stream << std::endl;
		stream << "class ThreadSyncPrimitive {\n";
		stream << "  public:\n";
		for (int i = 0; i < taskSyncList->NumElements() ; i++) {
			SyncRequirement *sync = taskSyncList->Nth(i);
			stream << indent << "RS *" << sync->getSyncName() << stmtSeparator;	
			stream << indent << "Barrier *" << sync->getReverseSyncName() << stmtSeparator;	
		}
		stream << "};\n";
		stream << std::endl;
		stream << "ThreadSyncPrimitive *getSyncPrimitives(ThreadIds *threadIds)" << stmtSeparator;
		stream.close();
	}
}

void SyncManager::generateFnForThreadsSyncStructureInit() {
	if (taskSyncList->NumElements() > 0) {
		std::cout << "\tGenerating function to initialize each thread's sync primitives holder\n";
		std::string stmtSeparator = ";\n";
		std::string indent = "\t";
		std::ofstream stream;
        	stream.open (programFile, std::ofstream::out | std::ofstream::app);
        	if (!stream.is_open()) {
                	std::cout << "Unable to open program file to generate thread's sync primitive initializer function";
                	std::exit(EXIT_FAILURE);
        	}
                	
		const char *message = "function for initializing thread's sync primitives";
		decorator::writeSectionHeader(stream, message);
		stream << std::endl;

		stream << "ThreadSyncPrimitive *" << initials << "::";
		stream << "getSyncPrimitives(ThreadIds *threadIds) {\n\n";
		stream << indent << "ThreadSyncPrimitive *threadSync = new ThreadSyncPrimitive()" << stmtSeparator;

		List<const char*> *lpsWithGroupIdCalculated = new List<const char*>;
		for (int i = 0; i < taskSyncList->NumElements() ; i++) {
			
			SyncRequirement *sync = taskSyncList->Nth(i);
			Space *syncOwner = sync->getSyncOwner();
			const char *syncOwnerName = syncOwner->getName();
			
			if (!string_utils::contains(lpsWithGroupIdCalculated, syncOwnerName)) {
				lpsWithGroupIdCalculated->Append(syncOwnerName);
				
				stream << std::endl;
				stream << indent << "int space" << syncOwnerName << "Group = ";
				stream << "threadIds->ppuIds[Space_" << syncOwnerName << "].groupId % ";
				stream << "Space_" << syncOwnerName << "_Threads_Per_Segment";
				stream << stmtSeparator;
			}
						
			stream << indent << "threadSync->" << sync->getSyncName();
			stream << " = " << sync->getSyncName() << "s[";
			stream << "space" << syncOwnerName << "Group]" << stmtSeparator;	
			stream << indent << "threadSync->" << sync->getReverseSyncName();
			stream << " = " << sync->getReverseSyncName() << "s[";
			stream << "space" << syncOwnerName << "Group]" << stmtSeparator;	
		}

		stream << std::endl << indent << "return threadSync" << stmtSeparator;
		stream << "}";
		stream << std::endl << std::endl;
		stream.close();
	}
}
