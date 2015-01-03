#ifndef _H_task_generator
#define _H_task_generator

#include <iostream>
#include <string.h>
#include <stdio.h>

#include "../syntax/ast_task.h"
#include "../utils/list.h"

class PPS_Definition;

class TaskGenerator {
  private:
	TaskDef *taskDef;
	const char *mappingFile;
	const char *headerFile;
	const char *programFile;
	const char *initials;
  public:
	TaskGenerator(TaskDef *taskDef, 
		const char *outputDirectory, 
		const char *mappingFile);
	void generate(List<PPS_Definition*> *pcubesConfig);
	const char *getHeaderFile() { return headerFile; }
	const char *getProgramFile() { return programFile; }
	TaskDef *getTaskDef() { return taskDef; }
	const char* getTaskName() { return taskDef->getName(); }
	const char* getInitials() { return initials; }		
};

#endif

