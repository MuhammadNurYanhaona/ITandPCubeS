#ifndef _H_env_instruction
#define _H_env_instruction

#include "environment.h"

/*-------------------------------------------------------------------------------------------------------------------------------------
				 Environment Instructions to be Processed At Task Initialization
-------------------------------------------------------------------------------------------------------------------------------------*/

// this is the base class for all types of instructions for initializing an environmental data structure that a task going to 
// access/create as part of its execution 
class TaskInitEnvInstruction {
  protected:
	// the item in the task environment this instruction is going to operate on
	TaskItem *itemToUpdate;
  public:
	TaskInitEnvInstruction(TaskItem *itemToUpdate) { this->itemToUpdate = itemToUpdate; }
	TaskItem *getItemToUpdate() { return itemToUpdate; }
	
	// this function should be called before the task has been scheduled for execution as without dimension lengths
	// information, partition configuration and other necessary metadata for parts of the data structure cannot be
	// constructed, precluding any further processing of data parts
	virtual void setupDimensions() = 0;

	// some instructions may lead to update/removal of existing versions of the data structure stored in the program
	// environment; this function should be called to do those changes 
	virtual void preprocessProgramEnv() = 0;

	// this function should be called after partition configurations, part-container tree, etc. metadata have been
	// gathered for the task item; to prepare the parts list for the data structure before processing of computation 
	// stages can begin
	virtual void setupPartsList() = 0;

	// this function should be invoked to ensure any new/updated parts list for the data structure has been included in 
	// the program environment
	virtual void postprocessProgramEnv() = 0;			
};

/* This is the default instruction for linked task environmental variables. If there is no other instruction associated with
 * such a variable at task invocation, a checking must be performed to ensure that the existing parts list is up-to-date. If
 * it is stale then an automatic data transfer instruction should be issued by the library to undertake a fresh to stale list
 * content transfer. 
 */
class StaleRefreshInstruction : public TaskInitEnvInstruction {
  public:
	StaleRefreshInstruction(TaskItem *itemToUpdate) : TaskInitEnvInstruction(itemToUpdate) {}
	void setupDimensions() {};
	void preprocessProgramEnv() {};
	void setupPartsList() {};
	void postprocessProgramEnv() {};			
};

/* This is the instruction for environmental variables created by the task; creation of a new data item for such a variable 
 * for may result in removal of a previously created item during a previous execution of the task.
 */
class CreateFreshInstruction : public TaskInitEnvInstruction {
  public:	
	CreateFreshInstruction(TaskItem *itemToUpdate) : TaskInitEnvInstruction(itemToUpdate) {}
	void setupDimensions() {};
	void preprocessProgramEnv() {};
	void setupPartsList() {};
	void postprocessProgramEnv() {};			
};

/* This, as the name suggests, causes the data parts content of a task item to be read from some external file.
 */
class ReadFromFileInstruction : public TaskInitEnvInstruction {
  protected:
	const char *fileName;
  public:	
	ReadFromFileInstruction(TaskItem *itemToUpdate) : TaskInitEnvInstruction(itemToUpdate) {}
	void setFileName(const char *fileName) { this->fileName = fileName; }
	void setupDimensions() {};
	void preprocessProgramEnv() {};
	void setupPartsList() {};
	void postprocessProgramEnv() {};			
};

/* This encodes an explicit object assignment from one task to another task environment in the form envA.a = envB.b; note
 * that only portion of the data item can be assigned from the source to the destination task's environment using the array
 * sub-range expression.
 */
class DataTransferInstruction : public TaskInitEnvInstruction {
  protected:
	TaskItem *dataSourceItem;
	// need to encode subpartition range here
  public:	
	DataTransferInstruction(TaskItem *itemToUpdate) : TaskInitEnvInstruction(itemToUpdate) {}
	void setSourceItem(TaskItem *source) { dataSourceItem = source; }
	void setupDimensions() {};
	void preprocessProgramEnv() {};
	void setupPartsList() {};
	void postprocessProgramEnv() {};			
};

/*-------------------------------------------------------------------------------------------------------------------------------------
		            Environment Instructions to be Processed At Task Completion or Program End
-------------------------------------------------------------------------------------------------------------------------------------*/

/* This is the base class for all types of instructions that tell how the completion of a task should affect the overall
 * program environment. For example, if a task updates a data item having multiple versions in the program environment 
 * then versions that are not updated should be marked stale.
 */
class TaskEndEnvInstruction {
  protected:
	TaskItem *envItem;
  public:
	TaskEndEnvInstruction(TaskItem *envItem);
	void execute() {
		updateProgramEnv();
		doAdditionalProcessing();
	}
	virtual void updateProgramEnv() = 0;
	virtual void doAdditionalProcessing() = 0;
};

/* instruction for recording updates in stale/fresh versions list for a data item at task completion
 */
class ChangeNotifyInstruction : public TaskEndEnvInstruction {
  public:
	ChangeNotifyInstruction(TaskItem *envItem) : TaskEndEnvInstruction(envItem) {}
	void updateProgramEnv() {};
	void doAdditionalProcessing() {};
};

/* instruction for recording that a particular data item should be written to output file/files after task completion or
 * at the end of the program 
 */
class WriteToFileInstruction : public TaskEndEnvInstruction {
  protected:
	const char *fileName;
  public:
	WriteToFileInstruction(TaskItem *envItem) : TaskEndEnvInstruction(envItem) {}
	void setFileName(const char *fileName) { this->fileName = fileName; }
	void updateProgramEnv() {};
	void doAdditionalProcessing() {};
}; 


#endif
