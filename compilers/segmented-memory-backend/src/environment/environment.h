#ifndef _H_environment
#define _H_environment

#include "../utils/list.h"
#include "../utils/hashtable.h"
#include "../utils/interval.h"
#include "../memory-management/allocation.h"
#include "../memory-management/part_generation.h"
#include "../memory-management/part_tracking.h"
#include "../input-output/data_handler.h"
#include "../runtime/structure.h"

#include <vector>

/* This library holds all the data structure definitionss related to enviornment managements of a multi-tasked IT program. 
 * The first set of classes are for program environment related data structures and the seconds are for task environment
 * related data structures. These two sets go hand-in-hand as program environment gets updated by execution of each task. 
 */

class TaskInitEnvInstruction;
class TaskEndEnvInstruction;

/*------------------------------------------------------------------------------------------------------------------------
						Program Environment
------------------------------------------------------------------------------------------------------------------------*/

/* The underlying data structures that the system retain is a set of parts lists for each data item used in the program.
 * There might be multiple parts lists due to different tasks using different partition configurations. Even a single task
 * may have multiple parts lists for its independent LPSes. At an instance, some of these parts lists may be in used while
 * others are not, some may have up-to-date content while others are stale. So a set of attributes are needed per parts 
 * list to ensure proper usage of of a parts list.  
 */
class PartsListAttributes {
  protected:
	// the number of environmental references there are on a single parts list; if the reference count goes to zero 
	// then the parts list will be removed and its memory will be reclaimed
	int referenceCount;
	// indicates if the content is up-to-date -- thus can be used right now; or it needs to be refreshed using content
	// from an up-to-date version 
	bool fresh;
	// if this flag is set to true then it means some task is currently updating the list
	bool dirty;
  public:
	PartsListAttributes() {
		referenceCount = 1;
		fresh = true;
		dirty = false;
	}
	void increaseReferenceCount() { referenceCount++; }
	void decreaseReferenceCount() { referenceCount--; }
	int getReferenceCount() { return referenceCount; }
	void flagStale() { fresh = false; }
	void flagFresh() { fresh = true; }
	bool isFresh() { return fresh; }
	void setDirtyBit(bool value) { dirty = value; }
	bool isDirty() { return dirty; }
};

/* This class represents a list of parts a segment holds for a data item for a particular partition configuration. Note
 * that multiple tasks can share a list when there partition configurations are the same.
 */
class PartsList {
  protected:
	PartsListAttributes *attributes;
	List<DataPart*> *parts;
  public:
	PartsList(List<DataPart*> *parts);
	PartsList(PartsListAttributes *attributes, List<DataPart*> *parts);
	~PartsList();
	List<DataPart*> *getDataParts() { return parts; }
	PartsListAttributes *getAttributes() { return attributes; }
};

/* Note that environmental dependencies among tasks happens through the exchanges of properties from corresponding task
 * environments. To decide whether to move data or share some existing parts list when a data item is assigned from one 
 * task environment to another, we need to compare the partition configurations of the concerned tasks and the dimension
 * ranges of the data item they are concerned about. Further, if a situation requiring data movements is identified then 
 * further information about the construction of the parts lists are needed. This class holds all such information.  
 */
class ListReferenceAttributes {
  protected:
	List<Dimension> *rootDimensions;
	DataPartitionConfig *partitionConfig;
	// a sorted hierarchical structure needed to efficiently read/write elements from/to multiple parts
	PartIdContainer *partContainerTree;
	// a mathematical description of the content of a parts list
	List<MultidimensionalIntervalSeq*> *segmentFold;
  public:
	ListReferenceAttributes(DataPartitionConfig *partitionConfig, List<Dimension> *rootDimensions);
	~ListReferenceAttributes();
	List<Dimension> *getRootDimensions() { return rootDimensions; }
	DataPartitionConfig *getPartitionConfig() { return partitionConfig; }
	void setPartContainerTree(PartIdContainer *containerTree) {
		this->partContainerTree = containerTree;
	}
	PartIdContainer *getPartContainerTree() { return partContainerTree; }
	void computeSegmentFold();
	List<MultidimensionalIntervalSeq*> *getSegmentFold() { return segmentFold; }
	// tells if the current fold contains all data elements included in the argument folds; this function is useful
	// to determine if the data need for a new reference can be satisfied with locally available contents
	bool isSuperFold(List<MultidimensionalIntervalSeq*> *otherFold);	
};

/* We mentioned before that some parts lists for a data item may be stale as multiple tasks can manipulate a single data 
 * item. We decide to refresh a parts list on-demand. As a result, an instruction such as env2.b = env1.a that assigns
 * a new reference for the data may have the parts list for env1.a to be stale at the time of its execution. Therefore,
 * a mechanism is needed then to find a fresh parts list that can satisfy the data need for env2.b in place of env1.a. To 
 * support such a mechanism, we need identifiers for references to a single data item. This class serves that purpose.  
 */
class ListReferenceKey {
  protected:
	// the runtime id assigned to the task environment holding a reference
	int taskEnvId;
	// the variable name for the data item inside that environment
	const char *varName;
	// the task's LPS that would need to allocate the parts list had it not been there already
	const char *allocatorLpsName;
  public:
	ListReferenceKey(int taskEnvId, const char *varName, const char *allocatorLpsName);
	void setTaskEnvId(int taskEnvId) { this->taskEnvId = taskEnvId; }
	void setVarName(const char *varName) { this->varName = varName; }
	void setAllocatorLps(const char *allocatorLpsName) { this->allocatorLpsName = allocatorLpsName; }
	// generates a dummy key for pattern matching against existing keys; if the user set any value for any property
	// on the pattern key then that property will be compared during the matching process; otherwise, that property
	// will be ignored 
	ListReferenceKey *initiatePatternKey() { return new ListReferenceKey(-1, NULL, NULL); }
	const char *generateKey();
	bool isEqual(ListReferenceKey *other);
	bool matchesPattern(ListReferenceKey *pattern);	
};

/* This class represents a task environment's reference to a data item's parts list with all associated attributes
 */
class PartsListReference {
  protected:
	ListReferenceAttributes *attributes;
	ListReferenceKey *key;
	PartsList *partsList;
  public:
	PartsListReference(ListReferenceAttributes *attr, ListReferenceKey *key, PartsList *list) {
		this->attributes = attr;
		this->key = key;
		this->partsList = list;
	}
	~PartsListReference() {
		delete attributes;
		delete key;
	}
	ListReferenceAttributes *getAttributes() { return attributes; }
	ListReferenceKey *getKey() { return key; }
	PartsList *getPartsList() { return partsList; }		
};

/* A mechanism is needed to collect all parts lists corresponding to a single data item under a common group within the
 * program environment. This class provides an identifer for such a group.  
 */
class ObjectIdentifier {
  protected:
	// An environmental data items always comes into being due to the execution of some task. Each execution of each
	// task in a program has a unique ID. That ID constitute the first part of the data item indentifer   
	int sourceTaskId;
	// the second part of the identifier is the environmental object number, in the order objects appear in the source
	// code within originator task
	int envLinkId;
  public:
	ObjectIdentifier(int sourceTaskId, int envLinkId);
	const char *generateKey();
};

/* This class holds all the references and parts lists corresponding to a single data item
 */
class ObjectVersionManager {
  protected:
	ObjectIdentifier *objectIdentifer;
	List<ListReferenceKey*> *freshVersionKeys;
	Hashtable<PartsListReference*> *dataVersions;
  public:
	ObjectVersionManager(ObjectIdentifier *objectIdentifier, PartsListReference* sourceReference);
	~ObjectVersionManager();
	void addNewVersion(PartsListReference *versionReference);
	void removeVersion(ListReferenceKey *versionKey);
	PartsListReference *getVersion(const char *versionKey);
	void markNonMatchingVersionsStale(ListReferenceKey *matchingKey);
	void addFreshVersionKey(ListReferenceKey *freshKey);
	List<PartsListReference*> *getFreshVersions();
};

/* The program environment is at the end a collection of object-version-managers for different data items 
 */
class ProgramEnvironment {
  protected:
	Hashtable<ObjectVersionManager*> *envObjects;
  public:
	ProgramEnvironment();
	void addNewDataItem(ObjectIdentifier *identifier, PartsListReference* sourceReference);
	ObjectVersionManager *getVersionManager(const char *dataItemKey);	
};

/*------------------------------------------------------------------------------------------------------------------------
						   Task Environment
------------------------------------------------------------------------------------------------------------------------*/

/* Inside a task an environmental data item is identified by its variable name. In addition a link Id is maintained which 
 * is the number for varible in the environment section's ordering of variables) to aid unique identifier creation for the 
 * variable if the task originates the data item.
 */
class EnvironmentLinkKey {
  protected:
	const char *varName;
	int linkId;
	// in case the task is going to use an existing environmental data item for the particular key -- as opposed to 
	// creating it -- a source identifier is needed to locate the group for the data item in the program environment. 
	const char *sourceKey;
  public:
	EnvironmentLinkKey(const char *varName, int linkId);
	const char *getVarName() { return varName; }
	void setSourceKey(const char *sourceKey) { this->sourceKey = sourceKey; }
	const char *getSourceKey() { return sourceKey; }
	void flagAsDataSource(int taskId);
	const char *generateKey(int taskId);
	ObjectIdentifier *generateObjectIdentifier(int taskId);	
	bool isEqual(EnvironmentLinkKey *other);
};

/* A single task can have multiple parts list for independent LPSes sharing the same data item. This class holds all 
 * information and data regarding one such allocation and serves as the bridge between the parts-list-reference in the 
 * program environemnt and an allocation within the task   
 */
class LpsAllocation {
  protected:
	const char *lpsId;
	DataPartitionConfig *partitionConfig;
	PartIdContainer *partContainerTree;
	PartsList *partsList;
  public:
	LpsAllocation(const char *lpsId, DataPartitionConfig *partitionConfig);
	const char *getLpsId() { return lpsId; }
	void setPartitionConfig(DataPartitionConfig *config) { partitionConfig = config; }
	void setPartContainerTree(PartIdContainer *containerTree) {
		this->partContainerTree = containerTree;
	}
	void setPartsList(PartsList *partsList) { this->partsList = partsList; }
	PartsList *getPartsList() { return partsList; }	
	ListReferenceKey *generatePartsListReferenceKey(int envId, const char *varName);	
	PartsListReference *generatePartsListReference(int envId, 
		const char *varName, List<Dimension> *rootDimensions);	
};

/* The linkage type of an task-item needs to be retained to determine what should we do about the item if no explicit 
 * instruction has been provided for its initialization
 */
enum EnvItemType { IN_OUT, OPTIONAL_IN_OUT, OUT };

/* This class traks all parts lists references and their metadata for a single data item used by a task */
class TaskItem {
  protected:
	EnvironmentLinkKey *key;
	EnvItemType type;
	List<Dimension> *rootDimensions;
	Hashtable<LpsAllocation*> *allocations;
	int elementSize;
  public:
	TaskItem(EnvironmentLinkKey *key, EnvItemType type, int dimensionality, int elementSize);
	void setRootDimensions(Dimension *dimensions);
	void setRootDimensions(List<Dimension> *dimensionList) { this->rootDimensions = dimensionList; }
	List<Dimension> *getRootDimensions() { return rootDimensions; }
	EnvironmentLinkKey *getEnvLinkKey() { return key; }
	void preConfigureLpsAllocation(const char *lpsId, DataPartitionConfig *partitionConfig);
	LpsAllocation *getLpsAllocation(const char *lpsId) { return allocations->Lookup(lpsId); }
	Dimension getDimension(int dimNo) { return rootDimensions->Nth(dimNo); }
	void setDimension(int dimNo, Dimension dimension) { rootDimensions->Nth(dimNo) = dimension; }
	EnvItemType getType() { return type; }
	bool isEmpty();
	const char *getFirstAllocationsLpsId();
};

/* This is the super-class for all task environments. Compiler will generate task specific environment objects to provide
 * easier access to a task's data structure
 */
class TaskEnvironment {
  protected:
	// the runtime ID for the environment that will be assigned to it at the instanciation time
	int envId;
	// environmental data item references for the underlying task
	Hashtable<TaskItem*> *envItems;
	// queues of instructions to be executed at the beginning and ending of a task execution for environment management
	std::vector<TaskInitEnvInstruction*> initInstrs;
	std::vector<TaskEndEnvInstruction*> endingInstrs;
	// maps of file readers-writers to load/store environmental data from/to files
	Hashtable<PartReader*> *readersMap;
	Hashtable<PartWriter*> *writersMap;
	// a stream for logging events during task environment processing
        std::ofstream *logFile;
  public:
	// this variable does not have any practical purpose; it is there just to keep track of the name of the task the
	// environment object has been created for 
	const char *name;
	// this variable is used to assign an unique ID to a new task-environment object during its creation
	static int CURRENT_ENV_ID;	
  public:
	TaskEnvironment();
	TaskItem *getItem(const char *itemName) { return envItems->Lookup(itemName); }
	void setReadersMap(Hashtable<PartReader*> *readersMap) { this->readersMap = readersMap; }
	void setWritersMap(Hashtable<PartWriter*> *writersMap) { this->writersMap = writersMap; }
	void setLogFile(std::ofstream *logFile) { this->logFile = logFile; }
	void setDefaultEnvInitInstrs();

	// function to write output to a file in response to a bind-output command
	void writeItemToFile(const char *itemName, const char *filePath);

	// task specific environment subclasses should provide implementation of the following two library functions
	virtual void prepareItemsMap() {}
	virtual void setDefaultTaskCompletionInstrs() {}

	// functions to register an environment object manipulation instruction
	void addInitEnvInstruction(TaskInitEnvInstruction *instr);
	void addEndEnvInstruction(TaskEndEnvInstruction *instr);

	// return the current initialization instruction of a specific type for a particular item, if exists; otherwise
	// returns NULL; check env_instruction.h header file to find what are the types of different initialization instrs
	TaskInitEnvInstruction *getInstr(const char *itemName, int instrType);

	// functions to be invoked at different phases of task invocation/execution/ending to do the work for various
	// environment manipulation instructions
	//------------------------------------------------ task initialization functions
	void setupItemsDimensions();
	void preprocessProgramEnvForItems();
	void setupItemsPartsLists();
	void postprocessProgramEnvForItems();
	//----------------------------------------------------- task completion function
	void executeTaskCompletionInstructions();

	// function to be invoked to re-populate the default initialization and completion time instructions to prepare for
	// future usages of the same task environment
	void resetEnvInstructions(); 
};

#endif
