#ifndef _H_task_space
#define _H_task_space

#include "list.h"
#include "hashtable.h"
#include "ast_def.h"
#include "ast.h"
#include "location.h"
#include "ast_type.h"

/*	Partition Order specifies in what order individual partitions (or subpartitions) of a space will be
	generated and processed by the runtime. This parameter is mostly relevant for subpartitioned spaces
	as in normal cases partitions of a space are independent. 
*/
enum PartitionOrder { AscendingOrder, DescendingOrder, RandomOrder };

/*	Partition Link Type specifies how partitions of a single data structure in a lower space is linked to 
	its parent partition in higher space. Some-times this linking may be specified in the Space level that
	dictates the relationships of all its data structure partitions to partitions of some parent space.
	Other-times, relationships are defined data structure by data structure basis. If like type is
	
	LinkTypePartition: then lower space partition further divides upper space partitions
	LinkTypeSubpartition: then lower space partition divides each upper space subpartitions
	LinkTypeUndefined: then there is no linkage type currently defined. Some may be derived from other 
			   information
*/
enum PartitionLinkType { LinkTypePartition, LinkTypeSubpartition, LinkTypeUndefined };

class PartitionArg;
class Space;

/*	This class stores partition funcion arguments regarding a single dimension of a task global array 
	within a single space configuration. 
*/
class DataDimensionConfig {
  protected:
	int dimensionNo;
	Node *dividingArg;
	Node *frontPaddingArg;
	Node *backPaddingArg;
  public:
	DataDimensionConfig(int dimensionNo, Node *dividingArg);
	DataDimensionConfig(int dimensionNo, Node *dividingArg, 
			Node *frontPaddingArg, Node *backPaddingArg);
	void setDividingArg(Node *dividingArg) {this->dividingArg = dividingArg; }
	void setPaddingArg(Node *paddingArg) {
		this->frontPaddingArg = this->backPaddingArg = paddingArg;
	}
	void setPaddingArg(Node *paddingArgFront, Node *paddingArgBack) {
		this->frontPaddingArg = paddingArgFront;
		this->backPaddingArg = paddingArgBack;
	}
	void setDimensionNo(int dimensionNo) { this->dimensionNo = dimensionNo; }
	int getDimensionNo() { return dimensionNo; }
	Node *getDividingArg() { return dividingArg; }
	bool hasPadding() { return (frontPaddingArg != NULL || backPaddingArg != NULL); }		
}; 

/*	This class embodies the detail of application of a single partition functions on a data structure.
	It holds the user specified partitioning arguments, name of the partition functions, and interfaces
	for determining how application of the underlying partition function affects the data structure. 
	Specific partition functions, i.e., block-count or block-size functions implements its interfaces
	to do their work. Our broad plan/goal is to allow user defined partition functions in the future.
	At that time further enrichment of this class might become necessary.
*/
class PartitionFunctionConfig {
  protected:
	yyltype *location;
	const char *functionName;
	List<DataDimensionConfig*> *arguments;
	PartitionOrder partitionOrder;
	PartitionFunctionConfig(yyltype *location, const char *functionName);		
	virtual void processArguments(List<PartitionArg*> *dividingArgs, List<PartitionArg*> *paddingArgs);
  public:
	static PartitionFunctionConfig *generateConfig(yyltype *location, 
			const char *functionName, 
			List<PartitionArg*> *dividingArgs, 
			List<PartitionArg*> *paddingArgs);
	virtual void setDimensionIds(List<int> *dimensionIds);
	virtual List<int> *getBlockedDimensions(Type *structureType) { return new List<int>; }
	virtual List<int> *getPartitionedDimensions();
	virtual int getDimensionality() { return arguments->NumElements(); }
	void setPartitionOrder(PartitionOrder partitionOrder) { this->partitionOrder = partitionOrder; }
	bool isOrdered() { return (partitionOrder == AscendingOrder || partitionOrder == DescendingOrder); }
	bool hasOverlappingsAmongPartitions();
};

class DataStructure {
  protected:
	VariableDef *definition;
	DataStructure *source;
	List<DataStructure*> *dependents;
	Space *space;

	// this variable is only useful for subpartitioned data structures. If it true, it signifies that
	// the immediate parent space of the subpartition space, i.e., the space some of whose data 
	// structures are subpartitioned, cannot store the content of this data structure due to space
	// limitation. Therefore, any update must be moved further up in the partition hierarchy if persistence
	// is desired.	
	bool nonStorable;
  public:
	DataStructure(VariableDef *definition);
	DataStructure(DataStructure *source);
	virtual ~DataStructure() {};
	const char *getName();
	void setSpaceReference(Space *space);
	Space *getSpace() { return space; }
	DataStructure *getSource() { return source; }
	Type *getType();
	void flagAsNonStorable() { nonStorable = true; }
	bool isNonStorable() { return nonStorable; }
	virtual bool hasOverlappingsAmongPartitions() { return false; }	
};

class ArrayDataStructure : public DataStructure {
  protected:
	List<int> *sourceDimensions;
	List<PartitionFunctionConfig*> *partitionSpecs;
	List<int> *afterPartitionDimensions;
  public:
	ArrayDataStructure(VariableDef *definition);
	ArrayDataStructure(ArrayDataStructure *source);
	bool isOrderDependent();
	void setSourceDimensions(List<int> *sourceDimensions) { this->sourceDimensions = sourceDimensions; }
	void setAfterPartitionDimensions(List<int> *afterPartitionDimensions) {
		this->afterPartitionDimensions = afterPartitionDimensions;
	}
	List<int> *getRemainingDimensions() { return afterPartitionDimensions; }
	void addPartitionSpec(PartitionFunctionConfig *partitionConfig);	
	bool hasOverlappingsAmongPartitions();	
};

/*	Token, Coordinate, and CoordinateSystem classes implement a mechanism for associating dimensions
	of data structures to the dimensions of corresponding spaces. Comparing toold Fortran-D terminology,
	this is the mechanism of storing data structure alignment information in IT. We view individual 
	spaces as a coordinate system of dimensionality as specified in the partition block. Then dimensions
	of individual data structures within it somehow must be mapped to the coordinate dimensions. If some
	data structure is replicated, it should have a wildcard token in along some dimensions.

	This representations gives a good way to visualize and think about the partitioning scheme. It also
	makes validation of partition specification easy. That boils down to enuring that all data structures
	have one-and-only-one token per coordinate dimension. 
*/
class Token {
  protected:
	int dimensionId;
	DataStructure *data;
  public:
 	static int wildcardTokenId;
	Token(DataStructure *data, int dimensionId) {
		this->data = data; 
		this->dimensionId = dimensionId; 
	}
	bool isWildcard() { return dimensionId == wildcardTokenId; }
	DataStructure *getData() {return data; }
};

class Coordinate {
  protected:
	int id;	
	List<Token*> *tokenList;
  public:
	Coordinate(int id) {
		this->id = id; 
		tokenList = new List<Token*>; 
	}
	void storeToken(Token *token) { tokenList->Append(token); }
	int getTokenCount() { return tokenList->NumElements(); }
	Token *getTokenForDataStructure(const char *dataStructureName);	
};

class CoordinateSystem {
  protected:
	int dimensionCount;
	List<Coordinate*> *dimensions;
  public:
	CoordinateSystem(int dimensionCount);
	bool isBalanced();
	Coordinate *getCoordinate(int dimensionNo);		
};

class Space {
  protected:
	const char *id;
	int dimensions;
	bool dynamic;
	bool subpartitionSpace;
	CoordinateSystem *coordSys;
	Space *parent;
	Hashtable<DataStructure*> *dataStructureList;
  public:
	static const char *RootSpaceName;
	static const char *SubSpaceSuffix;
	Space(const char *name, int dimensions, bool dynamic, bool subpartitionSpace);
	void setStructureList(Hashtable<DataStructure*> *dataStructureList);
	void initEmptyStructureList();
	void addDataStructure(DataStructure *structure);
	DataStructure *getStructure(const char *name);
	bool isInSpace(const char *structName);
	DataStructure *getLocalStructure(const char *name);
	void setParent(Space *parent) { this->parent = parent; }
	Space *getParent() { return parent; }
	int getDimensionCount() { return dimensions; }
	const char *getName() { return id; }
	void storeToken(int coordinate, Token *token);
	bool isValidCoordinateSystem() { return (coordSys == NULL || coordSys->isBalanced()); }	
	bool isParentSpace(Space *suspectedParent);
	Space *getClosestSubpartitionRoot();
	bool isDynamic() { return dynamic; }
	List<const char*> *getLocallyUsedArrayNames();
	List<const char*> *getLocalDataStructureNames();
	bool isReplicatedInCurrentSpace(const char *dataStructureName);
	bool isReplicated(const char *dataStructureName);
	bool isSubpartitionSpace() { return subpartitionSpace; }
	static List<Space*> *getConnetingSpaceSequenceForSpacePair(Space *first, Space *last);
	List<const char*> *getLocalDataStructuresWithOverlappedPartitions();
	List<const char*> *getNonStorableDataStructures();
};

/*	The entire partition block is seen as a hierarchy of coordinate systems of spaces. The hierarchy
	is rooted in the RootSpace that holds all task-global data structures. Below it a tree is formed
	based on the partial ordering visible in the user's specification for the partition block.
*/
class PartitionHierarchy {
  protected:
	Hashtable<Space*> *spaceHierarchy;
  public:
	PartitionHierarchy();
	Space *getSpace(char spaceId);
	Space *getSubspace(char spaceId);
	Space *getRootSpace();
	bool addNewSpace(Space *space);
	Space *getCommonAncestor(Space *space1, Space *space2);	
};

#endif
