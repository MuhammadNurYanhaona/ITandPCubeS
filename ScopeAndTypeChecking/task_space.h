#ifndef _H_task_space
#define _H_task_space

#include "list.h"
#include "hashtable.h"
#include "ast_def.h"
#include "ast.h"
#include "location.h"
#include "ast_type.h"

enum PartitionOrder { AscendingOrder, DescendingOrder, RandomOrder };
enum PartitionLinkType { LinkTypePartition, LinkTypeSubpartition, LinkTypeUndefined };

class PartitionArg;
class Space;

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
};

class DataStructure {
  protected:
	VariableDef *definition;
	DataStructure *source;
	List<DataStructure*> *dependents;
	Space *space;
  public:
	DataStructure(VariableDef *definition);
	DataStructure(DataStructure *source);
	virtual ~DataStructure() {};
	const char *getName();
	void setSpaceReference(Space *space);
	Type *getType();	
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
};

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
	DataStructure *getLocalStructure(const char *name);
	void setParent(Space *parent) { this->parent = parent; }
	Space *getParent() { return parent; }
	int getDimensionCount() { return dimensions; }
	const char *getName() { return id; }
	void storeToken(int coordinate, Token *token);
	bool isValidCoordinateSystem() { return (coordSys == NULL || coordSys->isBalanced()); }	
	bool isParentSpace(Space *suspectedParent);
	Space *getClosestSubpartitionRoot();
};

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
