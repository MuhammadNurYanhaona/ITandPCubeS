#ifndef _H_ast_type
#define _H_ast_type

#include "ast.h"
#include "../semantics/scope.h"
#include "../../../common-libs/utils/list.h"
#include "../../../common-libs/utils/hashtable.h"

class VariableDef;

class Type : public Node {
  protected:
	const char *typeName;

  public:
    	static Type 	*intType, *floatType, *doubleType, *boolType, 
			*charType, *stringType, *dimensionType, 
			*rangeType, *indexType, *errorType, *voidType;

    	Type(yyltype loc) : Node(loc) {}
    	Type(const char *str);
	virtual ~Type() {}
	virtual const char *getName() { return typeName; }
    
    	const char *GetPrintNameForNode() { return "Type"; }
    	void PrintChildren(int indentLevel);

	//------------------------------------------------------------------ Helper functions for Semantic Analysis

        virtual Node *clone();

	// These are helper methods for generating the C++ type declaration of an IT variable. Subclasses should 
	// override these methods to apply appropriate equivalent C++ type.
        virtual const char *getCType() { return typeName; }
        virtual const char *getCppDeclaration(const char *varName, bool pointer = false);

	static void storeBuiltInTypesInScope(Scope *scope);
};

class NamedType : public Type {
  protected:
  	Identifier *id;
	// This is a flag to indicate if this type represents the environment of some task. We need this to do 
	// specialized operations needed by environment objects. For example, we declare pointers for environment 
	// types as oppose to object instances done for other named types.
        bool environmentType;
        // The name of the task when this is an environment type for a task
        const char *taskName;

  public:
    	NamedType(Identifier *i);
    	const char *GetPrintNameForNode() { return "Named-Type"; }
   	void PrintChildren(int indentLevel);

	//------------------------------------------------------------------ Helper functions for Semantic Analysis

        Node *clone();
	const char *getCType();
        const char *getCppDeclaration(const char *varName, bool pointer);
	
	// functions added to generate task-specific environment object classes
	void flagAsEnvironmentType() { environmentType = true; }
        bool isEnvironmentType() { return environmentType; }
        void setTaskName(const char *taskName) { this->taskName = taskName; }
        const char *getTaskName() { return taskName; }
};

class ArrayType : public Type {
  protected:
    	Type *elemType;
	int dimensions;
  public:
    	ArrayType(yyltype loc, Type *elemType, int dimensions);
	virtual ~ArrayType() {}
    	const char *GetPrintNameForNode() { return "Dynamic-Array"; }
    	void PrintChildren(int indentLevel);
	int getDimensions() { return dimensions; }

	//------------------------------------------------------------------ Helper functions for Semantic Analysis

        virtual Node *clone();
	const char *getCType();
        virtual const char *getCppDeclaration(const char *varName, bool pointer);
	Type *getTerminalElementType();
};

class StaticArrayType : public ArrayType {
  protected:
	List<int> *dimensionLengths;
  public:
	StaticArrayType(yyltype loc, Type *elemType, int dimensions) 
		: ArrayType(loc, elemType, dimensions) {}
    	const char *GetPrintNameForNode() { return "Static-Array"; }
    	void PrintChildren(int indentLevel);
	void setLengths(List<int> *dimensionLengths);

	//------------------------------------------------------------------ Helper functions for Semantic Analysis

        Node *clone();
	Type *reduceADimension();
        const char *getCppDeclaration(const char *varName, bool pointer);
};

class ListType : public Type {
  protected:
	Type *elemType;
  public:
	ListType(yyltype loc, Type *elemType);	
    	const char *GetPrintNameForNode() { return "List"; }
    	void PrintChildren(int indentLevel);

	//------------------------------------------------------------------ Helper functions for Semantic Analysis

        Node *clone();
	const char *getCType();
        const char *getCppDeclaration(const char *varName, bool pointer);
	Type *getTerminalElementType();
};

class MapType : public Type {
  protected:
        Hashtable<VariableDef*> *elements;
  public:
        MapType(yyltype loc);
    	const char *GetPrintNameForNode() { return "Associated-List (Map)"; }
	const char *getName() { return "Map"; }
    	void PrintChildren(int indentLevel);
	bool hasElement(const char *elementName);
        Type *getElementType(const char *elementName);
        void setElement(VariableDef *newArg);
        VariableDef* getElement(const char *elementName);
        List<VariableDef*> *getElementList();

	//------------------------------------------------------------------ Helper functions for Semantic Analysis

        Node *clone();
	const char *getCType() { return strdup("Hashtable<void*>"); }
        const char *getCppDeclaration(const char *varName, bool pointer);
};
 
#endif
