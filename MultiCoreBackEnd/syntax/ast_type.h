/* File: ast_type.h
 * ----------------
 * In our parse tree, Type nodes are used to represent and
 * store type information. The base Type class is used
 * for built-in types, the NamedType for Tuples
 * and the Array and List Type for arrays and lists respectively.  
 */
 
#ifndef _H_ast_type
#define _H_ast_type

#include "ast.h"
#include "../utils/list.h"
#include "../semantics/scope.h"

class VariableDef;

class Type : public Node {
  protected:
	const char *typeName;

  public :
    	static Type 	*intType, *floatType, *doubleType, *boolType, *charType, *stringType, 
			*epochType, *dimensionType, *rangeType, *indexType, *errorType;

    	Type(yyltype loc) : Node(loc) {}
    	Type(const char *str);
	virtual ~Type() {}
    
    	const char *GetPrintNameForNode() { return "Type"; }
    	void PrintChildren(int indentLevel);

	// Built in types are stored in the 
	static void storeBuiltInTypesInScope(Scope *scope);
	
	virtual const char *getName() { return typeName; }
	virtual bool isAssignableFrom(Type *other);
	virtual bool isEqual(Type *other) { return this == other; }

	// These are helper method for generating the C++ type declaration of an IT variable.
	// Subclasses should override these methods to apply appropriate equivalent C++ type.
	virtual const char *getCType() { return typeName; }	
	virtual const char *getCppDeclaration(const char *varName);
};

class NamedType : public Type {
  protected:
  	Identifier *id;
    
  public:
    	NamedType(Identifier *i);
    	const char *GetPrintNameForNode() { return "NamedType"; }
   	void PrintChildren(int indentLevel);
	Identifier *getId() { return id; }
	const char *getName() { return id->getName(); }
	bool isAssignableFrom(Type *other) { return isEqual(other); }
	bool isEqual(Type *other);
	const char *getCType() { return id->getName(); }	
	virtual const char *getCppDeclaration(const char *varName);
};

class ArrayType : public Type {
  protected:
    	Type *elemType;
	int dimensions;
  public:
    	ArrayType(yyltype loc, Type *elemType, int dimensions);
	virtual ~ArrayType() {}
    	const char *GetPrintNameForNode() { return "Dynamic Array"; }
    	void PrintChildren(int indentLevel);
	int getDimensions() { return dimensions; }
	const char *getName();
	bool isAssignableFrom(Type *other) { return isEqual(other); }
	bool isEqual(Type *other);
	virtual Type *reduceADimension();
	Type *getTerminalElementType();
	const char *getCType();	
	virtual const char *getCppDeclaration(const char *varName);
};

class StaticArrayType : public ArrayType {
  protected:
	List<int> *dimensionLengths;
  public:
	StaticArrayType(yyltype loc, Type *elemType, int dimensions) 
		: ArrayType(loc, elemType, dimensions) {}
    	const char *GetPrintNameForNode() { return "Static Array"; }
    	void PrintChildren(int indentLevel);
	void setLengths(List<int> *dimensionLengths);	
	Type *reduceADimension();
	const char *getCppDeclaration(const char *varName);
};

/*  TODO: Need to add the methods for accessing list elements and related other features. */
class ListType : public Type {
  protected:
	Type *elemType;
  public:
	ListType(yyltype loc, Type *elemType);	
    	const char *GetPrintNameForNode() { return "List"; }
    	void PrintChildren(int indentLevel);
	const char *getName();
	Type *getTerminalElementType();
	bool isAssignableFrom(Type *other) { return isEqual(other); }
	bool isEqual(Type *other);
	Type *getElementType() { return elemType; }
	const char *getCType();	
	const char *getCppDeclaration(const char *varName);
};

class MapType : public Type {
  protected:
        Hashtable<VariableDef*> *elements;
  public:
        MapType(yyltype loc);
    	const char *GetPrintNameForNode() { return "Associated List (Map)"; }
	const char *getName() { return "Map"; }
    	void PrintChildren(int indentLevel);
        bool hasElement(const char *elementName);
        Type *getElementType(const char *elementName);
        void setElement(VariableDef *newArg);
        VariableDef* getElement(const char *elementName);
        List<VariableDef*> *getElementList(); 
};
 
#endif
