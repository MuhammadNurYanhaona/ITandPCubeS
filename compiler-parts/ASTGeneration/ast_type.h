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
#include "list.h"


class Type : public Node {
  protected:
	char *typeName;

  public :
    	static Type 	*intType, *floatType, *doubleType, *boolType, *charType, *stringType, 
			*epochType, *dimensionType, *rangeType, *indexType, *errorType;

    	Type(yyltype loc) : Node(loc) {}
    	Type(const char *str);
    
    	const char *GetPrintNameForNode() { return "Type"; }
    	void PrintChildren(int indentLevel);
};

class NamedType : public Type {
  protected:
  	Identifier *id;
    
  public:
    	NamedType(Identifier *i);
    	const char *GetPrintNameForNode() { return "NamedType"; }
   	void PrintChildren(int indentLevel);
};

class ArrayType : public Type {
  protected:
    	Type *elemType;
	int dimensions;
  public:
    	ArrayType(yyltype loc, Type *elemType, int dimensions);
    	const char *GetPrintNameForNode() { return "Dynamic Array"; }
    	void PrintChildren(int indentLevel);
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
};

class ListType : public Type {
  protected:
	Type *elemType;
  public:
	ListType(yyltype loc, Type *elemType);	
    	const char *GetPrintNameForNode() { return "List"; }
    	void PrintChildren(int indentLevel);
};

 
#endif
