/* This file defines the abstract base class 'Node' and the concrete Identifier 
 * and Error node subclasses that are used through the tree as leaf nodes. A 
 * parse tree is a hierarchical collection of ast nodes (or, more correctly, of 
 * instances of concrete subclassses such as VarDecl, ForStmt, and AssignExpr).
 * 
 * Location: Each node maintains its lexical location (line and columns in 
 * file), that location can be NULL for those nodes that don't care/use 
 * locations. The location is typcially set by the node constructor.  The 
 * location is used to provide the context when reporting semantic errors.
 *
 * Parent: Each node has a pointer to its parent. For a Program node, the 
 * parent is NULL, for all other nodes it is the pointer to the node one level
 * up in the parse tree.  The parent is not set in the constructor (during a 
 * bottom-up parse we don't know the parent at the time of construction) but 
 * instead we wait until assigning the children into the parent node and then 
 * set up links in both directions. The parent link is typically not used 
 * during parsing, but is more important in later phases.
 *
 * Printing: Each node class is responsible for printing itself/children by 
 * overriding the virtual PrintChildren() and GetPrintNameForNode() methods. 
 */

#ifndef _H_ast
#define _H_ast

#include <cstdlib>
#include <string.h>
#include "../common/location.h"

class Symbol;

class Node {
  protected:
    	yyltype *location;
    	Node *parent;
  public:
    	Node(yyltype loc);
    	Node();
    
    	yyltype *GetLocation()   { return location; }
    	void SetParent(Node *p)  { parent = p; }
    	Node *GetParent()        { return parent; }
	
	//------------------------------------------------------------------ Helper functions for Semantic Analysis
	
	// This function is needed to clone stage and function definitions from the abstract syntax
	// tree for resolving type polymorphism.
	virtual Node *clone() {};

    	//---------------------------------------------------------------------------------printing related methods
	
    	virtual const char *GetPrintNameForNode() = 0;
    	// Print() is deliberately _not_ virtual subclasses should override PrintChildren() instead
    	void Print(int indentLevel, const char *label = NULL); 
    	void PrintLabel(int indentLevel, const char *label = NULL); 
    	virtual void PrintChildren(int indentLevel)  {}

};
   

class Identifier : public Node {
  protected:
    	char *name;    
  public:
	static const char *RangeId, *IndexId, *LocalId;    		
    	Identifier(yyltype loc, const char *name);
	const char *getName() { return name; }
    	const char *GetPrintNameForNode()   { return "Identifier"; }
    	void PrintChildren(int indentLevel);
	
	//------------------------------------------------------------------ Helper functions for Semantic Analysis
	
	virtual Node *clone() { return new Identifier(*GetLocation(), strdup(name)); }
};

class DimensionIdentifier : public Identifier {
  protected:
	int dimension;
  public:
	DimensionIdentifier(yyltype loc, int dimension);    		 	
    	const char *GetPrintNameForNode()   { return "Dimension"; }
    	void PrintChildren(int indentLevel);
	int getDimensionNo() { return dimension; }
	
	//------------------------------------------------------------------ Helper functions for Semantic Analysis
	
	Node *clone() { return new DimensionIdentifier(*GetLocation(), dimension); }
};

// This node class is designed to represent a portion of the tree that  encountered syntax errors 
// during parsing. The partial completed tree is discarded along with the states being popped, and 
// an instance of the Error class can stand in as the placeholder in the parse tree when your parser 
// can continue after an error.
class Error : public Node {
  public:
	Error() : Node() {}
	const char *GetPrintNameForNode()   { return "Error"; }
	
	//------------------------------------------------------------------ Helper functions for Semantic Analysis
	
	Node *clone() { return new Error(); }
};



#endif
