/* File: ast.cc
 * ------------
 */

#include "ast.h"
#include <string.h> // strdup
#include <stdio.h>  // printf

Node::Node(yyltype loc) {
    location = new yyltype(loc);
    parent = NULL;
}

Node::Node() {
    location = NULL;
    parent = NULL;
}

/* The Print method is used to print the parse tree nodes.
 * If this node has a location (most nodes do, but some do not), it
 * will first print the line number to help you match the parse tree 
 * back to the source text. It then indents the proper number of levels 
 * and prints the "print name" of the node. It then will invoke the
 * virtual function PrintChildren which is expected to print the
 * internals of the node (itself & children) as appropriate.
 */
void Node::Print(int indentLevel, const char *label) { 
    	const int numSpaces = 3;
    	printf("\n");
    	if (GetLocation()) printf("%*d", numSpaces, GetLocation()->first_line);
    	else printf("%*s", numSpaces, "");
    	//printf("%*s%s: ", indentLevel*numSpaces, "",  label? label : GetPrintNameForNode());
    	printf("%*s%s%s: ", indentLevel*numSpaces, "",  label? label : "", GetPrintNameForNode());
	PrintChildren(indentLevel);
} 

void Node::PrintLabel(int indentLevel, const char *label) { 
    	const int numSpaces = 3;
    	printf("\n");
    	printf("%*s", numSpaces, "");
    	printf("%*s%s: ", indentLevel*numSpaces, "", label);
} 
	 
Identifier::Identifier(yyltype loc, const char *n) : Node(loc) {
    	name = strdup(n);
}

const char *Identifier::RangeId = "Range";
const char *Identifier::IndexId = "Index";
const char *Identifier::LocalId = "Local"; 

void Identifier::PrintChildren(int indentLevel) {
    	printf("%s", name);
}

DimensionIdentifier::DimensionIdentifier(yyltype loc, int dim) : Identifier(loc, "Dimension") {
	dimension = dim;
}

void DimensionIdentifier::PrintChildren(int indentLevel) {
	printf("%d", dimension);
}
