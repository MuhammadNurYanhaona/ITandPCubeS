#include <sstream>
#include <iostream>
#include <string>
#include <cstdlib>

#include "ast_def.h"
#include "ast_type.h"
#include "../../../common-libs/utils/list.h"
#include "../../../common-libs/utils/hashtable.h"

/* Class constants
 * ---------------------------------------------------------------------------------------------------------/
 * These are public constants for the built-in base types (int, double, etc.)
 * They can be accessed with the syntax Type::intType. This allows you to
 * directly access them and share the built-in types where needed rather that
 * creates lots of copies.
 */

Type *Type::intType             = new Type("int");
Type *Type::floatType           = new Type("float");
Type *Type::doubleType          = new Type("double");
Type *Type::charType            = new Type("char");
Type *Type::boolType            = new Type("bool");
Type *Type::stringType          = new Type("const char*");
Type *Type::dimensionType       = new Type("Dimension");
Type *Type::rangeType           = new Type("Range");
Type *Type::indexType           = new Type("Index");
Type *Type::errorType           = new Type("Error");
Type *Type::voidType            = new Type("void");

//---------------------------------------------- Built in Type ---------------------------------------------/

Type::Type(const char *n) {
        Assert(n);
        typeName = strdup(n);
}

void Type::PrintChildren(int indentLevel) {
        printf("%s", typeName);
}

//---------------------------------------------- Tuple Type ------------------------------------------------/

NamedType::NamedType(Identifier *i) : Type(*i->GetLocation()) {
        Assert(i != NULL);
        (id=i)->SetParent(this);
}

void NamedType::PrintChildren(int indentLevel) {
        id->Print(indentLevel+1);
}

//---------------------------------------------- Array Type ------------------------------------------------/

ArrayType::ArrayType(yyltype loc, Type *et, int dims) : Type(loc) {
        Assert(et != NULL && dims != 0);
        dimensions = dims;
        (elemType=et)->SetParent(this);
}

void ArrayType::PrintChildren(int indentLevel) {
        PrintLabel(indentLevel + 1, "Dimensions");
        printf("%d", dimensions);
        elemType->Print(indentLevel + 1, "Element ");
}

//------------------------------------------- Static Array Type --------------------------------------------/

void StaticArrayType::setLengths(List<int> *dimLengths) {
        dimensionLengths = dimLengths;
}

void StaticArrayType::PrintChildren(int indentLevel) {
        ArrayType::PrintChildren(indentLevel);
        PrintLabel(indentLevel + 1, "Dimension Lengths");
        for (int i = 0; i < dimensionLengths->NumElements(); i++) {
                printf("%d ", dimensionLengths->Nth(i));
        }
}

//---------------------------------------------- List Type -------------------------------------------------/

ListType::ListType(yyltype loc, Type *et) : Type(loc) {
        Assert(et != NULL);
        (elemType=et)->SetParent(this);
}

void ListType::PrintChildren(int indentLevel) {
        elemType->Print(indentLevel + 1, "Element ");
}

//---------------------------------------------- Map Type --------------------------------------------------/

MapType::MapType(yyltype loc) : Type(loc) {
        elements = new Hashtable<VariableDef*>;
}

void MapType::PrintChildren(int indentLevel) {
        List<VariableDef*> *elementList = getElementList();
        elementList->PrintAll(indentLevel + 1);
}

bool MapType::hasElement(const char *elementName) {
        VariableDef* elem = elements->Lookup(elementName);
        return (elem != NULL);
}

Type *MapType::getElementType(const char *elementName) {
        VariableDef* elem = elements->Lookup(elementName);
        if (elem == NULL) return NULL;
        return elem->getType();
}

void MapType::setElement(VariableDef *newElem) {
        elements->Enter(newElem->getId()->getName(), newElem, false);
}

VariableDef *MapType::getElement(const char *elementName) {
        return elements->Lookup(elementName);
}

List<VariableDef*> *MapType::getElementList() {

        List<VariableDef*> *elementList = new List<VariableDef*>;
        Iterator<VariableDef*> iterator = elements->GetIterator();
        VariableDef *var;

        while((var = iterator.GetNextValue()) != NULL) {
                elementList->Append(var);
        }
        return elementList;
}

