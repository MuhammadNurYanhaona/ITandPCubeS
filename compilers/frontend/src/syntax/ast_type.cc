#include <sstream>
#include <iostream>
#include <string>
#include <cstdlib>

#include "ast_def.h"
#include "ast_type.h"
#include "../semantics/symbol.h"
#include "../../../common-libs/utils/list.h"
#include "../../../common-libs/utils/hashtable.h"
#include "../../../common-libs/utils/string_utils.h"

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
Type *Type::indexType           = new Type("Index");
Type *Type::rangeType           = new Type("Range");
Type *Type::dimensionType       = new Type("Dimension");
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

Node *Type::clone() {
	return new Type(strdup(typeName));
}

const char *Type::getCppDeclaration(const char *varName, bool pointer) {
        std::ostringstream decl;
        decl << typeName << " ";
        if (pointer) decl << "*";
        decl << varName;
        return strdup(decl.str().c_str());
}

void Type::storeBuiltInTypesInScope(Scope *scope) {

        //----------------------------------------------------store the primitive types

        scope->insert_symbol(new Symbol(intType->getName(), intType));
        scope->insert_symbol(new Symbol(floatType->getName(), floatType));
        scope->insert_symbol(new Symbol(doubleType->getName(), doubleType));
        scope->insert_symbol(new Symbol(charType->getName(), charType));
        scope->insert_symbol(new Symbol(boolType->getName(), boolType));
        scope->insert_symbol(new Symbol(stringType->getName(), stringType));
        scope->insert_symbol(new Symbol(errorType->getName(), errorType));
        scope->insert_symbol(new Symbol(voidType->getName(), voidType));

        //-----------create nested scope for each built-in tuple types then store them

	// an integer range (typically needed for setting an array dimension)
        yyltype dummyLocation;
        List<VariableDef*> *rangeElements = new List<VariableDef*>;
        rangeElements->Append(new VariableDef(
                        new Identifier(dummyLocation, (const char*) "min"),
                        intType));
        rangeElements->Append(new VariableDef(
                        new Identifier(dummyLocation, (const char*) "max"),
                        intType));
        TupleDef *rangeTuple = new TupleDef(
                        new Identifier(dummyLocation, rangeType->getName()),
                        rangeElements);
        rangeTuple->attachScope(scope);

	// array index with stride and cyclical range traversal support (not in use)
	List<VariableDef*> *indexElements = new List<VariableDef*>;
        indexElements->Append(new VariableDef(
                        new Identifier(dummyLocation, (const char*) "stride"),
                        intType));
        indexElements->Append(new VariableDef(
                        new Identifier(dummyLocation, (const char*) "rotate"),
                        boolType));
        TupleDef *indexTuple = new TupleDef(
                        new Identifier(dummyLocation, indexType->getName()),
                        indexElements);
        indexTuple->attachScope(scope);

	// metadata representing an array dimension
	List<VariableDef*> *dimensionElements = new List<VariableDef*>;
        dimensionElements->Append(new VariableDef(
                        new Identifier(dummyLocation, (const char*) "range"),
                        rangeType));
        dimensionElements->Append(new VariableDef(
                        new Identifier(dummyLocation, (const char*) "index"),
                        indexType));
        dimensionElements->Append(new VariableDef(
                        new Identifier(dummyLocation, (const char*) "length"),
                        intType));
        TupleDef *dimensionTuple = new TupleDef(
                        new Identifier(dummyLocation, dimensionType->getName()),
                        dimensionElements);
        dimensionTuple->attachScope(scope);
}

bool Type::isAssignableFrom(Type *other) {

        if (this == Type::errorType || other == Type::errorType) {
                return true;
        } else if (this == Type::intType) {
                return (other == Type::charType
                                || other == Type::intType);
        } else if (this == Type::floatType) {
                return (other == Type::charType
                                || other == Type::intType
                                || other == Type::floatType);
        } else if (this == Type::doubleType) {
                return (other == Type::charType
                                || other == Type::intType
                                || other == Type::floatType
                                || other == Type::doubleType);
        } else { return this == other; }
}

//---------------------------------------------- Tuple Type ------------------------------------------------/

NamedType::NamedType(Identifier *i) : Type(*i->GetLocation()) {
        Assert(i != NULL);
        (id=i)->SetParent(this);
	environmentType = false;
        taskName = NULL;
}

void NamedType::PrintChildren(int indentLevel) {
        id->Print(indentLevel+1);
}

Node *NamedType::clone() {
	Identifier *newId = (Identifier*) id->clone();
	return new NamedType(newId);
}

const char *NamedType::getCType() {
        if (!environmentType) return id->getName();
        std::ostringstream stream;
        const char *initials = string_utils::getInitials(taskName);
        initials = string_utils::toLower(initials);
        stream << initials << "::TaskEnvironmentImpl";
        return strdup(stream.str().c_str());
}

const char *NamedType::getCppDeclaration(const char *varName, bool pointer) {
        std::ostringstream decl;
        decl << getCType() << " ";
        if (environmentType) decl << "*";
        decl << varName;
        return strdup(decl.str().c_str());
}

bool NamedType::isEqual(Type *other) {
        NamedType *otherType = dynamic_cast<NamedType*>(other);
        if (otherType == NULL) return false;
        return strcmp(this->getName(), otherType->getName()) == 0;
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

const char *ArrayType::getName() {
	const char *elementTypeName = elemType->getName();
        int elemNameLength = strlen(elementTypeName);
        std::ostringstream stm ;
        stm << dimensions; 
        std::string str =  stm.str();
        char *dimensionality = (char *) str.c_str();
        int dimensionalityLength = strlen(dimensionality);
        int length = dimensionalityLength + elemNameLength + 12;
        char *arrayName = (char *) malloc(length * sizeof(char));
        strcpy(arrayName, dimensionality);  
        strcat(arrayName, "D Array of ");
        strcat(arrayName, elementTypeName); 
        return (const char*) arrayName;
}

Node *ArrayType::clone() {
	Type *newElemType = (Type*) elemType->clone();
	return new ArrayType(*GetLocation(), newElemType, dimensions);
}

const char *ArrayType::getCType() {
        std::ostringstream cName;
        cName << elemType->getCType() << "*";
        return strdup(cName.str().c_str());
}

const char *ArrayType::getCppDeclaration(const char *varName, bool pointer) {
        std::ostringstream decl;
        decl << this->getCType() << " " << varName;
        return strdup(decl.str().c_str());
}

bool ArrayType::isEqual(Type *other) {
        ArrayType *otherArray = dynamic_cast<ArrayType*>(other);
        if (otherArray == NULL) return false;
        if (this->dimensions != otherArray->dimensions) return false;
        return this->elemType->isEqual(otherArray->elemType);
}

Type *ArrayType::reduceADimension() {
        if (dimensions == 1) return elemType;
        return new ArrayType(*GetLocation(), elemType, dimensions - 1);
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

Node *StaticArrayType::clone() {
	Type *newElemType = (Type*) elemType->clone();
	StaticArrayType *newType = new StaticArrayType(*GetLocation(), newElemType, dimensions);
	List<int> *newDimLengths = new List<int>;
	newDimLengths->AppendAll(dimensionLengths);
	newType->setLengths(newDimLengths);
	return newType;	
}

const char *StaticArrayType::getCppDeclaration(const char *varName, bool pointer) {
        std::ostringstream decl;
        decl << elemType->getCType() << " ";
        if (pointer) decl << "*";
        decl << varName;
        for (int i = 0; i < dimensionLengths->NumElements(); i++) {
                decl << '[' << dimensionLengths->Nth(i) << ']';
        }
        return strdup(decl.str().c_str());
}

Type *ArrayType::getTerminalElementType() {
        ArrayType *elemArray = dynamic_cast<ArrayType*>(elemType);
        if (elemArray != NULL) {
                return elemArray->getTerminalElementType();
        }
        return elemType;
}

Type *StaticArrayType::reduceADimension() {
        if (dimensions == 1) return elemType;
        StaticArrayType *arrayType = new StaticArrayType(*GetLocation(), elemType, dimensions - 1);
        List<int> *dims = new List<int>;
        for (int i = 1; i < dimensionLengths->NumElements(); i++) {
                dims->Append(dimensionLengths->Nth(i));
        }
        arrayType->setLengths(dims);
        return arrayType;
}

//---------------------------------------------- List Type -------------------------------------------------/

ListType::ListType(yyltype loc, Type *et) : Type(loc) {
        Assert(et != NULL);
        (elemType=et)->SetParent(this);
}

void ListType::PrintChildren(int indentLevel) {
        elemType->Print(indentLevel + 1, "Element ");
}

const char *ListType::getName() {
        const char *elementTypeName = elemType->getName();
        int elemNameLength = strlen(elementTypeName);
        int length = elemNameLength + 9;
        char *listName = (char *) malloc(length * sizeof(char));
        strcpy(listName, "List of ");
        strcat(listName, elementTypeName);
        return (const char*) listName;
}

Node *ListType::clone() {
	Type *newElemType = (Type*) elemType->clone();
	return new ListType(*GetLocation(), newElemType);
}

const char *ListType::getCType() {
        std::ostringstream cType;
        cType << "std::vector<" << elemType->getCType() << ">";
        return strdup(cType.str().c_str());
}

const char *ListType::getCppDeclaration(const char *varName, bool pointer) {
        std::ostringstream decl;
        decl << this->getCType() << " ";
        if (pointer) decl << "*";
        decl << varName;
        return strdup(decl.str().c_str());
}

Type *ListType::getTerminalElementType() {
        ArrayType *elemArray = dynamic_cast<ArrayType*>(elemType);
        if (elemArray != NULL) {
                return elemArray->getTerminalElementType();
        }
        return elemType;
}

bool ListType::isEqual(Type *other) {
        ListType *otherList = dynamic_cast<ListType*>(other);
        if (otherList == NULL) return false;
        return this->elemType->isEqual(otherList->elemType);
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

Node *MapType::clone() {
	
	MapType *newMap = new MapType(*GetLocation());
	List<VariableDef*> *elemList = getElementList();

	for (int i = 0; i < elemList->NumElements(); i++) {
		VariableDef *element = elemList->Nth(i);
		VariableDef *newElem = (VariableDef*) element->clone();
		newMap->setElement(newElem);
	}
	
	delete elemList;
	return newMap;
}

const char *MapType::getCppDeclaration(const char *varName, bool pointer) {
        std::ostringstream decl;
        decl << this->getCType() << " ";
        if (pointer) decl << "*";
        decl << varName;
        return strdup(decl.str().c_str());
}

bool MapType::isEqual(Type *other) {
	return (dynamic_cast<MapType*>(other) != NULL);
}
