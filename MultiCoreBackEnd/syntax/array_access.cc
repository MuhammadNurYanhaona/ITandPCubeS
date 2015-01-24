#include "ast.h"
#include "ast_stmt.h"
#include "ast_expr.h"
#include "ast_type.h"

#include "../utils/list.h"
#include "string.h"
#include "../semantics/symbol.h"
#include "../semantics/task_space.h"
#include "errors.h"

#include "../utils/hashtable.h"
#include "../static-analysis/data_access.h"
#include "../static-analysis/loop_index.h"
#include "../codegen/name_transformer.h"

#include <iostream>
#include <sstream>
#include <cstdlib>

//--------------------------------------------------- Array Access ----------------------------------------------------/

SubRangeExpr::SubRangeExpr(Expr *b, Expr *e, yyltype loc) : Expr(loc) {
	begin = b;
	if (begin != NULL) {
		begin->SetParent(this);
	}
	end = e;
	if (end != NULL) {
		end->SetParent(this);
	}
	fullRange = (b == NULL && e == NULL);		
}

void SubRangeExpr::PrintChildren(int indentLevel) {
	if (begin != NULL) begin->Print(indentLevel + 1);
	if (end != NULL) end->Print(indentLevel + 1);
}

void SubRangeExpr::inferType(Scope *scope, Type *rootType) {
	if (begin != NULL) begin->inferType(scope, Type::intType);
	if (end != NULL) end->inferType(scope, Type::intType);	
}

Hashtable<VariableAccess*> *SubRangeExpr::getAccessedGlobalVariables(TaskGlobalReferences *globalReferences) {
	Hashtable<VariableAccess*> *table = new Hashtable<VariableAccess*>;
	if (begin != NULL) {
		table = begin->getAccessedGlobalVariables(globalReferences);
	}
	if (end != NULL) {
		Hashtable<VariableAccess*> *eTable = end->getAccessedGlobalVariables(globalReferences);
		Iterator<VariableAccess*> iter = eTable->GetIterator();
		VariableAccess *accessLog;
		while ((accessLog = iter.GetNextValue()) != NULL) {
			if (table->Lookup(accessLog->getName()) != NULL) {
				table->Lookup(accessLog->getName())->mergeAccessInfo(accessLog);
			} else {
				table->Enter(accessLog->getName(), accessLog, true);
			}
		}
	}
	return table;
}

List<FieldAccess*> *SubRangeExpr::getTerminalFieldAccesses() {
	List<FieldAccess*> *list = Expr::getTerminalFieldAccesses();
	if (begin != NULL) Expr::copyNewFields(list, begin->getTerminalFieldAccesses());
	if (end != NULL) Expr::copyNewFields(list, end->getTerminalFieldAccesses());
	return list;
}

ArrayAccess::ArrayAccess(Expr *b, Expr *i, yyltype loc) : Expr(loc) {
	Assert(b != NULL && i != NULL);
	base = b;
	base->SetParent(this);
	index = i;
	index->SetParent(this);
}

void ArrayAccess::PrintChildren(int indentLevel) {
	base->Print(indentLevel + 1, "(Base) ");
	index->Print(indentLevel + 1, "(Index) ");
}

void ArrayAccess::resolveType(Scope *scope, bool ignoreFailure) {

	base->resolveType(scope, ignoreFailure);
	Type *baseType = base->getType();
	if (baseType == NULL) {
		if (!ignoreFailure) {
			ReportError::InvalidArrayAccess(GetLocation(), NULL, ignoreFailure);
			this->type = Type::errorType;
		}
	} else {
		ArrayType *arrayType = dynamic_cast<ArrayType*>(baseType);
		if (arrayType == NULL) {
			if (baseType != Type::errorType) {
				ReportError::InvalidArrayAccess(base->GetLocation(), baseType, ignoreFailure);
			}
			this->type = Type::errorType;
		} else {
			SubRangeExpr *subRange = dynamic_cast<SubRangeExpr*>(index);
			if (subRange != NULL) {
				this->type = arrayType;
				subRange->inferType(scope, NULL);
			} else {
				this->type = arrayType->reduceADimension();
				index->resolveType(scope, ignoreFailure);
				index->inferType(scope, Type::intType);
				
				int position = getIndexPosition();
				FieldAccess *indexField = dynamic_cast<FieldAccess*>(index);
				if (indexField != NULL && indexField->isTerminalField()) {
					const char *indexName = indexField->getBaseVarName();
					const char *arrayName = base->getBaseVarName();
					IndexScope *indexScope = IndexScope::currentScope->getScopeForAssociation(indexName);
					if (indexScope != NULL) {
						IndexArrayAssociation *association = new IndexArrayAssociation(indexName, 
								arrayName, position);
						indexScope->saveAssociation(association);
						indexField->markAsIndex();
					}
				}
			}	
		}
	}	
}

Hashtable<VariableAccess*> *ArrayAccess::getAccessedGlobalVariables(TaskGlobalReferences *globalReferences) {
	
	Hashtable<VariableAccess*> *table = base->getAccessedGlobalVariables(globalReferences);
	const char *baseVarName = getBaseVarName();
	FieldAccess *baseField = dynamic_cast<FieldAccess*>(base);
	if (baseField != NULL && baseField->isTerminalField()) {
		VariableAccess *accessLog = table->Lookup(baseVarName);
		if (accessLog != NULL) {
			accessLog->markContentAccess();
		}
	}
	Hashtable<VariableAccess*> *indexTable = index->getAccessedGlobalVariables(globalReferences);
	Iterator<VariableAccess*> iter = indexTable->GetIterator();
	VariableAccess *indexAccess;
	while ((indexAccess = iter.GetNextValue()) != NULL) {
		if (indexAccess->isMetadataAccessed()) indexAccess->getMetadataAccessFlags()->flagAsRead();
		if(indexAccess->isContentAccessed()) indexAccess->getContentAccessFlags()->flagAsRead();
		if (table->Lookup(indexAccess->getName()) != NULL) {
			VariableAccess *accessLog = table->Lookup(indexAccess->getName());
			accessLog->mergeAccessInfo(indexAccess);
		} else {
			table->Enter(indexAccess->getName(), indexAccess, true);
		}
	}
	return table;
}

int ArrayAccess::getIndexPosition() {
	ArrayAccess *precedingAccess = dynamic_cast<ArrayAccess*>(base);
	if (precedingAccess != NULL) return precedingAccess->getIndexPosition() + 1;
	return 0;
}

Expr *ArrayAccess::getEndpointOfArrayAccess() {
	ArrayAccess *precedingAccess = dynamic_cast<ArrayAccess*>(base);
	if (precedingAccess != NULL) {
		return precedingAccess->getEndpointOfArrayAccess();
	} else return base;
}

List<FieldAccess*> *ArrayAccess::getTerminalFieldAccesses() {
	List<FieldAccess*> *list = base->getTerminalFieldAccesses();
	Expr::copyNewFields(list, index->getTerminalFieldAccesses());
	return list;
}

void ArrayAccess::translate(std::ostringstream &stream, int indentLevel, int currentLineLength) {
	Type *baseType = base->getType();
	if (dynamic_cast<StaticArrayType*>(baseType) != NULL) {
		base->translate(stream, indentLevel, currentLineLength);
		stream << "["; 
		index->translate(stream, indentLevel, currentLineLength);
		stream << "]";
	} else {
		// Note that since user defined objects cannot have dynamic arrays, any dynamic array access must be
		// an access to a dynamic task global variable. Therefore we can assume that the base of this array
		// access has the name of that variable and its type is an array type.
		Expr *endpoint = getEndpointOfArrayAccess();
		endpoint->translate(stream, indentLevel, currentLineLength);
		std::ostringstream indexStream;
		ArrayType *arrayType = (ArrayType*) endpoint->getType();
		const char *arrayName = endpoint->getBaseVarName();
		// we now translate a possibly multidimensional array access into a unidimensional access as arrays
		// are stored as 1D memory blocks in the back-end
		generate1DIndexAccess(indexStream, arrayName, arrayType);
		// then we write the index access logic in the array
		stream << '[' << indexStream.str() << ']';
	}
}

void ArrayAccess::generate1DIndexAccess(std::ostringstream &stream, const char *array, ArrayType *type) {
	int dimension = getIndexPosition();
	// If this is not the beginning index access in the expression then pass the computation forward
	// first to get the earlier part been translated.
	if (dimension > 0) {
		ArrayAccess *previous = dynamic_cast<ArrayAccess*>(base);
		previous->generate1DIndexAccess(stream, array, type);
		stream << " + ";
	}
	int dimensionCount = type->getDimensions();
	// If the index access corresponds to some index of a parallel or sequential loop then it is readily
	// available some pre-translated expression holder variable. So write the name of the variable in 
	// the stream
	FieldAccess *indexAccess = dynamic_cast<FieldAccess*>(index);
	if (indexAccess != NULL && indexAccess->isIndex()) {
		indexAccess->translateIndex(stream, array, dimension);
	// Otherwise, there might be a need for translating the index
	} else {
		stream << '('; 
		index->translate(stream, 0, 0);
                stream << " - " << array << "partDims[" << dimension << "].getPositiveRange().min";
		stream << ')';
		std::ostringstream xform;
                for (int i = dimensionCount - 1; i > dimension; i--) {
                        stream << " * " << array << "StoreDims[" << i << "].getLength()";
                }
	}
}
