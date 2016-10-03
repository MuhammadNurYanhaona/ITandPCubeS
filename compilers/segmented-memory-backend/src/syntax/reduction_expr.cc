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
#include "../static-analysis/array_assignment.h"
#include "../codegen/name_transformer.h"
#include "../codegen/task_generator.h"

#include <iostream>
#include <sstream>
#include <cstdlib>

//--------------------------------------------------- Old Sequential Reduction Expression ------------------------------------------------/

ReductionExpr::ReductionExpr(char *o, Expr *r, yyltype loc) : Expr(loc) {
	Assert(r != NULL && o != NULL);
	if (strcmp(o, "sum") == 0) op = SUM;
	else if (strcmp(o, "product") == 0) op = PRODUCT;
	else if (strcmp(o, "max") == 0) op = MAX;
	else if (strcmp(o, "maxEntry") == 0) op = MAX_ENTRY;
	else if (strcmp(o, "min") == 0) op = MIN;
	else if (strcmp(o, "minEntry") == 0) op = MIN_ENTRY;
	else if (strcmp(o, "avg") == 0) op = AVG;
	else {
		// Forcefully through a fault for now. Later we will add user defined reduction function, God willing.
		std::cout << "Currently the compiler does not support user defined reduction functions";
		Assert(0 == 1);	
	}
	right = r;
	right->SetParent(this);
	reductionLoop = NULL;
}

void ReductionExpr::PrintChildren(int indentLevel) {
	PrintLabel(indentLevel + 1, "Operator");
	switch (op) {
		case SUM: printf("Sum"); break;
		case PRODUCT: printf("Product"); break;
		case MAX: printf("Maximum"); break;
		case MIN: printf("Minimum"); break;
		case AVG: printf("Average"); break;
		case MIN_ENTRY: printf("Minimum Entry"); break;
		case MAX_ENTRY: printf("Maximum Entry"); break;
	}
	right->Print(indentLevel + 1);
}

void ReductionExpr::resolveType(Scope *scope, bool ignoreFailure) {

	// first do type resolution like any other expression
	right->resolveType(scope, ignoreFailure);
	Type *rightType = right->getType();
	if (rightType == NULL) {
		ReportError::UnknownExpressionType(right, ignoreFailure);
	}

	if (rightType != NULL && rightType != Type::intType 
			&& rightType != Type::floatType
			&& rightType != Type::doubleType
			&& rightType != Type::errorType) {
		ReportError::UnsupportedOperand(right, rightType, 
				"reduction expression", ignoreFailure);
		this->type = Type::errorType;
	}

	if (op == MIN_ENTRY || op == MAX_ENTRY) {
			this->type = Type::intType;
	} else {
		this->type = rightType;
	}

	// then flag the encompassing loop as a reduction loop
	reductionLoop = LoopStmt::currentLoop;
	if (LoopStmt::currentLoop == NULL) {
		ReportError::ReductionOutsideForLoop(GetLocation(), ignoreFailure);
	}
	LoopStmt::currentLoop->flagAsReductionLoop();
	LoopStmt::currentLoop->setReductionExpr(this);
}

void ReductionExpr::inferType(Scope *scope, Type *rootType) {

	if (op == MIN_ENTRY || op == MAX_ENTRY) {
		this->type = Type::intType;
		right->inferType(scope);
	} else {	
		if (rootType == NULL) { right->inferType(scope);
		} else if (this->type == NULL || rootType->isAssignableFrom(this->type)) {
			this->type = rootType;
			right->inferType(scope, rootType);
		} else if (!rootType->isAssignableFrom(this->type)) {
			ReportError::InferredAndActualTypeMismatch(this->GetLocation(), 
					rootType, this->type, false);
		}
	}
}

Hashtable<VariableAccess*> *ReductionExpr::getAccessedGlobalVariables(TaskGlobalReferences *globalReferences) {
	
	const char *rightVarName = right->getBaseVarName();
	ArrayAccess *rArray = dynamic_cast<ArrayAccess*>(right);
	if (rArray != NULL) {
		// Reduction of some array
		if (globalReferences->doesReferToGlobal(rightVarName)) {
			VariableSymbol *symbol = globalReferences->getGlobalRoot(rightVarName);
			rightVarName = symbol->getName();
		}
	}
	
	Hashtable<VariableAccess*> *table = right->getAccessedGlobalVariables(globalReferences);
	Iterator<VariableAccess*> iter = table->GetIterator();
	VariableAccess *accessLog;
	while ((accessLog = iter.GetNextValue()) != NULL) {
		if (rightVarName != NULL && strcmp(rightVarName, accessLog->getName()) == 0) {
			accessLog->getContentAccessFlags()->flagAsReduced();
		} else {
			if (accessLog->isMetadataAccessed()) 
				accessLog->getMetadataAccessFlags()->flagAsRead();
			if(accessLog->isContentAccessed()) 
				accessLog->getContentAccessFlags()->flagAsRead();
		}
	}
	return table;
}

void ReductionExpr::setEpochVersions(Space *space, int epoch) {
	right->setEpochVersions(space, epoch);
}

List<FieldAccess*> *ReductionExpr::getTerminalFieldAccesses() { return right->getTerminalFieldAccesses(); }

void ReductionExpr::translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space) {
	if (op == SUM) stream << "Sum";
	else if (op == PRODUCT) stream << "Product";
	else if (op == MAX) stream << "Max";
	else if (op == MIN) stream << "Min";
	else if (op == AVG) stream << "Avg";
	else if (op == MAX_ENTRY) stream << "MaxEntry";
	else if (op == MIN_ENTRY) stream << "MinEntry";
}

void ReductionExpr::generateCode(std::ostringstream &stream, int indentLevel, Space *space) {
	
	std::string stmtSeparator = ";\n";
	std::ostringstream indent;
	for (int i = 0; i < indentLevel; i++) indent << '\t';
	
	if (op == SUM) {
		stream << indent.str() << "Sum += ";
		right->translate(stream, indentLevel, 0, space);
		stream << stmtSeparator;
	} else if (op == PRODUCT) {
		stream << indent.str() << "Product *= ";
		right->translate(stream, indentLevel, 0, space);
		stream << stmtSeparator;
	} else if (op == AVG) {
		stream << indent.str() << "Avg += ";
		right->translate(stream, indentLevel, 0, space);
		stream << stmtSeparator;
		stream << indent.str() << "Count++" << stmtSeparator;
	} else if (op == MAX) {
		stream << indent.str() << "if (Max < ";
		right->translate(stream, indentLevel, 0, space);
		stream << ") Max = ";
		right->translate(stream, indentLevel, 0, space);
		stream << stmtSeparator;
	} else if (op == MIN) {
		stream << indent.str() << "if (Min > ";
		right->translate(stream, indentLevel, 0, space);
		stream << ") Min = ";
		right->translate(stream, indentLevel, 0, space);
		stream << stmtSeparator;
	} else if (op == MAX_ENTRY) {
		// For now, we are assuming that max/min entry reductions are done within loops 
		// that iterate over a single index only. In the future we can lift of this 
		// restriction if needed. Then the entry can be an static array of indexes with
		// dimensionality that equals to the number of indexes in the encompassing loop.
		const char *indexName = reductionLoop->getIndexNames()->Nth(0);
		const char *transformedName 
				= ntransform::NameTransformer::transformer->getTransformedName(
						indexName, false, true, Type::intType);
		stream << indent.str() << "if (Max < ";
		right->translate(stream, indentLevel, 0, space);
		stream << ") {\n";
		stream << indent.str() << '\t' << "Max = ";
		right->translate(stream, indentLevel, 0, space);
		stream << stmtSeparator;
		stream << indent.str() << '\t' << "MaxEntry = " << transformedName;
		stream << stmtSeparator;
		stream << indent.str() << "}\n";	
	} else if (op == MIN_ENTRY) {
		// The same comment above applies for this case too
		const char *indexName = reductionLoop->getIndexNames()->Nth(0);
		const char *transformedName 
				= ntransform::NameTransformer::transformer->getTransformedName(
						indexName, false, true, Type::intType);
		stream << indent.str() << "if (Min > ";
		right->translate(stream, indentLevel, 0, space);
		stream << ") {\n";
		stream << indent.str() << '\t' << "Min = ";
		right->translate(stream, indentLevel, 0, space);
		stream << stmtSeparator;
		stream << indent.str() << '\t' << "MinEntry = " << transformedName;
		stream << stmtSeparator;
		stream << indent.str() << "}\n";	
	}
}

void ReductionExpr::setupForReduction(std::ostringstream &stream, int indentLevel) {
	std::ostringstream indent;
	for (int i = 0; i < indentLevel; i++) indent << '\t';
	std::string stmtSeparator = ";\n";
	if (op == SUM) {
		stream << indent.str() << type->getCType() << " Sum = 0" << stmtSeparator;
	} else if (op == PRODUCT) {
		stream << indent.str() << type->getCType() << " Product = 1" << stmtSeparator;
	} else if (op == MAX) {
		stream << indent.str() << type->getCType() << " Max = ";
		if (type == Type::charType) stream << "CHAR_MIN";
		else if (type == Type::intType) stream << "INT_MIN";
		else stream << "LONG_MIN";
		stream << stmtSeparator;
	} else if (op == MIN) {
		stream << indent.str() << type->getCType() << " Min = ";
		if (type == Type::charType) stream << "CHAR_MAX";
		else if (type == Type::intType) stream << "INT_MAX";
		else stream << "LONG_MAX";
		stream << stmtSeparator;
	} else if (op == AVG) {
		stream << indent.str() << "int Count = 0" << stmtSeparator;
		stream << indent.str() << type->getCType() << " Avg = 0" << stmtSeparator;
	} else if (op == MAX_ENTRY) {
		stream << indent.str() << "int MaxEntry = 0" << stmtSeparator;
		stream << indent.str() << right->getType()->getCType() << " Max = ";
		if (type == Type::charType) stream << "CHAR_MIN";
		else if (type == Type::intType) stream << "INT_MIN";
		else stream << "LONG_MIN";
		stream << stmtSeparator;
	} else if (op == MIN_ENTRY) {
		stream << indent.str() << "int MinEntry = 0" << stmtSeparator;
		stream << indent.str() << right->getType()->getCType() << " Min = ";
		if (type == Type::charType) stream << "CHAR_MAX";
		else if (type == Type::intType) stream << "INT_MAX";
		else stream << "LONG_MAX";
		stream << stmtSeparator;
	}
}

void ReductionExpr::finalizeReduction(std::ostringstream &stream, int indentLevel) {
	if (op == AVG) {
		std::ostringstream indent;
		for (int i = 0; i < indentLevel; i++) indent << '\t';
		std::string stmtSeparator = ";\n";
		stream << indent.str() << "Avg = Avg / Count" << stmtSeparator;
	}
}
