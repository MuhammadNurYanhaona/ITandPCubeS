#include "../../../../../../frontend/src/syntax/ast.h"
#include "../../../../../../frontend/src/syntax/ast_expr.h"
#include "../../../../../../frontend/src/semantics/task_space.h"
#include "../../../../../../frontend/src/semantics/partition_function.h"

#include <string>
#include <sstream>
#include <cstdlib>

//--------------------------------------------------- Strided Block ----------------------------------------------/

const char *StridedBlock::getTransformedIndex(int dimensionNo, const char *origIndexName, bool copyMode) {
	
	DataDimensionConfig *argument = getArgsForDimension(dimensionNo);
	Node *dividingArg = argument->getDividingArg();
	bool argNeeded = false;
	int argValue = 0;
	const char *argName = NULL;
	Identifier *identifier = dynamic_cast<Identifier*>(dividingArg);
	if (identifier != NULL) {
		argNeeded = true;
		argName = identifier->getName();
	} else {
		IntConstant *constant = dynamic_cast<IntConstant*>(dividingArg);
		argValue = constant->getValue();
	}

	std::ostringstream expr;
	expr << "((" << origIndexName << " / (";
	expr << "partConfig.count " << " * ";
	if (argNeeded) {
		expr << "partition." << argName;
		expr << "))";
		expr << " * partition." << argName;
		expr << " + " << origIndexName << " % " << "partition." << argName << ")";
	} else {
		expr << argValue;
		expr << "))";
		expr << " * " << argValue;
		expr << " + " << origIndexName << " % " << argValue << ")";
	}

	return strdup(expr.str().c_str());	
}

const char *StridedBlock::getOriginalIndex(int dimensionNo, const char *xformIndexName, bool copyMode) {
	
	DataDimensionConfig *argument = getArgsForDimension(dimensionNo);
	Node *dividingArg = argument->getDividingArg();
	std::ostringstream sizeParam;
	Identifier *identifier = dynamic_cast<Identifier*>(dividingArg);
	if (identifier != NULL) {
		sizeParam << "partition.";
		sizeParam << identifier->getName();
	} else {
		IntConstant *constant = dynamic_cast<IntConstant*>(dividingArg);
		sizeParam << constant->getValue();
	}

	std::ostringstream expr;
	expr << "((" << xformIndexName << " / " << sizeParam.str() << ")";
	expr << " * partConfig.count";
	expr << " + partConfig.index) * " << sizeParam.str();
	expr << " + " << xformIndexName << " % " << sizeParam.str();
	return strdup(expr.str().c_str());	
}

const char *StridedBlock::getInclusionTestExpr(int dimensionNo, const char *origIndexName, bool copyMode) {

	DataDimensionConfig *argument = getArgsForDimension(dimensionNo);
	Node *dividingArg = argument->getDividingArg();
	std::ostringstream sizeParam;
	Identifier *identifier = dynamic_cast<Identifier*>(dividingArg);
	if (identifier != NULL) {
		sizeParam << "partition.";
		sizeParam << identifier->getName();
	} else {
		IntConstant *constant = dynamic_cast<IntConstant*>(dividingArg);
		sizeParam << constant->getValue();
	}

	std::ostringstream expr;
	expr << "(" << origIndexName << " % (";
	expr << sizeParam.str() << " * partConfig.count))";
	expr << " / " << sizeParam.str() << " == partConfig.index";
	return strdup(expr.str().c_str());	
}

const char *StridedBlock::getImpreciseBoundOnXformedIndex(int dimensionNo, const char *index, 
		bool lowerBound, bool copyMode, int indentLevel) {
	
	std::ostringstream indent;
	for (int i = 0; i < indentLevel; i++) indent << '\t';
	
	DataDimensionConfig *argument = getArgsForDimension(dimensionNo);
	Node *dividingArg = argument->getDividingArg();
	std::ostringstream sizeParam;
	Identifier *identifier = dynamic_cast<Identifier*>(dividingArg);
	if (identifier != NULL) {
		sizeParam << "partition.";
		sizeParam << identifier->getName();
	} else {
		IntConstant *constant = dynamic_cast<IntConstant*>(dividingArg);
		sizeParam << constant->getValue();
	}

	// first check if the index falls within the stride range of the current partition; if it does
	// then compute the transformed index
	std::ostringstream expr;
	expr << "(" << getInclusionTestExpr(dimensionNo, index, copyMode) << ") ";
	expr << '\n' << indent.str() << "\t\t\t\t";
	expr << "? (" << getTransformedIndex(dimensionNo, index, copyMode) << ")";

	// then consider the case that  the index is not within the stride range of the current partition;
	// in that case we need to calculate an imprecise boundary
	expr << '\n' << indent.str() << "\t\t\t\t";
	expr << " : (";
	if (lowerBound) {
		expr << "(" << index << " % (" << sizeParam.str() << " * partConfig.count) > ";
		expr << sizeParam.str() << " * partConfig.index)";
		// when index is at the right of the current stride range boundary
		expr << '\n' << indent.str() << "\t\t\t\t\t\t";
		expr << " ? ((" << index << " / (";
		expr << "partConfig.count " << " * " << sizeParam.str();
		expr << ")) * " << sizeParam.str() << " + " << sizeParam.str() << " - 1)";
		// when index is at the left of the current stride range boundary
		expr << '\n' << indent.str() << "\t\t\t\t\t\t";
		expr << " : ((" << index << " / (";
		expr << "partConfig.count " << " * " << sizeParam.str();
		expr << ")) * " << sizeParam.str() << " - 1)";				
	} else {
		expr << "(" << index << " % (" << sizeParam.str() << " * partConfig.count) > ";
		expr << sizeParam.str() << " * partConfig.index)";
		// when index is at the right of the current stride range boundary
		expr << '\n' << indent.str() << "\t\t\t\t\t\t";
		expr << " ? ((" << index << " / (";
		expr << "partConfig.count " << " * " << sizeParam.str();
		expr << ") + 1) * " << sizeParam.str() << ")";
		// when index is at the left of the current stride range boundary
		expr << '\n' << indent.str() << "\t\t\t\t\t\t";
		expr << " : ((" << index << " / (";
		expr << "partConfig.count " << " * " << sizeParam.str();
		expr << ")) * " << sizeParam.str() << ")";				

	}
	expr << ")";

	// finally, wrap up the entire expression within another expression that protect against modifying
	// negative original index -- that represents an invalid index -- that may arise due to index 
	// xformation took place on some earlier step
	std::ostringstream protectExpr;
	protectExpr << "(" << index << " < 0)";
	protectExpr << '\n' << indent.str() << "\t\t";
	protectExpr << " ? (";
	if (lowerBound) protectExpr << "-1";
	else protectExpr << "0";
	protectExpr << ")";
	protectExpr << '\n' << indent.str() << "\t\t";
	protectExpr << " : (" << expr.str() << ")"; 

	return strdup(protectExpr.str().c_str());	
}

//----------------------------------------------------- Strided --------------------------------------------------/

const char *Strided::getTransformedIndex(int dimensionNo, const char *origIndexName, bool copyMode) {
	std::ostringstream expr;
	expr << "(" << origIndexName << " / " << "partConfig.count)";
	return strdup(expr.str().c_str());
}

const char *Strided::getOriginalIndex(int dimensionNo, const char *xformIndexName, bool copyMode) {
	std::ostringstream expr;
	expr << "(" << "partConfig.index" <<  " + ";
	expr << xformIndexName << " * " << "partConfig.count" << ")";
	return strdup(expr.str().c_str());
}

const char *Strided::getInclusionTestExpr(int dimensionNo, const char *origIndexName, bool copyMode) {
	std::ostringstream expr;
	expr << "(" << origIndexName << " % " << "partConfig.count == partConfig.index)";
	return strdup(expr.str().c_str());
}

const char *Strided::getImpreciseBoundOnXformedIndex(int dimensionNo, const char *index, bool lowerBound, 
		bool copyMode, int indentLevel) {

	std::ostringstream indent;
	for (int i = 0; i < indentLevel; i++) indent << '\t';
	
	// generate the expression for appropriate lower/upper bound
	std::ostringstream impreciseExpr;
	if (lowerBound) {
		impreciseExpr << "((" << index << " \% partConfig.count ) >= partConfig.index) ";
		// when index falls on the right of the stride range or on the stride
		impreciseExpr << '\n' << indent.str() << "\t\t\t\t";
		impreciseExpr << "? (" << index << " / " << "partConfig.count" << ") ";
		// when index falls on the left of the stride range	
		impreciseExpr << '\n' << indent.str() << "\t\t\t\t";
		impreciseExpr << ": (" << index << " / " << "partConfig.count" << " - 1)";
	} else {
		impreciseExpr << "((" << index << " \% partConfig.count ) > partConfig.index) ";
		// when index falls on the right of the stride range	
		impreciseExpr << '\n' << indent.str() << "\t\t\t\t";
		impreciseExpr << "? (" << index << " / " << "partConfig.count" << " + 1) ";
		// when index falls on the left of the stride range or on the stride
		impreciseExpr << '\n' << indent.str() << "\t\t\t\t";
		impreciseExpr << ": (" << index << " / " << "partConfig.count)";
	}

	// wrap the previous expression within another expression for safe-keeping against negative 
	// original index
	std::ostringstream expr;
	expr << "(" << index << " < 0)";
	expr << '\n' << indent.str() << "\t\t";
	expr << " ? (";
	if (lowerBound) expr << "-1";
	else expr << "0";
	expr << ")";
	expr << '\n' << indent.str() << "\t\t";
	expr << " : (" << impreciseExpr.str() << ")"; 
	
	return strdup(expr.str().c_str());
}
