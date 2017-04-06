#include "../syntax/ast_expr.h"
#include "../semantics/task_space.h"
#include "../../../common-libs/utils/list.h"
#include <sstream>

void Expr::generateCode(std::ostringstream &stream, int indentLevel, Space *space) {}
void Expr::translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space) {}

void IntConstant::translate(std::ostringstream &s, int i, int c, Space *space) {}
void FloatConstant::translate(std::ostringstream &s, int i, int c, Space *space) {}
void DoubleConstant::translate(std::ostringstream &s, int i, int c, Space *space) {}
void BoolConstant::translate(std::ostringstream &s, int i, int c, Space *space) {}
void CharConstant::translate(std::ostringstream &s, int i, int c, Space *space) {}
void StringConstant::translate(std::ostringstream &s, int i, int c, Space *space) {}
void ReductionVar::translate(std::ostringstream &s, int i, int c, Space *space) {}

void ArithmaticExpr::translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space) {}

void LogicalExpr::translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space) {}
List<LogicalExpr*> *LogicalExpr::getANDBreakDown() { return NULL; }
List<LogicalExpr*> *LogicalExpr::getIndexRestrictExpr(List<LogicalExpr*> *exprList,
		std::ostringstream &stream,
		const char *indexVar, const char *rangeExpr,
		int indentLevel, Space *space,
		bool xformedArrayRange, const char *arrayName, int dimensionNo) { return NULL; }
int LogicalExpr::isLoopRestrictExpr(const char *loopIndex) { return -1; }
bool LogicalExpr::transformIndexRestriction(std::ostringstream &stream,
		const char *varName, const char *arrayName, int dimensionNo,
		int indentLevel, Space *space,
		bool normalizedToMinOfRange, bool lowerBound) { return false; }

void EpochExpr::translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space) {}

void FieldAccess::translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space) {}
void FieldAccess::translateIndex(std::ostringstream &stream, const char *array, int dimension) {}
bool FieldAccess::isEnvArrayDim() { return false; }
void FieldAccess::translateEnvArrayDim(std::ostringstream &stream,
		int indentLevel,
		int currentLineLength, Space *space) {}

const char *RangeExpr::getIndexExpr() { return NULL; }
const char *RangeExpr::getRangeExpr(Space *space) { return NULL; }
const char *RangeExpr::getStepExpr(Space *space) { return NULL; }
void RangeExpr::translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space) {}
const char *RangeExpr::getBaseArrayForRange(Space *executionSpace) { return NULL; }
int RangeExpr::getDimensionForRange(Space *executionSpace) { return -1; }
void RangeExpr::generateLoopForRangeExpr(std::ostringstream &stream,
		int indentation, Space *space, const char *loopbounRestrictCond) {}
void RangeExpr::translateArrayRangeExprCheck(std::ostringstream &stream, int indentLevel, Space *space) {}
void RangeExpr::generateAssignmentExprForXformedIndex(std::ostringstream &stream,
		int indentLevel, Space *space) {}

void AssignmentExpr::translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space) {}
void AssignmentExpr::generateCode(std::ostringstream &stream, int indentLevel, Space *space) {}

void IndexRange::translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space) {}

void ArrayAccess::generate1DIndexAccess(std::ostringstream &stream, int indentLevel,
		const char *array, ArrayType *type, Space *space) {}
void ArrayAccess::translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space) {}
void ArrayAccess::generateXformedIndex(std::ostringstream &stream, int indentLevel,
		const char *indexExpr,
		const char *arrayName, int dimensionNo, Space *space) {}

void FunctionCall::translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space) {}

void TaskInvocation::generateCode(std::ostringstream &stream, int indentLevel, Space *space) {}

void NamedArgument::generateAssignment(Expr *object, std::ostringstream &stream, int indentLevel);

void ObjectCreate::translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space) {}
void ObjectCreate::generateCodeForProperties(Expr *object, std::ostringstream &stream, int indentLevel) {}
bool ObjectCreate::isDynamicArrayCreate(Expr *candidateExpr) { return false; }
