#include "../syntax/ast_stmt.h"
#include "../syntax/ast_expr.h"
#include "../semantics/task_space.h"
#include "../../../common-libs/utils/list.h"
#include <sstream>
#include <cstdlib>

void Stmt::generateCode(std::ostringstream &stream, int indentLevel, Space *space) {}
void StmtBlock::generateCode(std::ostringstream &stream, int indentLevel, Space *space) {}
void ConditionalStmt::generateCode(std::ostringstream &stream, int indentLevel, bool first, Space *space) {}
void IfStmt::generateCode(std::ostringstream &stream, int indentLevel, Space *space) {}
void PLoopStmt::generateCode(std::ostringstream &stream, int indentLevel, Space *space) {}
void SLoopStmt::generateCode(std::ostringstream &stream, int indentLevel, Space *space) {}
void WhileStmt::generateCode(std::ostringstream &stream, int indentLevel, Space *space) {}
void ReductionStmt::generateCode(std::ostringstream &stream, int indentLevel, Space *space) {}
void ExternCodeBlock::generateCode(std::ostringstream &stream, int indentLevel, Space *space) {}
void ReturnStmt::generateCode(std::ostringstream &stream, int indentLevel, Space *space) {}

LogicalExpr *IndexRangeCondition::getRestrictions() { return NULL; }
void LoopStmt::declareVariablesInScope(std::ostringstream &stream, int indentLevel) {}
void LoopStmt::generateIndexLoops(std::ostringstream &stream, int indentLevel,
		Space *space, Stmt *body, List<LogicalExpr*> *indexRestrictions) {}
List<LogicalExpr*> *LoopStmt::getApplicableExprs(Hashtable<const char*> *indexesInvisible,
		List<LogicalExpr*> *currentExprList,
		List<LogicalExpr*> *remainingExprList) { return NULL; }
void LoopStmt::initializeReductionLoop(std::ostringstream &stream, int indentLevel, Space *space) {}
void LoopStmt::finalizeReductionLoop(std::ostringstream &stream, int indentLevel, Space *space) {}
void LoopStmt::performReduction(std::ostringstream &stream, int indentLevel, Space *space) {}
List<LogicalExpr*> *PLoopStmt::getIndexRestrictions() { return NULL; }
