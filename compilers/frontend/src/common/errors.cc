#include "errors.h"
#include <iostream>
#include <sstream>
#include <stdarg.h>
#include <stdio.h>
using namespace std;

#include "../syntax/ast.h"
#include "../syntax/ast_type.h"
#include "../syntax/ast_expr.h"
#include "../lex/scanner.h" // for GetLineNumbered

int ReportError::numErrors = 0;

void ReportError::UnderlineErrorInLine(const char *line, yyltype *pos) {
    if (!line) return;
    cerr << line << endl;
    for (int i = 1; i <= pos->last_column; i++)
        cerr << (i >= pos->first_column ? '^' : ' ');
    cerr << endl;
}

void ReportError::OutputError(yyltype *loc, string msg) {
    numErrors++;
    fflush(stdout); // make sure any buffered text has been output
    if (loc) {
        cerr << endl << "*** Error line " << loc->first_line << "." << endl;
        UnderlineErrorInLine(GetLineNumbered(loc->first_line), loc);
    } else
        cerr << endl << "*** Error." << endl;
    cerr << "*** " << msg << endl << endl;
}

void ReportError::Formatted(yyltype *loc, const char *format, ...) {
	va_list args;
    	char errbuf[2048];
    	va_start(args, format);
    	vsprintf(errbuf,format, args);
    	va_end(args);
    	OutputError(loc, errbuf);
}

void ReportError::OptionalErrorReport(yyltype *loc, bool suppressFailure, const char *format, ...) {
	if (!suppressFailure) {    
		va_list args;
    		char errbuf[2048];
    		va_start(args, format);
    		vsprintf(errbuf,format, args);
    		va_end(args);
    		OutputError(loc, errbuf);
	}
}

void ReportError::TypeInferenceError(Identifier *id, bool suppressFailure) {
	OptionalErrorReport(id->GetLocation(), suppressFailure, 
			"Type inference failed for '%s'", id->getName());
}

void ReportError::UndeclaredTypeError(Identifier *variable, Type *type, 
		const char *prefix, bool suppressFailure) {
	if (prefix == NULL) {
		prefix = "";
	}
	OptionalErrorReport(type->GetLocation(), suppressFailure,
			"Unknown %stype '%s' for variable '%s'", 
			prefix, type->getName(), variable->getName());
}

void ReportError::ConflictingDefinition(Identifier *id, bool suppressFailure) {
	OptionalErrorReport(id->GetLocation(), suppressFailure, 
			"'%s' has already been declared in this scope", id->getName());
}

void ReportError::NotReductionType(Identifier *id, bool suppressFailure) {
	OptionalErrorReport(id->GetLocation(), suppressFailure,
			"'%s' is not a task-global Reduction variable", id->getName());
}

void ReportError::InferredAndActualTypeMismatch(yyltype *loc, Type *inferred, 
		Type *actual, bool suppressFailure) {
	OptionalErrorReport(loc, suppressFailure,
			"Inferred type '%s' does not comply with actual expression type '%s'", 
			inferred->getName(), actual->getName());
}

void ReportError::UnknownExpressionType(Expr *expr, bool suppressFailure) {
	OptionalErrorReport(expr->GetLocation(), suppressFailure, "Unknown expression type");
}

void ReportError::ConflictingArrayDimensionCounts(yyltype *loc, const char *name, 
                        int oldDimension, int currDimension, bool suppressFailure) {
	OptionalErrorReport(loc, suppressFailure,
                        "Conflicting access of '%s' as a %dD array that has been accessed as a %dD array before",
                        name, currDimension, oldDimension);
}

void ReportError::UnsupportedOperand(Expr* expr, Type *type, 
		const char *operation, bool suppressFailure) {
	OptionalErrorReport(expr->GetLocation(), suppressFailure,
			"Unsupported operand type '%s' in %s", 
			type->getName(), operation);
}

void ReportError::InvalidExprType(Expr *expr, Type *type, bool suppressFailure) {
	OptionalErrorReport(expr->GetLocation(), suppressFailure, "Unsupported type '%s'", type->getName());
}

void ReportError::TypeMixingError(Expr* expr, Type *type1, 
			Type *type2, const char *operation, bool suppressFailure) {
	OptionalErrorReport(expr->GetLocation(), suppressFailure,
			"Cannot mix '%s' type and '%s' type in %s",
			type1->getName(), type2->getName(), operation);
}

void ReportError::IncompatibleTypes(yyltype *loc, 
		Type *actual, Type *expected, bool suppressFailure) {
	OptionalErrorReport(loc, suppressFailure, 
			"Expected type '%s' but got type '%s'", expected->getName(), actual->getName());
}

void ReportError::UndefinedSymbol(Identifier *id, bool suppressFailure) {
	OptionalErrorReport(id->GetLocation(), suppressFailure, 
			"No definition found for '%s'", id->getName());
}

void ReportError::UndefinedSymbol(yyltype *loc, const char *name, bool suppressFailure) {
	OptionalErrorReport(loc, suppressFailure, "No definition found for '%s'", name);
}

void ReportError::UnspecifiedTaskToExecute(yyltype *loc, bool suppressFailure) {
	OptionalErrorReport(loc, suppressFailure, "A valid 'task' name and 'environment' must exist to execute");
}

void ReportError::WrongSymbolType(Identifier *id, const char *expectedType, bool suppressFailure) {
	OptionalErrorReport(id->GetLocation(), suppressFailure, 
			"Symbol type for '%s' does not match; expecting '%s'", 
			id->getName(), expectedType);
}

void ReportError::WrongSymbolType(yyltype *loc, const char *name, const char *expectedType, bool suppressFailure) {
	OptionalErrorReport(loc, suppressFailure, "Symbol type for '%s' does not match; expecting '%s'", 
			name, expectedType);
}

void ReportError::NoSuchFieldInBase(Identifier *field, Type *type, bool suppressFailure) {
	OptionalErrorReport(field->GetLocation(), suppressFailure, 
			"'%s' has no field named '%s'", type->getName(), field->getName());
}

void ReportError::NonExistingDimensionInArray(Identifier *id, 
			int dimensionality, int dimension, bool suppressFailure) {
	OptionalErrorReport(id->GetLocation(), suppressFailure, 
			"Trying to access dimension %d from a %dD array", 
			dimension, dimensionality);
}

void ReportError::NonLValueInAssignment(Expr *expr, bool suppressFailure) {
	OptionalErrorReport(expr->GetLocation(), suppressFailure, 
			"Left side of an assignment must be either an array or a field");
}

void ReportError::InvalidArrayAccess(yyltype *loc, Type *actualType, bool suppressFailure) {
	if (actualType != NULL) {
		OptionalErrorReport(loc, suppressFailure,
				"Non-array variable of type '%s' accessed like an array", 
				actualType->getName());
	} else {
		OptionalErrorReport(loc, suppressFailure, "Invalid array access");
	}
}

void ReportError::TooFewOrTooManyParameters(Identifier *name, int actual, 
		int expected, bool suppressFailure) {
	OptionalErrorReport(name->GetLocation(), suppressFailure,
			"Function '%s' expects %d arguments but provided %d", 
			name->getName() , expected, actual);
}

void ReportError::TaskNameRequiredInEnvironmentCreate(yyltype *loc, bool suppressFailure) {
	OptionalErrorReport(loc, suppressFailure, "The 'name' of the task must be specified in environment creation");
}

void ReportError::InvalidObjectTypeInNew(Type *type, bool suppressFailure) {
	OptionalErrorReport(type->GetLocation(), suppressFailure, 
			"A user defined or built-in tuple type is expected instead of '%s'", 
			type->getName());
}

void ReportError::TooManyParametersInNew(yyltype *loc, const char *objectType, int actual, 
		int expected, bool suppressFailure) {
	OptionalErrorReport(loc, suppressFailure,
			"An object of type '%s' has only %d fields but initialization arguments are provided for %d", 
			 objectType, expected, actual);
}

void ReportError::UndefinedTask(yyltype *loc, const char *name, bool suppressFailure) {
	OptionalErrorReport(loc, suppressFailure, "Task %s is undefined", name);
}

void ReportError::NonArrayInIndexedIteration(Identifier *id, Type *type, bool suppressFailure) {
	OptionalErrorReport(id->GetLocation(), suppressFailure, 
			"Indexed iteration can be done only over arrays but '%s' is of type %s", id->getName(), type->getName());
}

void ReportError::UnknownIndexToArrayAssociation(Identifier *index, Identifier *array, bool suppressFailure) {
	OptionalErrorReport(index->GetLocation(), suppressFailure, 
			"Do not know what dimension of '%s' index '%s' is referring to. Use more specific syntax clarifying the association", 
			array->getName(), index->getName());
}

void ReportError::InvalidInitArg(yyltype *loc, const char *object, const char *arg, bool suppressFailure) {
	OptionalErrorReport(loc, suppressFailure, "Class '%s' has no property named '%s'", object, arg);
}

void ReportError::CouplingOfReductionWithOtherExpr(yyltype *loc, bool suppressFailure) {
	OptionalErrorReport(loc, suppressFailure, "Current compiler cannot process other operations alongside a reduction within the same loop");
}

void ReportError::ReductionOutsideForLoop(yyltype *loc, bool suppressFailure) {
	OptionalErrorReport(loc, suppressFailure, "Reduction expression must be within an encompassing loop statement");
}

void ReportError::ReductionRangeInvalid(yyltype *loc, const char *rdRootLps, const char *rdBoundaryLps, bool suppressFailure) {
	OptionalErrorReport(loc, suppressFailure, 
			"The root LPS of the reduction, Space %s, is situated above the encircling LPS, Space %s", 
			rdRootLps, rdBoundaryLps);
}

void ReportError::NotAnEnvironment(yyltype *loc, Type *type, bool suppressFailure) {
	OptionalErrorReport(loc, suppressFailure, "'%s' is not an environment type", type->getName());
}

void ReportError::NotAConstant(yyltype *loc, const char *constType, bool suppressFailure) {
	OptionalErrorReport(loc, suppressFailure, "Expression should be a constant of type '%s'", constType);
}

void ReportError::SpaceNotFound(yyltype *loc, char spaceName) {
	Formatted(loc, "No space with name '%c' found in the Partition section", spaceName);
} 

void ReportError::InvalidSpaceNesting(yyltype *loc, const char *nestedSpace, const char *upperSpace) {
	Formatted(loc, "Cannot nest a Space %s computation stage inside Space %s", nestedSpace, upperSpace);
}

void ReportError::RepeatLoopAdvanceImposssible(yyltype *loc, const char *spaceName, const char *repeatLoopSpace) {
	Formatted(loc, "Repeat loop cannot advance beyond Space %s but contain a Space %s computation",
			repeatLoopSpace, spaceName);
}

void ReportError::SubpartitionRepeatMeesing(yyltype *loc, const char *spaceName, const char *repeatRoot) {
	Formatted(loc, "A Space %s computation stage must be within a sub-partition repeat loop traversing Space %s", 
			spaceName, repeatRoot);
}

void ReportError::ImpermissibleRepeat(yyltype *loc, const char *spaceName, const char *repeatLoopSpace) {
	Formatted(loc, "Cannot nest a repeat loop spanning up to Space %s inside Space %s", repeatLoopSpace, spaceName);
}

void ReportError::SubpartitionRepeatNotSupported(yyltype *loc, char spaceName) {
	Formatted(loc, "Space %c is not sub-partitioned to have a repeat loop like this", spaceName);
}

void ReportError::RepeatBeginningInvalid(yyltype *loc, const char *allowedFurthestRoot) {
	Formatted(loc, "The beginning stage of repeat should be %s or any that comes after that", allowedFurthestRoot);
}

void ReportError::ArrayPartitionUnknown(yyltype *loc, const char *arrayName, const char *stageName, const char spaceId) {
	Formatted(loc, "Array '%s' cannot be used in %s as '%s' is not partitioned in corresponding Space %c", 
		arrayName, stageName, arrayName, spaceId);
}

void ReportError::DuplicateSpaceDefinition(yyltype *loc, char spaceName) {
	Formatted(loc, "Duplicate definition found for Space %c", spaceName);
}

void ReportError::ParentSpaceNotFound(yyltype *loc, char parentSpace, char spaceName, bool isSubpartition) {
	if (isSubpartition) {
		Formatted(loc, "Parent space %c sub-partition must be defined before child space %c", 
				parentSpace, spaceName);
	} else { 
		Formatted(loc, "Parent space %c must be defined before child space %c", 
				parentSpace, spaceName);
	}
}

void ReportError::NonTaskGlobalArrayInPartitionSection(Identifier *id) {
	Formatted(id->GetLocation(), 
			"'%s' is not a task-global array; therefore cannot be partitioned", 
			id->getName());
}

void ReportError::InvalidSpaceCoordinateSystem(yyltype *loc, char spaceName, int dimensions, bool isSubpartition) {
	if (isSubpartition) {
		Formatted(loc, "%d dimensions of Space %c subpartition are not properly fulfilled by the data structures", 
				dimensions, spaceName);
	} else {
		Formatted(loc, "%d dimensions of space %c are not properly fulfilled by the data structures", 
				dimensions, spaceName);
	}
}

void ReportError::UnpartitionedDataInPartitionedSpace(yyltype *loc, char spaceName, int dimensionality) {
	Formatted(loc, "Unpartitioned data structures cannot reside in the %dD Space %c", 
			spaceName, dimensionality);
}

void ReportError::SubpartitionDimensionsNotPositive(yyltype *loc) {
	Formatted(loc, "A subpartition must have a nonzero positive dimension count");
}

void ReportError::SubpartitionOrderConflict(yyltype *loc) {
	Formatted(loc, "An unordered subpartition space cannot have order dependent data partition");
}

void ReportError::DimensionMissingOrInvalid(yyltype *loc) {
	Formatted(loc, "The underlying dimension does not exist or cannot be partitionned");
}

void ReportError::DimensionMissingOrInvalid(Identifier *id, int dimension) {
	Formatted(id->GetLocation(), "Dimension %d of '%s' does not exist or cannot be partitionned", 
			dimension, id->getName());
}

void ReportError::SubpartitionedStructureMissingInParentSpace(yyltype *loc, const char *name) {
	Formatted(loc, "Variable '%s' is not in the parent space of the subpartition", name);
}

void ReportError::ParentForDataStructureNotFound(yyltype *loc, char parentSpace, const char *variable) {
	Formatted(loc, "Parent space %c for variable '%s' must be defined first", parentSpace, variable);
}

void ReportError::ParentDataStructureNotFound(yyltype *loc, char parentSpace, const char *variable) {
	Formatted(loc, "Parent space %c does not have variable '%s'", parentSpace, variable);
}

void ReportError::InvalidPartitionArgument(yyltype *loc) {
	Formatted(loc, "Only a partition parameter or constant can be used as an argument for a partition function.");
}

void ReportError::SpaceAndDataHierarchyConflict(Identifier *id, const char *spaceParent, const char *dataParent) {
	Formatted(id->GetLocation(), 
			"Space parent %s has '%s' but is different from the parent partition %s of '%s'", 
			spaceParent, id->getName(), dataParent, id->getName());
}

void ReportError::TooFineGrainedVariablePartition(Identifier *id) {
	Formatted(id->GetLocation(), "dimensionality of the partition for variable '%s' exceeds that of the space", 
			id->getName());
}

void ReportError::ArgumentCountMismatchInPartitionFunction(const char *functionName) {
	Formatted(NULL, "Internal error in partition function implementation. Argument count does not match what expected");
}

void ReportError::UnknownPartitionFunction(yyltype *loc, const char *functionName) {
	Formatted(loc, "Partition function %s is undefined", functionName);
}

void ReportError::ArgumentMissingInPartitionFunction(yyltype *loc, const char *functionName, const char* argument) {
	Formatted(loc, "Argument '%s' is needed for each partitioned dimension in %s", argument, functionName);
}

void ReportError::InvalidPadding(yyltype *loc, const char *functionName) {
	const char* message = "For uniform padding in front and back padding arguments count should match partitioning arguments count. For non-uniform padding the former should be twice of the latter in %s.";
	Formatted(loc, message, functionName);
}

void ReportError::PartitionArgumentsNotSupported(yyltype *loc, const char *functionName) {
	Formatted(loc, "Partition function %s does not support partitioning arguments", functionName);
}
        
void ReportError::PaddingArgumentsNotSupported(yyltype *loc, const char *functionName) {
	Formatted(loc, "Partition function %s does not support padding arguments", functionName);
}

void ReportError::NotLValueinAssignment(yyltype *loc) {
	Formatted(loc, "Left side of an assignment must be a field or array access");
}

/* Function: yyerror()
 * -------------------
 * Standard error-reporting function expected by yacc. Our version merely
 * just calls into the error reporter above, passing the location of
 * the last token read. If you want to suppress the ordinary "parse error"
 * message from yacc, you can implement yyerror to do nothing and
 * then call ReportError::Formatted yourself with a more descriptive 
 * message.
 */
void yyerror(const char *msg) {
    ReportError::Formatted(&yylloc, "%s", msg);
}
