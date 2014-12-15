/* File: errors.h
 * --------------
 * This file defines an error-reporting class with a set of already
 * implemented static methods for reporting the standard IT errors.
 */

#ifndef _H_errors
#define _H_errors

#include <string>
#include "location.h"
#include "ast.h"
#include "ast_type.h"
#include "ast_expr.h"

using std::string;

class ReportError
{
 public:
  	// Specific error reporting
  	static void TypeInferenceError(Identifier *id, bool suppressFailure);
  	static void UndeclaredTypeError(Identifier *variable, Type *type, const char *prefix, bool suppressFailure);
	static void ConflictingDefinition(Identifier *id, bool suppressFailure);
	static void InferredAndActualTypeMismatch(yyltype *loc, Type *inferred, Type *actual, bool suppressFailure);
	static void UnknownExpressionType(Expr *expr, bool suppressFailure);
	static void UnsupportedOperand(Expr *expr, Type *type, const char *operation, bool suppressFailure);
	static void TypeMixingError(Expr *expr, Type *type1, Type *type2, const char *operation, bool suppressFailure);
	static void IncompatibleTypes(yyltype *loc, Type *actual, Type *expected, bool suppressFailure);
	static void UndefinedSymbol(Identifier *id, bool suppressFailure);
	static void UndefinedSymbol(yyltype *loc, const char *name, bool suppressFailure);
	static void WrongSymbolType(Identifier *id, const char *expectedType, bool suppressFailure);
	static void WrongSymbolType(yyltype *loc, const char *name, const char *expectedType, bool suppressFailure);
	static void NoSuchFieldInBase(Identifier *field, Type *type, bool suppressFailure);
	static void NonExistingDimensionInArray(Identifier *id, int dimensionality, int dimension, bool suppressFailure);	
	static void NonLValueInAssignment(Expr *expr, bool suppressFailure);
	static void InvalidArrayAccess(yyltype *loc, Type *actualType, bool suppressFailure);
	static void TooFewOrTooManyParameters(Identifier *name, int actual, int expected, bool suppressFailure);
	static void TaskNameRequiredInEnvironmentCreate(yyltype *loc, bool suppressFailure);
	static void InvalidObjectTypeInNew(Type *type, bool suppressFailure);
	static void TooManyParametersInNew(yyltype *loc, const char *objectType, int actual, int expected, bool suppressFailure);
	static void UndefinedTask(yyltype *loc, const char *name, bool suppressFailure);
	static void NonArrayInIndexedIteration(Identifier *id, Type *type, bool suppressFailure);	
	static void UnknownIndexToArrayAssociation(Identifier *index, Identifier *array, bool suppressFailure);	
	
	// Errors with computation stage to space mappings
	static void SpaceNotFound(yyltype *loc, char spaceName); 
	static void InvalidSpaceNesting(yyltype *loc, const char *nestedSpace, const char *upperSpace);
	static void RepeatLoopAdvanceImposssible(yyltype *loc, const char *spaceName, const char *repeatLoopSpace);
	static void SubpartitionRepeatMeesing(yyltype *loc, const char *spaceName, const char *repeatRoot);
	static void ImpermissibleRepeat(yyltype *loc, const char *spaceName, const char *repeatLoopSpace);
	static void SubpartitionRepeatNotSupported(yyltype *loc, char spaceName);
	static void RepeatBeginningInvalid(yyltype *loc, const char *allowedFurthestRoot);
	static void ArrayPartitionUnknown(yyltype *loc, const char *arrayName, const char *stageName, const char spaceId);
	
	// partition specific error
	static void DuplicateSpaceDefinition(yyltype *loc, char spaceName);
	static void ParentSpaceNotFound(yyltype *loc, char parentSpace, char spaceName, bool isSubpartition);
	static void UnpartitionedDataInPartitionedSpace(yyltype *loc, char spaceName, int dimensionality);
	static void NonTaskGlobalArrayInPartitionSection(Identifier *id);
	static void InvalidSpaceCoordinateSystem(yyltype *loc, char spaceName, int dimensions, bool isSubpartition);
	static void SubpartitionDimensionsNotPositive(yyltype *loc);
	static void SubpartitionOrderConflict(yyltype *loc);
	static void SubpartitionedStructureMissingInParentSpace(yyltype *loc, const char *name);
	static void DimensionMissingOrInvalid(yyltype *loc);	
	static void DimensionMissingOrInvalid(Identifier *id, int dimension);	
	static void ParentForDataStructureNotFound(yyltype *loc, char parentSpace, const char *variable);
	static void ParentDataStructureNotFound(yyltype *loc, char parentSpace, const char *variable);
	static void InvalidPartitionArgument(yyltype *loc);
	static void SpaceAndDataHierarchyConflict(Identifier *id, const char *spaceParent, const char *dataParent);
	static void TooFineGrainedVariablePartition(Identifier *id);
	static void ArgumentCountMismatchInPartitionFunction(const char *functionName);
	static void UnknownPartitionFunction(yyltype *loc, const char *functionName);
	static void ArgumentMissingInPartitionFunction(yyltype *loc, const char *functionName, const char* argument);
	static void InvalidPadding(yyltype *loc, const char *functionName);
	static void PartitionArgumentsNotSupported(yyltype *loc, const char *functionName);
	static void PaddingArgumentsNotSupported(yyltype *loc, const char *functionName);

	// Errors discovered during static analysis	
	static void NotLValueinAssignment(yyltype *loc);

  	// Generic method to report a printf-style error message
  	static void OptionalErrorReport(yyltype *loc, bool suppressFailure, const char *format, ...);
  	static void Formatted(yyltype *loc, const char *format, ...);

  	// Returns number of error messages printed
  	static int NumErrors() { return numErrors; }
  
 private:		
  	static void UnderlineErrorInLine(const char *line, yyltype *pos);
  	static void OutputError(yyltype *loc, string msg);
  	static int numErrors;
  
};

#endif
