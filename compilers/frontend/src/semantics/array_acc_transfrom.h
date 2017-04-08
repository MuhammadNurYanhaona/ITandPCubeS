#ifndef _H_array_acc_transform
#define _H_array_acc_transform

/* This header file incorporate instructions regarding how to transform access to metadata and elements
 * of an array part -- not a full array -- as an argument to a compute stage. Full arrays need no trans-
 * formation. 
 */

#include "../syntax/ast_type.h"
#include "../syntax/ast_expr.h"
#include "../../../common-libs/utils/list.h"

// class instructing how to set up the accessible index range of a dimension when the array part has an
// index-subrange expression for that dimension 
class SubrangeMapping {
  protected:
	int dimNo;
	IndexRange *indexRange;
  public:
	SubrangeMapping(int dimNo, IndexRange *indexRange) {
		this->dimNo = dimNo;
		this->indexRange = indexRange;
	}
	int getDimNo() { return dimNo; }
	IndexRange *getIndexRange() { return indexRange; }	
};

// class holding all instructions for array part access transformations and associated utility functions
class ArrayPartConfig {
  protected:
	// expression representing the array part
	ArrayAccess *arrayPartExpr;
	// the base field representing the whole array
	FieldAccess *baseArrayAccess;
	// type of the base array
	ArrayType *type;
	// listing that tells how each dimension in the array part is mapped to a dimension in the base
	// array 
	List<int> *dimMappings;

	// NOTE THAT: these two fields use NULL expression to indicate the lacking of any instruction
	// for the corresponding array dimension
	// listing of index range access limiter for different dimensions
	List<SubrangeMapping*> *subrangeMappings;
	// listing of index access expressions that pick a specific index entry in some dimension that 
	// cause alteration in the understanding of dimension numbering in the array part
	List<Expr*> *fillerDimAccExprList;
  public:
	ArrayPartConfig(ArrayAccess *arrayPartExpr);
	FieldAccess *getBaseArrayAccess() { return baseArrayAccess; }

	// tells if the dimension indicated by the argument has an associated subrange limiter expression
	bool isLimitedIndexRange(int dimNo);	
	// returns the subrange limiter expression for the argument dimension, when exists
	IndexRange *getAccessibleIndexRange(int dimNo);

	// returns the dimension number in the base array actually accessed by a dimension access on the
	// array part
	int getOrigDimension(int partDimNo);

	// generates an array access relative to the base array from the argument expression relative to 
	// the array part then updates pointers in the argument expression to make it swich access to the
	// generated expression 
	void transformedAccessToArrayPart(ArrayAccess *accessExpr);		
};

#endif
