#include "array_acc_transfrom.h"
#include "../syntax/ast_type.h"
#include "../syntax/ast_expr.h"
#include "../../../common-libs/utils/list.h"

ArrayPartConfig::ArrayPartConfig(ArrayAccess *arrayPartExpr) {

	this->arrayPartExpr = arrayPartExpr;
	ArrayType *partType = (ArrayType*) arrayPartExpr->getType();
	
	FieldAccess *fieldAccess = NULL;
        Expr *currentExpr = arrayPartExpr;
	List<Expr*> *indexExprList = new List<Expr*>;
        while ((fieldAccess = dynamic_cast<FieldAccess*>(currentExpr)) == NULL) {
                ArrayAccess *array = dynamic_cast<ArrayAccess*>(currentExpr);
		Expr *indexExpr = array->getIndex();
		
		// insert index expression at the beginning of the list as we are moving from tail to the head 
		indexExprList->InsertAt(indexExpr, 0);
                
		currentExpr = array->getBase();
        }
        baseArrayAccess = fieldAccess;
	this->type = (ArrayType*) baseArrayAccess->getType();
	
	dimMappings = new List<int>;
	subrangeMappings = new List<SubrangeMapping*>;
	fillerDimAccExprList = new List<Expr*>;

	int totalDims = type->getDimensions();
	int partDims = partType->getDimensions();
	int dimNo = 0;
	int baseDimNo = 0;
	for (; dimNo < partDims; dimNo++, baseDimNo++) {
		Expr *dimIndexExpr = indexExprList->Nth(dimNo);
		IndexRange *indexRange = dynamic_cast<IndexRange*>(dimIndexExpr);

		// if the index expression is a sub-range then determine the expanse of the subrange; this dim-
		// ension does not cause any dimension number reordering
		if (indexRange != NULL) {
			// put a NULL entry in the filler dimension access list for the current dimension as
			fillerDimAccExprList->Append(NULL);

			// tracking of index-subrange expression is only needed when the index expression does
			// not represent the entirity of the underlying dimension range
			if (!indexRange->isFullRange()) {
				subrangeMappings->Append(new SubrangeMapping(dimNo, indexRange));
			} else {
				subrangeMappings->Append(NULL);
			}
		
			// put an entry to the part to base dimension mapping
			dimMappings->Append(baseDimNo);	

		// if, on the other hand, the index expression is a specific index entry then this dimension
		// causes dimension number reordering 
		} else {
			// we should keep track of the index expression to be used to update any array access
			// expression done on the array part within the underlying compute stage
			fillerDimAccExprList->Append(dimIndexExpr);

			// put a NULL entry in the subrange mapping list for the current dimension
			subrangeMappings->Append(NULL);

			// we must advance the baseDimNo an extra step to indicate that access expressions to
			// the part does not include this dimension
			baseDimNo++;
		}
	}
	
	// fill list entries for the remaining dimensions with placeholder entries 
	for (; dimNo < totalDims; dimNo++, baseDimNo++) {
		subrangeMappings->Append(NULL);
		fillerDimAccExprList->Append(NULL);
		dimMappings->Append(baseDimNo);
	}
}

bool ArrayPartConfig::isLimitedIndexRange(int dimNo) {
	return (subrangeMappings->Nth(dimNo) != NULL);
}

IndexRange *ArrayPartConfig::getAccessibleIndexRange(int dimNo) {
	return subrangeMappings->Nth(dimNo)->getIndexRange();
}

int ArrayPartConfig::getOrigDimension(int partDimNo) {
	return dimMappings->Nth(partDimNo);
}

void ArrayPartConfig::transformedAccessToArrayPart(ArrayAccess *accessExpr) {
	
	// extract the base array field access and the index accesses from the argument expression
	FieldAccess *baseArray = NULL;
        Expr *currentExpr = accessExpr;
	List<Expr*> *indexExprList = new List<Expr*>;
        while ((baseArray = dynamic_cast<FieldAccess*>(currentExpr)) == NULL) {
                ArrayAccess *array = dynamic_cast<ArrayAccess*>(currentExpr);
		Expr *indexExpr = array->getIndex();
		
		// insert index expression at the beginning of the list as we are moving from tail to the head 
		indexExprList->InsertAt(indexExpr, 0);
                
		currentExpr = array->getBase();
        }

	// update the name of the base field access to reflect access to the original array
	const char *arrayName = baseArrayAccess->getField()->getName();
	baseArray->getField()->changeName(arrayName);

	// create a list of index access expressions relative to the original array
	List<Expr*> *newIndexExprList = new List<Expr*>;
	int totalDims = type->getDimensions();
	List<bool> *clonedDimConfigList = new List<bool>;
	for (int dimNo = 0; dimNo < totalDims; dimNo++) {
		
		// retrieve the filler expression for current dimension
		Expr *fillerExpr = fillerDimAccExprList->Nth(dimNo);
		
		// if there is no filler expression then the current index access expression from the argument
		// array access should fit into the place
		if (fillerExpr == NULL) {
			Expr *indexAccess = indexExprList->Nth(0);
			indexExprList->RemoveAt(0);	// remove the head to advance to the next expression
			newIndexExprList->Append(indexAccess);
			clonedDimConfigList->Append(false);

		// otherwise, copy the stored filler expression and put it in the index access expression list
		} else {
			// cloning is needed as there are parent pointers in any expression that we must not
			// corrupt by placing the same filler index access expression in multiple array access
			// expressions of the compute stage
			Expr *clonedIndexAccess = (Expr*) fillerExpr->clone();	
			newIndexExprList->Append(clonedIndexAccess);
			clonedDimConfigList->Append(true);
		}
	}

	// stick the expressions together to form the output expression
	yyltype location = *accessExpr->GetLocation();
	currentExpr = baseArray;
	for (int dimNo = 0; dimNo < totalDims; dimNo++) {
		Expr *indexExpr = newIndexExprList->Nth(dimNo);
		currentExpr = new ArrayAccess(currentExpr, indexExpr, location);
		if (clonedDimConfigList->Nth(dimNo)) {
			((ArrayAccess*) currentExpr)->flagAsFillerIndexAccess();
		}
	}
	ArrayAccess *finalArrayAcc = (ArrayAccess*) currentExpr;

	// now swich the pointers in the argument array access expression to refer to the generated expression
	accessExpr->setBase(finalArrayAcc->getBase());
	accessExpr->setIndex(finalArrayAcc->getIndex());
	if (clonedDimConfigList->Nth(totalDims - 1)) {
		accessExpr->flagAsFillerIndexAccess();
	}	 
}
