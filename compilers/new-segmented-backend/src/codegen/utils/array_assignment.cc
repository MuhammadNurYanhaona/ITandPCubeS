#include "array_assignment.h"
#include "name_transformer.h"
#include "code_constant.h"

#include "../../../../frontend/src/syntax/ast.h"
#include "../../../../frontend/src/syntax/ast_expr.h"
#include "../../../../frontend/src/syntax/ast_type.h"
#include "../../../../frontend/src/semantics/scope.h"
#include "../../../../frontend/src/semantics/task_space.h"

#include "../../../../common-libs/utils/list.h"

#include <iostream>
#include <sstream>
#include <cstdlib>
#include <iostream>

void codecntx::enterTaskContext() { codecntx::coordinator = false; }

void codecntx::enterCoordinatorContext() { codecntx::coordinator = true; }

bool isArrayAssignment(AssignmentExpr *expr) {
	Type *type = expr->getType();
	if (type == NULL) return false;
	Expr *right = expr->getRight();
	FieldAccess *field = dynamic_cast<FieldAccess*>(right);
	ArrayAccess *array = dynamic_cast<ArrayAccess*>(right);
	AssignmentExpr *nextAssignment = dynamic_cast<AssignmentExpr*>(right);
	if (field == NULL && array == NULL && nextAssignment == NULL) return false;
	ArrayType *arrayType = dynamic_cast<ArrayType*>(type);
	StaticArrayType *staticType = dynamic_cast<StaticArrayType*>(type);
	return (arrayType != NULL && staticType == NULL);
}

ArrayName *getArrayName(Expr *expr) {
	FieldAccess *fieldAccess = NULL;
	Expr *currentExpr = expr;
	while ((fieldAccess = dynamic_cast<FieldAccess*>(currentExpr)) == NULL) {
		ArrayAccess *array = dynamic_cast<ArrayAccess*>(currentExpr);
		currentExpr = array->getBase();	
	}
	ArrayName *arrayName = new ArrayName();

	// The only ways we can have an array is either as an stand-alone reference or as an element of
	// an environment object. This is because user defined types cannot have arrays as element. Thus
	// we can easily determine if an array is part of an environment or not by simply checking if 
	// corresponding field-access reference is a terminal field access. 
	bool partOfEnv = !fieldAccess->isTerminalField();

	arrayName->setPartOfEnv(partOfEnv);
	arrayName->setName(fieldAccess->getField()->getName());
	if (partOfEnv) {
		FieldAccess *base = dynamic_cast<FieldAccess*>(fieldAccess->getBase());
		arrayName->setEnvObjName(base->getField()->getName());
	}
	arrayName->setType((ArrayType*) fieldAccess->getType());

	return arrayName;
}

AssignmentMode determineAssignmentMode(AssignmentExpr *expr) {
	
	Expr *left = expr->getLeft();
	ArrayName *leftArray = getArrayName(left);
	Type *leftType = leftArray->getType();
	Expr *right = expr->getRight();
	ArrayName *rightArray = getArrayName(right);
	Type *rightType = rightArray->getType();
	Type *exprType = expr->getType();

	// Right at this moment (March 28, 2015) we don't support reference assignment from one array
	// to another within the compute stages of a task. We need to decide how to handle it as the
	// mechanism will affect how we do dependency analysis and consequently synchronization and 
	// communication. For now as a quick solution, we avoid reference passing altogether within the
	// confinement of a task
	if (!codecntx::coordinator) return COPY;

	// if the left side of the assignment is not referencing an stand-alone data structure then 
	// there is no other option than copying content from right to left.
	FieldAccess *leftField = dynamic_cast<FieldAccess*>(left);
	if (leftField == NULL) return COPY; 

	// We choose to do a reference assignment from left to right side for an assignment expression 
	// if the arrays on both sides have the same dimensionality and they are accessed in full as 
	// part of the assignment expression or the left side is part of a task environment.  
	if ((leftType->isEqual(rightType) && leftType->isEqual(exprType)) 
			|| leftArray->isPartOfEnv()) return REFERENCE;
	// Otherwise, we copy data from right to the left side array.
	return COPY;
}

List<DimensionAccess*> *generateDimensionAccessInfo(ArrayName *array, Expr *expr) {
	
	int arrayDimensions = array->getType()->getDimensions();
	List<DimensionAccess*> *accessList = new List<DimensionAccess*>;

	// First we add default entries in the access list assuming that each dimension of the array 
	// has been explicitly accessed in the expression. This is needed as just an standalone reference
	// such as in 'a = b[...][i]' should be treated as accessing the whole first dimension of a. 
	for (int i = 0; i < arrayDimensions; i++) {
		accessList->Append(new DimensionAccess(i));
	}

	// then we track down the dimensions actually been accessed in the expression
	List<Expr*> *accessExprList = new List<Expr*>;
	FieldAccess *fieldAccess = NULL;
	Expr *currentExpr = expr;
	while ((fieldAccess = dynamic_cast<FieldAccess*>(currentExpr)) == NULL) {
		ArrayAccess *arrayAccess = dynamic_cast<ArrayAccess*>(currentExpr);
		currentExpr = arrayAccess->getBase();
		// since the traversal moves backward, elements should be added at the front of the list
		accessExprList->InsertAt(arrayAccess->getIndex(), 0);	
	}

	// then we replace entries in the original list based on the expressions retrieved
	for (int i = 0; i < accessExprList->NumElements(); i++) {
		DimensionAccess *actualAccessInfo = new DimensionAccess(accessExprList->Nth(i), i);
		accessList->RemoveAt(i);
		accessList->InsertAt(actualAccessInfo, i);
	} 

	return accessList;
}

//---------------------------------------------------------- Array Name ---------------------------------------------------------/

ArrayName::ArrayName() {
	this->partOfEnv = false;
	this->envObjName = NULL;
	this->name = NULL;
	this->type = NULL;
}

void ArrayName::describe(int indent) {
	for (int i = 0; i < indent; i++) std::cout << '\t';
	if (partOfEnv) std::cout << envObjName << ".";
	std::cout << name;
	std::cout << ": " << type->getName() << std::endl;
}

const char *ArrayName::getTranslatedName() {
	if (partOfEnv) {
		// name translation is not applicable for environmental variables
		return NULL;
	} else {
		std::ostringstream nameStr;
		ntransform::NameTransformer *transformer = ntransform::NameTransformer::transformer;
		return transformer->getTransformedName(name, false, true, type);
	}
}

const char *ArrayName::getTranslatedMetadataPrefix() {
	if (partOfEnv) {
		// name translation is not applicable for environmental variables
		return NULL;
	} else {
		std::ostringstream prefixStr;
		ntransform::NameTransformer *transformer = ntransform::NameTransformer::transformer;
		return transformer->getTransformedName(name, true, true, type);
	}
}

//-------------------------------------------------------- Dimension Access -----------------------------------------------------/

DimensionAccess::DimensionAccess(int dimensionNo) {
	this->accessType = WHOLE;
	this->accessExpr = NULL;
	this->dimensionNo = dimensionNo;
}

DimensionAccess::DimensionAccess(Expr *accessExpr, int dimensionNo) {
	this->accessExpr = accessExpr;
	IndexRange *subRange = dynamic_cast<IndexRange*>(accessExpr);
	if (subRange == NULL) accessType = INDEX;
	else if (!subRange->isFullRange()) accessType = SUBRANGE;
	else accessType = WHOLE;
	this->dimensionNo = dimensionNo;
}

void DimensionAccess::describe(int indent) {
	for (int i = 0; i < indent; i++) std::cout << '\t';
	if (accessType == INDEX) std::cout << "index-access";
	else if (accessType == SUBRANGE) std::cout << "subrange-access";
	else std::cout << "whole-dimension-access";
	std::cout << ": Dimension #" << dimensionNo << std::endl;	
}

//------------------------------------------------------- Dimension Annotation --------------------------------------------------/

void DimensionAnnotation::describe(int indent) {
	for (int i = 0; i < indent; i++) std::cout << '\t';
	std::cout << "Left trail:" << std::endl;
	for (int i = 0; i < assigneeInfo->NumElements(); i++) {
		assigneeInfo->Nth(i)->describe(indent + 1);
	}	
	for (int i = 0; i < indent; i++) std::cout << '\t';
	std::cout << "Right trail:" << std::endl;
	for (int i = 0; i < assignerInfo->NumElements(); i++) {
		assignerInfo->Nth(i)->describe(indent + 1);
	}
}

//------------------------------------------------------- Assignment Directive --------------------------------------------------/

AssignmentDirective::AssignmentDirective(AssignmentExpr *expr) {
	this->expr = expr;
	this->mode = determineAssignmentMode(expr);
	this->annotations = new List<DimensionAnnotation*>;
	Expr *left = expr->getLeft();
	this->assigneeArray = getArrayName(left);
	Expr *right = expr->getRight();
	this->assignerArray = getArrayName(right);
}

void AssignmentDirective::generateAnnotations() {

	// get information about how different dimensions are accessed within the arrays present on two sides
	List<DimensionAccess*> *leftSideAccesses 
			= generateDimensionAccessInfo(assigneeArray, expr->getLeft());
	List<DimensionAccess*> *rightSideAccesses 
			= generateDimensionAccessInfo(assignerArray, expr->getRight());

	// then traverse over a list and generate annotation for each dimension transfer from right to left
	int i = 0;
	int j = 0;
	List<DimensionAccess*> *leftAccessList = new List<DimensionAccess*>;
	for (; i < leftSideAccesses->NumElements(); i++) {
		
		DimensionAccess *currentAccess = leftSideAccesses->Nth(i);
		leftAccessList->Append(currentAccess);
		if (currentAccess->isSingleEntry()) continue;

		// once a whole/partial dimension access is found for the left side, determine how to get to
		// the corresponding dimension on the right side
		List<DimensionAccess*> *rightAccessList = new List<DimensionAccess*>;
		for (; j < rightSideAccesses->NumElements(); ) {
			DimensionAccess *rightAccess = rightSideAccesses->Nth(j);
			rightAccessList->Append(rightAccess);
			j++;
			if (!rightAccess->isSingleEntry()) break;
		}
		
		// then create a dimension annotation and store that for current mapping
		DimensionAnnotation *annotation = new DimensionAnnotation(false);
		annotation->setAssigneeInfo(leftAccessList);
		annotation->setAssignerInfo(rightAccessList);

		// then refresh the left side list for next annotation and add the current annotation in the list
		leftAccessList = new List<DimensionAccess*>;
		annotations->Append(annotation);
	}

	// Finally, if there are entries left in the left and/or right expression list then generate a place 
	// holder annotation for it and store it in the list too.
	for (; i < leftSideAccesses->NumElements(); i++) leftAccessList->Append(leftSideAccesses->Nth(i));
	List<DimensionAccess*> *rightAccessList = new List<DimensionAccess*>;
	for (; j < rightSideAccesses->NumElements(); j++) rightAccessList->Append(rightSideAccesses->Nth(j));
	if (leftAccessList->NumElements() > 0 || rightAccessList->NumElements() > 0) {
		DimensionAnnotation *annotation = new DimensionAnnotation(true);
		annotation->setAssigneeInfo(leftAccessList);
		annotation->setAssignerInfo(rightAccessList);
		annotations->Append(annotation);
	}
}

void AssignmentDirective::describe(int indent) {
	std::ostringstream indStr;
	for (int i = 0; i < indent; i++) indStr << '\t';
	std::cout << indStr.str();
	if (mode == COPY) std::cout << "Copy transfer:\n";
	else std::cout << "Reference assignment:\n";
	assigneeArray->describe(indent + 1);
	assignerArray->describe(indent + 1);
	std::cout << indStr.str() << "\tTransfer detail:\n";
	for (int i = 0; i < annotations->NumElements(); i++) {
		annotations->Nth(i)->describe(indent + 2);
	} 
}

void AssignmentDirective::generateCode(std::ostringstream &stream, int indentLevel, Space *space) {
	if (mode == COPY) generateCodeForCopy(stream, indentLevel, space);
	else generateCodeForReference(stream, indentLevel, space);
}

void AssignmentDirective::generateCodeForCopy(std::ostringstream &stream, int indentLevel, Space *space) {
	std::cout << "Code for translating array assignments as copying data is still not implemented\n";
	std::cout << "Kindly, rewrite the concerned expression as a for loop having direct data copy.\n";
	std::exit(EXIT_FAILURE);	
}

void AssignmentDirective::generateCodeForReference(std::ostringstream &stream, int indentLevel, Space *space) {
	
	std::ostringstream indent;
	for (int i = 0; i < indentLevel; i++) indent << '\t';
	std::string indents = indent.str();

	bool leftLocal = !assigneeArray->isPartOfEnv();	
	const char *leftName = assigneeArray->getTranslatedName();
	const char *leftMetaPrefix = assigneeArray->getTranslatedMetadataPrefix();
	bool rightLocal = !assignerArray->isPartOfEnv();
	const char *rightName = assignerArray->getTranslatedName();
	const char *rightMetaPrefix = assignerArray->getTranslatedMetadataPrefix();

	// create a scope for the array transfer operation
	stream << '\n' << indents << "{ // scope starts for transferring contents between arrays\n";

	// if the left-side array is not a local variable then this is an environment object transfar and we need
	// retrieve the environmental item that will get updated and initialize a transfer instruction for it
	if (!leftLocal) {
		stream << indents << "TaskItem *destItem = " << assigneeArray->getEnvObjName();
		stream << "->getItem(\"" << assigneeArray->getName() << "\")" << stmtSeparator;
		stream << indents << "DataTransferInstruction *instr =";
		stream << " new DataTransferInstruction(destItem)" << stmtSeparator;
		stream << indents << "ArrayTransferConfig *destConfig = new ArrayTransferConfig()";
		stream << stmtSeparator;
		stream << indents << "instr->setTransferConfig(destConfig)" << stmtSeparator;
	} else {
		stream << indents << "ArrayTransferConfig *destConfig = &" << leftName << "TransferConfig";
		stream << stmtSeparator;
	}

	// If the right-side array is not a local variable then we retreive the task-item to be able to access
	// dimension information from it. Otherwise just retrieve the local source transfer configuration object
	// as a reference. At the same time set the data source reference from the source to the destination 
	if (!rightLocal) {
		const char *sourceEnvObjName = assignerArray->getEnvObjName();
		const char *sourceProperty = assignerArray->getName();
		stream << indents << "TaskItem *sourceItem = " << sourceEnvObjName;
		stream << "->getItem(\"" << sourceProperty << "\")" << stmtSeparator;
		// if there is a transfer configuration associated with the right-side array then retrieve that
		stream << indents << "TaskInitEnvInstruction *sourceInstr = " << assignerArray->getEnvObjName();
		stream << "->getInstr(\"" << sourceProperty << "\"" << paramSeparator;
		// array transfer is of type three 
		stream << '3' << ")" << stmtSeparator;
		stream << indents << "ArrayTransferConfig *sourceConfig " << '\n'  << indents << doubleIndent;
		stream << "= (sourceInstr != NULL) " << '\n' << indents << doubleIndent;
		stream << "? ((DataTransferInstruction*) sourceInstr)->getTransferConfig()";
		stream << '\n' << indents << doubleIndent << ": NULL" << stmtSeparator;
		stream << indents << "destConfig->setSource(" << sourceEnvObjName << paramSeparator;
		stream << '"' << sourceProperty << "\")" << stmtSeparator;
	} else {
		stream << indents << "ArrayTransferConfig *sourceConfig = &" << rightName;
		stream << "TransferConfig" << stmtSeparator;
		stream << indents << "destConfig->setSource(" << rightName << paramSeparator;
		stream << "NULL)" << stmtSeparator;
	}

	// if neither side is an environmental array, we can assign the reference from the right to the left
	if (leftLocal && rightLocal) {
		stream << indents << leftName << " = " << rightName << stmtSeparator;
	}
	
	// copy metadata from right to left for each dimension of the array
	for (int i = 0; i < annotations->NumElements(); i++) {
		DimensionAnnotation *annotation = annotations->Nth(i);

		// Note that the current logic for determining the array assignment mode (copy/reference) says
		// that the left side is a field access. Thus, there cannot be multiple dimension accesses per
		// annotation at the left side. TODO we should remove this restriction in the future and support
		// more data transfers to happen as reference passing
		DimensionAccess *leftAccess = annotation->getAssigneeInfo()->Nth(0);

		// get the list of dimension information from the right side and populate dimension transfers
		// on the array transfer configuration of the left
		List<DimensionAccess*> *rightAccessList = annotation->getAssignerInfo();
		for (int j = 0; j < rightAccessList->NumElements(); j++) {
			DimensionAccess *rightAccess = rightAccessList->Nth(j);
			int dimNo = rightAccess->getDimensionNo();
			
			stream << indents << "Dimension storeDim" << dimNo << " = ";
			if (rightLocal) {
				stream << rightMetaPrefix << "[" << dimNo << "].storage" << stmtSeparator;
			} else {
				stream << "sourceItem->getDimension(" << dimNo << ")" << stmtSeparator;
			}
			
			stream << indents << "destConfig->recordTransferDimConfig(";
			if (rightAccess->getAccessType() == SUBRANGE) {
				stream << dimNo << paramSeparator << "Range(";
				IndexRange *accessExpr = (IndexRange*) rightAccess->getAccessExpr();
				accessExpr->getBegin()->translate(stream, indentLevel);
				stream << paramSeparator;
				accessExpr->getEnd()->translate(stream, indentLevel);
				stream << ")";
			} else if (rightAccess->getAccessType() == INDEX) {
				stream << dimNo << paramSeparator;
				rightAccess->getAccessExpr()->translate(stream, indentLevel);
			} else {
				stream << dimNo << paramSeparator << "storeDim" << dimNo << ".range";
			}
			stream << ")" << stmtSeparator;
			
			// if the left side is a local array then we should update its storage dimension as
			// that information is available here; note that we can be confident that the left and 
			// right dimension numbers are the same as this is a reference assignment
			if (leftLocal) {
				stream << indents << leftMetaPrefix;
				stream << "[" << dimNo << "].storage = storeDim" << dimNo << stmtSeparator;
			}  
		}
	}
		
	// assign the transfer configuration of the right side as the parrent to the configuration for
	// the left side
	stream << indents << "destConfig->setParent(sourceConfig)" << stmtSeparator; 

	// finally, if the left-side array is a local variable then update the partition dimensions
	// using the destination transfer configuration object
	if (leftLocal) {
		ArrayType *leftType = assigneeArray->getType();
		int dimensionality = leftType->getDimensions();
		stream << indents << "destConfig->copyDimensions(" << leftMetaPrefix;
		stream << paramSeparator << dimensionality << ")" << stmtSeparator;
		
	// otherwise, record the transfer instruction in the environment object holding the left array
	} else {
		stream << indents << assigneeArray->getEnvObjName();
		stream << "->addInitEnvInstruction(instr)" << stmtSeparator;
	}

	// close scope
	stream  << indents << "} // scope ends for transferring contents between arrays\n\n";
}

//---------------------------------------------------- Assignment Directive List ------------------------------------------------/

AssignmentDirectiveList::AssignmentDirectiveList(AssignmentExpr *expr) {
	
	this->mainExpr = expr;
	
	AssignmentExpr *rightAssignment = NULL;
	Expr *currentRight = expr;
	List<Expr*> *leftList = new List<Expr*>;	
	while ((rightAssignment = dynamic_cast<AssignmentExpr*>(currentRight)) != NULL) {
		leftList->Append(rightAssignment->getLeft());
		currentRight = rightAssignment->getRight();
	}
	
	List<AssignmentExpr*> *assignmentList = new List<AssignmentExpr*>;
	for (int i = 0; i < leftList->NumElements(); i++) {
		Expr *leftExpr = leftList->Nth(i);
		AssignmentExpr *assignment = new AssignmentExpr(leftExpr, currentRight, *leftExpr->GetLocation());
		assignment->setType(expr->getType());
		assignmentList->Append(assignment);
	}

	this->directives = new List<AssignmentDirective*>;
	for (int i = 0; i < assignmentList->NumElements(); i++) {
		AssignmentExpr *assignment = assignmentList->Nth(i);
		AssignmentDirective *directive = new AssignmentDirective(assignment);
		directive->generateAnnotations();
		this->directives->Append(directive);
	}
}

void AssignmentDirectiveList::describe(int indent) {
	for (int i = 0; i < directives->NumElements(); i++) {
		directives->Nth(i)->describe(indent);	
	}
}

void AssignmentDirectiveList::generateCode(std::ostringstream &stream, int indentLevel, Space *space) {
	for (int i = 0; i < directives->NumElements(); i++) {
		directives->Nth(i)->generateCode(stream, indentLevel, space);	
	}
}
