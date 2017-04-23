#include "../../../utils/code_constant.h"
#include "../../../../../../frontend/src/syntax/ast_expr.h"
#include "../../../../../../frontend/src/syntax/ast_type.h"
#include "../../../../../../frontend/src/semantics/task_space.h"

#include <sstream>

void NamedArgument::generateAssignment(Expr *object, std::ostringstream &stream, int indentLevel) {
        for (int i = 0; i < indentLevel; i++) stream << indent;
        object->translate(stream, indentLevel);
        NamedType *userDefinedObject = dynamic_cast<NamedType*>(object->getType());
        if (userDefinedObject != NULL && userDefinedObject->isEnvironmentType()) {
                stream << "->";
        } else stream << '.';
        stream << argName << " = ";
        argValue->translate(stream, indentLevel);
        stream << stmtSeparator;
}

void ObjectCreate::translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space) {

        ArrayType *array = dynamic_cast<ArrayType*>(type);
        if (array == NULL) {
                NamedType *userDefinedType = dynamic_cast<NamedType*>(type);
                if (userDefinedType != NULL) {
                        if (userDefinedType->isEnvironmentType()) stream << "new ";
                }
                stream << type->getCType() << "()";
        }
}

void ObjectCreate::generateCodeForProperties(Expr *object, std::ostringstream &stream, int indentLevel) {

        // generate property transfer statements for individual arguments passed to the controller
        for (int i = 0; i < initArgs->NumElements(); i++) {
                NamedArgument *currentArg = initArgs->Nth(i);
                currentArg->generateAssignment(object, stream, indentLevel);
        }

        // for task-environment objects, we need to assign a back pointer to the sole program environment instance created
        // in the coordinator function for environmental data structures management; at the same time, we assign the log
        // file created for the segment to the environment to track all environment management activities
        NamedType *userType = dynamic_cast<NamedType*>(object->getType());
        if (userType != NULL && userType->isEnvironmentType()) {
                for (int i = 0; i < indentLevel; i++) stream << indent;
                object->translate(stream, indentLevel);
                stream << "->setProgramEnvironment(programEnv)" << stmtSeparator;
                for (int i = 0; i < indentLevel; i++) stream << indent;
                object->translate(stream, indentLevel);
                stream << "->setLogFile(&logFile)" << stmtSeparator;
        }
}

bool ObjectCreate::isDynamicArrayCreate(Expr *candidateExpr) {
        ObjectCreate *objectCreate = dynamic_cast<ObjectCreate*>(candidateExpr);
        if (objectCreate == NULL) return false;
        Type *objectType = candidateExpr->getType();
        ArrayType *array = dynamic_cast<ArrayType*>(objectType);
        StaticArrayType *staticArray = dynamic_cast<StaticArrayType*>(objectType);
        return (array != NULL && staticArray == NULL);
} 
