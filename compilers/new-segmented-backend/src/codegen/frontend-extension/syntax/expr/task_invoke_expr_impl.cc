#include "../../../../../../frontend/src/syntax/ast_task.h"
#include "../../../../../../frontend/src/syntax/ast_def.h"
#include "../../../../../../frontend/src/syntax/ast_expr.h"
#include "../../../../../../frontend/src/semantics/task_space.h"
#include "../../../utils/code_constant.h"
#include "../../../utils/task_generator.h"

#include <sstream>
#include <iostream>
#include <cstdlib>

void TaskInvocation::generateCode(std::ostringstream &stream, int indentLevel, Space *space) {
	
	TupleDef *partitionTuple = taskDef->getPartitionTuple();
        std::ostringstream indent;
        for (int i = 0; i < indentLevel; i++) indent << '\t';

	const char *taskName = getTaskName();
        stream << indent.str() << "{ // scope starts for invoking: " << taskName << "\n";
	stream << indent.str() << "logFile << \"going to execute task: " << taskName;
	stream << "\\n\"" << stmtSeparator;
	stream << indent.str() << "logFile.flush()" << stmtSeparator;

	std::ostringstream envStream;
	FieldAccess *envArg = getEnvArgument();
	envArg->translate(envStream, 0, 0, space);
	const char *envName = strdup(envStream.str().c_str());

	// first invoke the task environment's linked arrays' dimensions initialization function as the dimension
	// lengths will be needed to construct task's partition hierarchy
	stream << indent.str() << envName << "->setupItemsDimensions()" << stmtSeparator;        

	// assign a task invocation Id to the environment for the upcoming task and increase the inocation id for
	// later usage
	stream << indent.str() << envName << "->setTaskId(taskId)" << stmtSeparator;
	stream << indent.str() << "taskId++" << stmtSeparator;
	
	// create a partition object for the task
	stream << indent.str() << partitionTuple->getId()->getName() << " partition" << stmtSeparator;

	// collect initialization arguments into a stream, if exist
	std::ostringstream initParams;
	bool initParamsPresent = false;
	List<Expr*> *initArgs = getInitArguments();
	if (initArgs != NULL && initArgs->NumElements() > 0) {
		for (int i = 0; i < initArgs->NumElements(); i++) {
			initParams << paramSeparator;
			Expr *argument = initArgs->Nth(i);
			argument->translate(initParams, 0);
		}
	}

	// populate properties of the partition tuple
	List<Expr*> *partitionArgs = getPartitionArguments();
	if (partitionArgs != NULL && partitionArgs->NumElements() > 0) {

		List<VariableDef*> *tupleParts = partitionTuple->getComponents();
		for (int i = 0; i < partitionArgs->NumElements(); i++) {
			stream << indent.str();
			const char *propertyName = tupleParts->Nth(i)->getId()->getName();
			stream << "partition." << propertyName;
			stream << " = ";
			partitionArgs->Nth(i)->translate(stream, 0);
			stream << stmtSeparator;
		}
	}

	// then invoke the task with appropriate parameters
	stream << indent.str();
	stream << TaskGenerator::getNamespace(taskDef) << "::execute(";
	stream << envName;
	if (initParamsPresent) stream << initParams.str();
	stream << paramSeparator << "partition";
	stream << paramSeparator << "segmentId";
	stream << paramSeparator << "logFile)" << stmtSeparator;	

	// finally reset all environment update instructions
	stream << indent.str() << envName << "->resetEnvInstructions()" << stmtSeparator;

	stream << indent.str() << "} // scope ends for task invocation\n";
}
