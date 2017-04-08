#include "../../../../../../frontend/src/semantics/task_space.h"
#include "../../../../../../frontend/src/semantics/computation_flow.h"

#include <fstream>

void LpsTransitionBlock::generateInvocationCode(std::ofstream &stream, int indentation, Space *containerSpace) {
	CompositeStage::generateInvocationCode(stream, indentation, containerSpace);
}
