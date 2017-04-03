#include "../semantics/computation_flow.h"
#include "../semantics/task_space.h"
#include <fstream>

void StageInstanciation::translateCode(std::ofstream &stream) {}
void StageInstanciation::generateInvocationCode(std::ofstream &stream, int indentation, Space *containerSpace) {}
void CompositeStage::generateInvocationCode(std::ofstream &stream, int indentation, Space *containerSpace) {}
void RepeatControlBlock::generateInvocationCode(std::ofstream &stream, int indentation, Space *containerSpace) {}
void ConditionalExecutionBlock::generateInvocationCode(std::ofstream &stream, int indentation, Space *containerSpace) {}
void LpsTransitionBlock::generateInvocationCode(std::ofstream &stream, int indentation, Space *containerSpace) {}
void EpochBoundaryBlock::generateInvocationCode(std::ofstream &stream, int indentation, Space *containerSpace) {}
void ReductionBoundaryBlock::generateInvocationCode(std::ofstream &stream, int indentation, Space *containerSpace) {}
