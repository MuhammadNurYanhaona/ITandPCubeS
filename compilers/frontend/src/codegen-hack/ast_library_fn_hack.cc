#include "../syntax/ast_library_fn.h"
#include "../semantics/task_space.h"
#include <sstream>

void Root::translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space) {}
void Random::translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space) {}

void LoadArray::generateCode(std::ostringstream &stream, int indentLevel, Space *space) {}
void StoreArray::generateCode(std::ostringstream &stream, int indentLevel, Space *space) {}
void BindInput::generateCode(std::ostringstream &stream, int indentLevel, Space *space) {}
void BindOutput::generateCode(std::ostringstream &stream, int indentLevel, Space *space) {}

