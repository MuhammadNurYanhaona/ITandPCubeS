#include "../../../../../../common-libs/domain-obj/constant.h"
#include "../../../../../../frontend/src/syntax/ast_expr.h"
#include "../../../../../../frontend/src/semantics/task_space.h"

#include <sstream>
#include <iostream>
#include <cstdlib>

void ArithmaticExpr::translate(std::ostringstream &stream, int indentLevel, int currentLineLength, Space *space) {
        if (op != POWER) {
                stream << "(";
                left->translate(stream, indentLevel, currentLineLength, space);
                switch (op) {
                        case ADD: stream << " + "; break;
                        case SUBTRACT: stream << " - "; break;
                        case MULTIPLY: stream << " * "; break;
                        case DIVIDE: stream << " / "; break;
                        case MODULUS: stream << ' ' << '%' << ' '; break;
                        case LEFT_SHIFT: stream <<" << "; break;
                        case RIGHT_SHIFT: stream << " >> "; break;
                        case BITWISE_AND: stream << " & "; break;
                        case BITWISE_XOR: stream << " ^ "; break;
                        case BITWISE_OR: stream << " | "; break;
                        default: break;
                }
                right->translate(stream, indentLevel, currentLineLength, space);
                stream << ")";
        } else {
                stream << "(";
                stream << "(" << type->getCType() << ")";
                stream << "pow(";
                left->translate(stream, indentLevel, currentLineLength, space);
                stream << ", ";
                right->translate(stream, indentLevel, currentLineLength, space);
                stream << ")";
                stream << ")";
        }
}
