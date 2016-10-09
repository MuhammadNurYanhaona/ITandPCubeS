#ifndef _H_common_const
#define _H_common_const

enum ReductionOperator	{       SUM, PRODUCT, MAX, MIN, AVG, MAX_ENTRY, MIN_ENTRY,      // numeric reductions
                                LAND, LOR,                                              // logical reductions    
                                BAND, BOR };                                            // bitwise reductions

enum ArithmaticOperator {       ADD, SUBTRACT, MULTIPLY, DIVIDE, MODULUS, POWER,	// regular arithmatic
                                LEFT_SHIFT, RIGHT_SHIFT,				// shift arithmatic
                                BITWISE_AND, BITWISE_XOR, BITWISE_OR };			// bitwise arithmatic

enum LogicalOperator 	{	AND, OR, NOT, EQ, NE, GT, LT, GTE, LTE };

#endif
