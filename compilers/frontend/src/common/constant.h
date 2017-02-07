#ifndef _H_common_const
#define _H_common_const

enum ReductionOperator	{       SUM, PRODUCT, MAX, MIN, AVG, MAX_ENTRY, MIN_ENTRY,      // numeric reductions
                                LAND, LOR,                                              // logical reductions    
                                BAND, BOR };                                            // bitwise reductions

enum ArithmaticOperator {       ADD, SUBTRACT, MULTIPLY, DIVIDE, MODULUS, POWER,	// regular arithmatic
                                LEFT_SHIFT, RIGHT_SHIFT,				// shift arithmatic
                                BITWISE_AND, BITWISE_XOR, BITWISE_OR };			// bitwise arithmatic

enum LogicalOperator 	{	AND, OR, NOT, EQ, NE, GT, LT, GTE, LTE };

enum IntSize		{	BYTE, TWO_BYTES, FOUR_BYTE };

enum ArgumentType	{	VALUE_TYPE, REFERENCE_TYPE };

/*      Partition Order specifies in what order individual partitions (or subpartitions) of a space will be
        generated and processed by the runtime. This parameter is mostly relevant for subpartitioned spaces
        as in normal cases partitions of a space are independent. 
*/
enum PartitionOrder { AscendingOrder, DescendingOrder, RandomOrder };

/*      Partition Link Type specifies how partitions of a single data structure in a lower space is linked to 
        its parent partition in higher space. Some-times this linking may be specified in the Space level that
        dictates the relationships of all its data structure partitions to partitions of some parent space.
        Other-times, relationships are defined data structure by data structure basis. If like type is
        
        LinkTypePartition: then lower space partition further divides upper space partitions
        LinkTypeSubpartition: then lower space partition divides each upper space subpartitions
        LinkTypeUndefined: then there is no linkage type currently defined. Some may be derived from other 
                           information
*/
enum PartitionLinkType { LinkTypePartition, LinkTypeSubpartition, LinkTypeUndefined };

#endif
