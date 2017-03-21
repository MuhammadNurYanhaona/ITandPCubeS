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

/*	Environment variable linkage types indicate how an environmental data structure is made available to
	or by a task
	TypeCreate: means the task execution creates a new instance
	TypeLink: means the structure is linked to the task environment at the beginning
	TypeCreateIfNotLinked: is self-explanatory after the above two
*/
enum LinkageType { TypeCreate, TypeLink, TypeCreateIfNotLinked };

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

/*	Identifier for different kinds of expression. Each new expression type should have an unique ID to be
	used for quick retrieval of expressions by types.
*/
enum ExprTypeId { 	INT_CONST, FLOAT_CONST, DOUBLE_CONST, 
			BOOL_CONST, CHAR_CONST, STRING_CONST,
                  	REDUCTION_VAR, ARITH_EXPR, LOGIC_EXPR, 
			EPOCH_EXPR, FIELD_ACC, RANGE_EXPR,
                  	ASSIGN_EXPR, INDEX_RANGE, ARRAY_ACC, 
			FN_CALL, TASK_INVOKE, OBJ_CREATE, LIB_FN_CALL };

/*      Task global variables may be synchronized/retrieved in several cases. The cases are
        Entrance: moving from a higher to a lower space
        Exit: exiting from a lower to higher space
        Return: returning from a lower to a higher space
        Reappearance: moving from one stage of a space to another of the same space
*/
enum SyncStageType { Entrance_Sync, Exit_Sync, Return_Sync, Reappearance_Sync };

/*      Depending on the type of sync stage synchronization need is different. Scenarios are
        Load: need to read some data structures into a space
        Load_And_Configure: need to read data structures and also to update metadata of those structures
        Ghost_Region_Update: need to do padding region synchronization
        Restore: Need to upload changes from below to upper region for persistence
*/
enum SyncMode { Load, Load_And_Configure, Ghost_Region_Update, Restore };

/*      Repeat Cycles can be of three types
	Range_Repeat: repitition of the indexes of an index range
        Condition_Repeat: is a generic while loop condition based repeat
	Subpartition_Repeat: iteration of an LPU sub-partitions
*/
enum RepeatCycleType { Range_Repeat, Condition_Repeat, Subpartition_Repeat };

/*	The types in this enum are needed for polymorphic stage resolution. We update statements and 
	expressions within a stage instanciation based on the argument expression types.
	Evaluate_Before: says that the argument should be evaluated at the beginning of the stage and
		a local variable with a name matching the parameter name should be created.
	No_Replacement: says that the argument and parameter names are the same and no change is needed.
	Change_Name: says that field accesses of the parameter should be renamed to access the argument.
	Update_Expr: is needed for arrays with sub-range expressions. All array element and metadata
		accesses must be changed to alternative expressions appropriate for the argument.
*/
enum FieldReplacementType { Evaluate_Before, No_Replacement, Change_Name, Update_Expr };

#endif
