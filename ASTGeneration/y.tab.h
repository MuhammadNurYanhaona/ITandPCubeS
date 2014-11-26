/* A Bison parser, made by GNU Bison 2.5.  */

/* Bison interface for Yacc-like parsers in C
   
      Copyright (C) 1984, 1989-1990, 2000-2011 Free Software Foundation, Inc.
   
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   
   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.
   
   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */


/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     Program = 258,
     Tuple = 259,
     Task = 260,
     Function = 261,
     New = 262,
     Execute = 263,
     S_Define = 264,
     S_Environment = 265,
     S_Initialize = 266,
     S_Compute = 267,
     S_Partition = 268,
     S_Arguments = 269,
     S_Results = 270,
     T_Integer = 271,
     T_Character = 272,
     T_Real = 273,
     T_Boolean = 274,
     T_Epoch = 275,
     T_Index = 276,
     T_Range = 277,
     T_Array = 278,
     T_List = 279,
     Single = 280,
     Double = 281,
     Format = 282,
     Space = 283,
     Activate = 284,
     For = 285,
     If = 286,
     Repeat = 287,
     Else = 288,
     From = 289,
     In = 290,
     Step = 291,
     Foreach = 292,
     Range = 293,
     Local = 294,
     Index = 295,
     C_Sub_Partition = 296,
     While = 297,
     Do = 298,
     Sequence = 299,
     To = 300,
     Of = 301,
     Link = 302,
     Create = 303,
     Link_or_Create = 304,
     Dynamic = 305,
     P_Sub_Partition = 306,
     Ordered = 307,
     Unordered = 308,
     Replicated = 309,
     Padding = 310,
     Relative_To = 311,
     Divides = 312,
     Sub_Partitions = 313,
     Partitions = 314,
     Unpartitioned = 315,
     Ascends = 316,
     Descends = 317,
     New_Line = 318,
     Integer = 319,
     Dimensionality = 320,
     Dimension_No = 321,
     V_Dimension = 322,
     Character = 323,
     Space_ID = 324,
     Real = 325,
     Boolean = 326,
     Type_Name = 327,
     Variable_Name = 328,
     String = 329,
     R_AVG = 330,
     R_MIN_ENTRY = 331,
     R_MAX_ENTRY = 332,
     R_MIN = 333,
     R_MAX = 334,
     R_MULT = 335,
     R_ADD = 336,
     O_OR = 337,
     O_AND = 338,
     O_NE = 339,
     O_EQ = 340,
     O_GTE = 341,
     O_LTE = 342,
     O_SB_RANGE = 343,
     O_RSH = 344,
     O_LSH = 345,
     O_POWER = 346,
     At = 347,
     Field = 348
   };
#endif
/* Tokens.  */
#define Program 258
#define Tuple 259
#define Task 260
#define Function 261
#define New 262
#define Execute 263
#define S_Define 264
#define S_Environment 265
#define S_Initialize 266
#define S_Compute 267
#define S_Partition 268
#define S_Arguments 269
#define S_Results 270
#define T_Integer 271
#define T_Character 272
#define T_Real 273
#define T_Boolean 274
#define T_Epoch 275
#define T_Index 276
#define T_Range 277
#define T_Array 278
#define T_List 279
#define Single 280
#define Double 281
#define Format 282
#define Space 283
#define Activate 284
#define For 285
#define If 286
#define Repeat 287
#define Else 288
#define From 289
#define In 290
#define Step 291
#define Foreach 292
#define Range 293
#define Local 294
#define Index 295
#define C_Sub_Partition 296
#define While 297
#define Do 298
#define Sequence 299
#define To 300
#define Of 301
#define Link 302
#define Create 303
#define Link_or_Create 304
#define Dynamic 305
#define P_Sub_Partition 306
#define Ordered 307
#define Unordered 308
#define Replicated 309
#define Padding 310
#define Relative_To 311
#define Divides 312
#define Sub_Partitions 313
#define Partitions 314
#define Unpartitioned 315
#define Ascends 316
#define Descends 317
#define New_Line 318
#define Integer 319
#define Dimensionality 320
#define Dimension_No 321
#define V_Dimension 322
#define Character 323
#define Space_ID 324
#define Real 325
#define Boolean 326
#define Type_Name 327
#define Variable_Name 328
#define String 329
#define R_AVG 330
#define R_MIN_ENTRY 331
#define R_MAX_ENTRY 332
#define R_MIN 333
#define R_MAX 334
#define R_MULT 335
#define R_ADD 336
#define O_OR 337
#define O_AND 338
#define O_NE 339
#define O_EQ 340
#define O_GTE 341
#define O_LTE 342
#define O_SB_RANGE 343
#define O_RSH 344
#define O_LSH 345
#define O_POWER 346
#define At 347
#define Field 348




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
{

/* Line 2068 of yacc.c  */
#line 25 "IT.y"

        int 				intConstant;
        bool 				booleanConstant;
        double 				doubleConstant;
	char				characterConstant;
	char 				*stringConstant;
	VariableDef 			*var;
	List<VariableDef*>		*varList;
	CoordinatorDef			*coordinator;
	ProgramDef			*program;
	TaskDef 			*task;
	FunctionDef			*fn;
	TupleDef			*tuple;	
	Node				*node;
	List<Node*>			*nodeList;
	Type				*type;
	NamedType			*nType;
	ArrayType			*aType;
	ListType			*lType;
	Expr				*expr;
	List<Expr*>			*exprList;
	Stmt				*stmt;
	List<Stmt*>			*stmtList;
	ConditionalStmt			*condStmt;
	List<ConditionalStmt*>		*condStmtList;
	IndexRangeCondition		*rangeCond;
	List<IndexRangeCondition*>	*rangeCondList;
	Identifier			*id;
	List<Identifier*>		*idList;
	List<int>			*intList;
	EpochValue			*epoch;
	OptionalInvocationParams	*invokeArgs;
	List<OptionalInvocationParams*>	*invokeArgsList;
	FunctionHeader			*fnHeader;
	InitializeInstr 		*initInstr;
	LinkageType			linkageType;
	List<EnvironmentLink*>		*envLinks;
	EnvironmentConfig		*envConfig;
	StageHeader			*stageHeader;
	ComputeStage			*stage;
	List<ComputeStage*>		*stageList;
	MetaComputeStage		*metaStage;
	List<MetaComputeStage*>		*metaStageList;
	ComputeSection			*compute;
	RepeatControl			*repeat;
	PartitionOrder			order;
	PartitionInstr			*pInstr;
	List<PartitionInstr*>		*pInstrList;
	PartitionArg			*pArg;
	List<PartitionArg*>		*pArgList;
	SpaceLinkage			*sLink;
	List<IntConstant*>		*iConList;
	VarDimensions			*vDims;
	List<VarDimensions*>		*vDimsList;
	List<DataConfigurationSpec*>	*dConSpecList;
	SubpartitionSpec		*subPartSpec;
	PartitionSpec			*partSpec;
	List<PartitionSpec*>		*partSpecList;
	PartitionSection		*partition;



/* Line 2068 of yacc.c  */
#line 299 "y.tab.h"
} YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
#endif

extern YYSTYPE yylval;

#if ! defined YYLTYPE && ! defined YYLTYPE_IS_DECLARED
typedef struct YYLTYPE
{
  int first_line;
  int first_column;
  int last_line;
  int last_column;
} YYLTYPE;
# define yyltype YYLTYPE /* obsolescent; will be withdrawn */
# define YYLTYPE_IS_DECLARED 1
# define YYLTYPE_IS_TRIVIAL 1
#endif

extern YYLTYPE yylloc;

