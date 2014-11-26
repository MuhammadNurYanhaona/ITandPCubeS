/* A Bison parser, made by GNU Bison 2.5.  */

/* Bison implementation for Yacc-like parsers in C
   
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

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "2.5"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1

/* Using locations.  */
#define YYLSP_NEEDED 1



/* Copy the first part of user declarations.  */

/* Line 268 of yacc.c  */
#line 2 "IT.y"

/* Just like lex, the text within this first region delimited by %{ and %}
 * is assumed to be C/C++ code and will be copied verbatim to the y.tab.c
 * file ahead of the definitions of the yyparse() function. Add other header
 * file inclusions or C++ variable declarations/prototypes that are needed
 * by your code here.
 */
#include "scanner.h" // for yylex
#include "parser.h"
#include "utility.h"
#include "errors.h"

void yyerror(const char *msg); // standard error-handling routine


/* Line 268 of yacc.c  */
#line 87 "y.tab.c"

/* Enabling traces.  */
#ifndef YYDEBUG
# define YYDEBUG 1
#endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* Enabling the token table.  */
#ifndef YYTOKEN_TABLE
# define YYTOKEN_TABLE 0
#endif


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

/* Line 293 of yacc.c  */
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



/* Line 293 of yacc.c  */
#line 372 "y.tab.c"
} YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
#endif

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


/* Copy the second part of user declarations.  */


/* Line 343 of yacc.c  */
#line 397 "y.tab.c"

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#elif (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
typedef signed char yytype_int8;
#else
typedef short int yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(msgid) dgettext ("bison-runtime", msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(msgid) msgid
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(e) ((void) (e))
#else
# define YYUSE(e) /* empty */
#endif

/* Identity function, used to suppress warnings about constant conditions.  */
#ifndef lint
# define YYID(n) (n)
#else
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static int
YYID (int yyi)
#else
static int
YYID (yyi)
    int yyi;
#endif
{
  return yyi;
}
#endif

#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's `empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (YYID (0))
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
	     && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
	 || (defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL \
	     && defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss_alloc;
  YYSTYPE yyvs_alloc;
  YYLTYPE yyls_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE) + sizeof (YYLTYPE)) \
      + 2 * YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)				\
    do									\
      {									\
	YYSIZE_T yynewbytes;						\
	YYCOPY (&yyptr->Stack_alloc, Stack, yysize);			\
	Stack = &yyptr->Stack_alloc;					\
	yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	yyptr += yynewbytes / sizeof (*yyptr);				\
      }									\
    while (YYID (0))

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from FROM to TO.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(To, From, Count) \
      __builtin_memcpy (To, From, (Count) * sizeof (*(From)))
#  else
#   define YYCOPY(To, From, Count)		\
      do					\
	{					\
	  YYSIZE_T yyi;				\
	  for (yyi = 0; yyi < (Count); yyi++)	\
	    (To)[yyi] = (From)[yyi];		\
	}					\
      while (YYID (0))
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  16
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   604

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  113
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  94
/* YYNRULES -- Number of rules.  */
#define YYNRULES  210
/* YYNRULES -- Number of states.  */
#define YYNSTATES  425

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   348

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,   101,     2,     2,     2,    97,     2,     2,
     106,   109,    98,    95,    75,    96,   104,    99,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,   107,   112,
      88,    76,    89,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,   105,     2,   108,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,   110,     2,   111,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      77,    78,    79,    80,    81,    82,    83,    84,    85,    86,
      87,    90,    91,    92,    93,    94,   100,   102,   103
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     5,     7,    10,    12,    14,    16,    18,
      23,    25,    28,    32,    34,    38,    40,    42,    44,    47,
      50,    52,    54,    56,    58,    60,    65,    70,    74,    83,
      87,    89,    92,    96,    98,   100,   102,   104,   108,   114,
     115,   118,   122,   124,   127,   131,   133,   135,   137,   138,
     143,   144,   148,   152,   154,   156,   159,   162,   164,   167,
     172,   177,   184,   185,   191,   197,   203,   206,   211,   216,
     218,   221,   233,   242,   243,   247,   248,   253,   258,   260,
     263,   268,   270,   274,   276,   281,   283,   287,   289,   293,
     295,   300,   309,   310,   312,   316,   318,   320,   321,   326,
     327,   338,   340,   342,   344,   347,   351,   353,   357,   360,
     361,   363,   365,   366,   369,   371,   374,   378,   380,   383,
     385,   387,   389,   391,   403,   410,   417,   419,   423,   428,
     429,   432,   441,   442,   444,   447,   452,   461,   462,   465,
     469,   473,   477,   481,   485,   489,   493,   497,   501,   505,
     509,   513,   517,   521,   524,   528,   532,   536,   540,   544,
     548,   552,   556,   560,   564,   566,   568,   570,   572,   574,
     580,   584,   586,   588,   590,   592,   594,   598,   603,   605,
     609,   611,   616,   617,   619,   621,   625,   627,   631,   633,
     635,   637,   639,   641,   648,   649,   652,   655,   661,   669,
     670,   673,   678,   682,   684,   686,   692,   694,   697,   701,
     705
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
     114,     0,    -1,   115,    -1,   116,    -1,   115,   116,    -1,
     117,    -1,   125,    -1,   202,    -1,   194,    -1,     4,    72,
     107,   118,    -1,   119,    -1,   118,   119,    -1,   120,   107,
     121,    -1,    73,    -1,   120,    75,    73,    -1,   122,    -1,
     123,    -1,    16,    -1,    18,    25,    -1,    18,    26,    -1,
      17,    -1,    19,    -1,    20,    -1,    22,    -1,    72,    -1,
      23,   124,    46,   122,    -1,   124,   105,    64,   108,    -1,
     105,    64,   108,    -1,     5,    74,   107,   126,   134,   138,
     140,   149,    -1,     9,   107,   127,    -1,   128,    -1,   127,
     128,    -1,   120,   107,   129,    -1,   121,    -1,   130,    -1,
     131,    -1,   132,    -1,    24,    46,   121,    -1,    65,    23,
      46,   122,   133,    -1,    -1,    27,    72,    -1,    10,   107,
     135,    -1,   136,    -1,   135,   136,    -1,   120,   107,   137,
      -1,    47,    -1,    48,    -1,    49,    -1,    -1,    11,   139,
     107,   171,    -1,    -1,   106,   120,   109,    -1,    12,   107,
     141,    -1,   143,    -1,   142,    -1,   142,   141,    -1,   143,
     147,    -1,   144,    -1,   143,   144,    -1,   145,   110,   141,
     111,    -1,   145,   110,   171,   111,    -1,    74,   106,    28,
      69,   109,   146,    -1,    -1,    29,    31,    73,    35,   186,
      -1,    32,   107,    34,    74,   148,    -1,    30,    73,    35,
     186,   185,    -1,    42,   186,    -1,    37,    28,    69,    41,
      -1,    13,   139,   107,   150,    -1,   151,    -1,   150,   151,
      -1,    28,    69,    88,    65,    89,   152,   153,   110,   154,
     164,   111,    -1,    28,    69,    88,    60,    89,   110,   120,
     111,    -1,    -1,    88,    50,    89,    -1,    -1,    57,    28,
      69,    59,    -1,    57,    28,    69,    58,    -1,   155,    -1,
     154,   155,    -1,   156,   107,   159,   163,    -1,   157,    -1,
     156,    75,   157,    -1,    73,    -1,    73,    88,   158,    89,
      -1,    67,    -1,   158,    75,    67,    -1,   160,    -1,   159,
      75,   160,    -1,    54,    -1,    73,   106,   161,   109,    -1,
      73,   106,   161,   109,    55,   106,   161,   109,    -1,    -1,
     162,    -1,   161,    75,   162,    -1,    73,    -1,    64,    -1,
      -1,   112,    56,    28,    69,    -1,    -1,    51,    88,    65,
      89,    88,   165,    89,   110,   166,   111,    -1,    52,    -1,
      53,    -1,   167,    -1,   166,   167,    -1,   156,   107,   168,
      -1,   169,    -1,   168,    75,   169,    -1,   160,   170,    -1,
      -1,    61,    -1,    62,    -1,    -1,   172,   173,    -1,   175,
      -1,   175,   174,    -1,   175,   174,   173,    -1,    63,    -1,
      63,   174,    -1,   177,    -1,   176,    -1,   181,    -1,   186,
      -1,    43,    35,    44,   110,   173,   111,    30,   193,    35,
     186,   185,    -1,    43,   110,   173,   111,    30,   178,    -1,
      43,   110,   173,   111,    42,   186,    -1,   179,    -1,   178,
     112,   179,    -1,   120,    35,    73,   180,    -1,    -1,    85,
     186,    -1,    31,   106,   186,   109,   110,   173,   111,   182,
      -1,    -1,   183,    -1,   184,   182,    -1,    33,   110,   173,
     111,    -1,    33,    31,   106,   186,   109,   110,   173,   111,
      -1,    -1,    36,   186,    -1,   186,    95,   186,    -1,   186,
      96,   186,    -1,   186,    98,   186,    -1,   186,    99,   186,
      -1,   186,    97,   186,    -1,   186,    94,   186,    -1,   186,
      93,   186,    -1,   186,   100,   186,    -1,   186,    88,   186,
      -1,   186,    89,   186,    -1,   186,    84,   186,    -1,   186,
      85,   186,    -1,   186,    87,   186,    -1,   186,    86,   186,
      -1,   101,   186,    -1,   186,    90,   186,    -1,   186,    91,
     186,    -1,   186,    76,   186,    -1,   186,    83,   186,    -1,
     186,    82,   186,    -1,   186,    81,   186,    -1,   186,    80,
     186,    -1,   186,    77,   186,    -1,   186,    79,   186,    -1,
     186,    78,   186,    -1,   187,    -1,   188,    -1,   190,    -1,
     198,    -1,   197,    -1,   186,   102,   106,   192,   109,    -1,
     106,   186,   109,    -1,    64,    -1,    70,    -1,    71,    -1,
      68,    -1,   193,    -1,   188,   104,   193,    -1,   188,   105,
     189,   108,    -1,   186,    -1,   186,    92,   186,    -1,    92,
      -1,    73,   106,   191,   109,    -1,    -1,    74,    -1,   186,
      -1,   191,    75,   186,    -1,    73,    -1,    73,    96,    64,
      -1,    73,    -1,    66,    -1,    38,    -1,    39,    -1,    40,
      -1,     3,   106,    73,   109,   107,   195,    -1,    -1,   196,
     173,    -1,     7,   130,    -1,     7,   121,   106,   191,   109,
      -1,     8,   106,    74,   112,   193,   199,   109,    -1,    -1,
     112,   200,    -1,   112,   200,   112,   200,    -1,   201,   107,
     191,    -1,    11,    -1,    13,    -1,     6,    73,   107,   203,
     206,    -1,   205,    -1,   204,   205,    -1,    14,   107,   127,
      -1,    15,   107,   127,    -1,    12,   107,   171,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   180,   180,   185,   186,   187,   187,   187,   187,   190,
     191,   192,   193,   194,   195,   196,   196,   197,   198,   199,
     200,   201,   202,   203,   204,   205,   209,   210,   214,   220,
     221,   222,   223,   224,   224,   225,   225,   226,   227,   229,
     229,   233,   234,   235,   236,   237,   238,   239,   243,   244,
     245,   246,   250,   251,   252,   253,   254,   255,   256,   257,
     258,   259,   261,   262,   263,   264,   265,   266,   270,   271,
     272,   273,   275,   277,   278,   279,   280,   281,   282,   283,
     284,   285,   286,   287,   288,   289,   290,   291,   292,   293,
     294,   296,   298,   299,   300,   301,   302,   303,   304,   305,
     306,   308,   309,   310,   311,   312,   313,   314,   315,   316,
     317,   318,   322,   322,   323,   324,   325,   326,   326,   327,
     328,   329,   330,   331,   333,   334,   335,   336,   337,   338,
     339,   340,   344,   345,   346,   347,   348,   350,   351,   352,
     353,   354,   355,   356,   357,   358,   359,   360,   361,   362,
     363,   364,   365,   366,   367,   368,   369,   370,   371,   372,
     373,   374,   375,   376,   377,   378,   379,   380,   381,   382,
     383,   384,   385,   386,   387,   388,   389,   390,   391,   392,
     393,   394,   395,   396,   397,   398,   399,   400,   401,   402,
     403,   404,   405,   409,   410,   410,   411,   412,   413,   414,
     415,   416,   418,   419,   420,   424,   426,   427,   428,   429,
     430
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "Program", "Tuple", "Task", "Function",
  "New", "Execute", "S_Define", "S_Environment", "S_Initialize",
  "S_Compute", "S_Partition", "S_Arguments", "S_Results", "T_Integer",
  "T_Character", "T_Real", "T_Boolean", "T_Epoch", "T_Index", "T_Range",
  "T_Array", "T_List", "Single", "Double", "Format", "Space", "Activate",
  "For", "If", "Repeat", "Else", "From", "In", "Step", "Foreach", "Range",
  "Local", "Index", "C_Sub_Partition", "While", "Do", "Sequence", "To",
  "Of", "Link", "Create", "Link_or_Create", "Dynamic", "P_Sub_Partition",
  "Ordered", "Unordered", "Replicated", "Padding", "Relative_To",
  "Divides", "Sub_Partitions", "Partitions", "Unpartitioned", "Ascends",
  "Descends", "New_Line", "Integer", "Dimensionality", "Dimension_No",
  "V_Dimension", "Character", "Space_ID", "Real", "Boolean", "Type_Name",
  "Variable_Name", "String", "','", "'='", "R_AVG", "R_MIN_ENTRY",
  "R_MAX_ENTRY", "R_MIN", "R_MAX", "R_MULT", "R_ADD", "O_OR", "O_AND",
  "O_NE", "O_EQ", "'<'", "'>'", "O_GTE", "O_LTE", "O_SB_RANGE", "O_RSH",
  "O_LSH", "'+'", "'-'", "'%'", "'*'", "'/'", "O_POWER", "'!'", "At",
  "Field", "'.'", "'['", "'('", "':'", "']'", "')'", "'{'", "'}'", "';'",
  "$accept", "program", "components", "component", "tuple", "element_defs",
  "element_def", "names", "static_type", "scalar_type", "static_array",
  "static_dims", "task", "define", "definitions", "definition", "type",
  "dynamic_type", "list", "dynamic_array", "format", "environment",
  "linkages", "linkage", "mode", "initialize", "arguments", "compute",
  "meta_stages", "meta_stage", "stage_sequence", "compute_stage",
  "stage_header", "activation_command", "repeat_control", "repeat_loop",
  "partition", "partition_specs", "partition_spec", "dynamic", "divides",
  "main_dist", "data_spec", "var_list", "var", "dimensions", "instr_list",
  "instr", "partition_args", "partition_arg", "relativity", "sub_dist",
  "nature", "data_sub_dist", "data_sub_spec", "ordered_instr_list",
  "ordered_instr", "order", "code", "$@1", "stmt_block", "new_lines",
  "stmt", "sequencial_loop", "parallel_loop", "index_ranges",
  "index_range", "restrictions", "if_else_block", "else_block", "else",
  "else_if", "step_expr", "expr", "constant", "field", "array_index",
  "function_call", "args", "epoch", "id", "coordinator", "meta_code",
  "$@2", "create_obj", "task_invocation", "optional_secs", "optional_sec",
  "section", "function", "in_out", "input", "output", "function_body", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315,   316,   317,   318,   319,   320,   321,   322,   323,   324,
     325,   326,   327,   328,   329,    44,    61,   330,   331,   332,
     333,   334,   335,   336,   337,   338,   339,   340,    60,    62,
     341,   342,   343,   344,   345,    43,    45,    37,    42,    47,
     346,    33,   347,   348,    46,    91,    40,    58,    93,    41,
     123,   125,    59
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,   113,   114,   115,   115,   116,   116,   116,   116,   117,
     118,   118,   119,   120,   120,   121,   121,   122,   122,   122,
     122,   122,   122,   122,   122,   123,   124,   124,   125,   126,
     127,   127,   128,   129,   129,   130,   130,   131,   132,   133,
     133,   134,   135,   135,   136,   137,   137,   137,   138,   138,
     139,   139,   140,   141,   141,   141,   142,   143,   143,   144,
     144,   145,   146,   146,   147,   148,   148,   148,   149,   150,
     150,   151,   151,   152,   152,   153,   153,   153,   154,   154,
     155,   156,   156,   157,   157,   158,   158,   159,   159,   160,
     160,   160,   161,   161,   161,   162,   162,   163,   163,   164,
     164,   165,   165,   166,   166,   167,   168,   168,   169,   170,
     170,   170,   172,   171,   173,   173,   173,   174,   174,   175,
     175,   175,   175,   176,   177,   177,   178,   178,   179,   180,
     180,   181,   182,   182,   182,   183,   184,   185,   185,   186,
     186,   186,   186,   186,   186,   186,   186,   186,   186,   186,
     186,   186,   186,   186,   186,   186,   186,   186,   186,   186,
     186,   186,   186,   186,   186,   186,   186,   186,   186,   186,
     186,   187,   187,   187,   187,   188,   188,   188,   189,   189,
     189,   190,   191,   191,   191,   191,   192,   192,   193,   193,
     193,   193,   193,   194,   196,   195,   197,   197,   198,   199,
     199,   199,   200,   201,   201,   202,   203,   203,   204,   205,
     206
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     1,     2,     1,     1,     1,     1,     4,
       1,     2,     3,     1,     3,     1,     1,     1,     2,     2,
       1,     1,     1,     1,     1,     4,     4,     3,     8,     3,
       1,     2,     3,     1,     1,     1,     1,     3,     5,     0,
       2,     3,     1,     2,     3,     1,     1,     1,     0,     4,
       0,     3,     3,     1,     1,     2,     2,     1,     2,     4,
       4,     6,     0,     5,     5,     5,     2,     4,     4,     1,
       2,    11,     8,     0,     3,     0,     4,     4,     1,     2,
       4,     1,     3,     1,     4,     1,     3,     1,     3,     1,
       4,     8,     0,     1,     3,     1,     1,     0,     4,     0,
      10,     1,     1,     1,     2,     3,     1,     3,     2,     0,
       1,     1,     0,     2,     1,     2,     3,     1,     2,     1,
       1,     1,     1,    11,     6,     6,     1,     3,     4,     0,
       2,     8,     0,     1,     2,     4,     8,     0,     2,     3,
       3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
       3,     3,     3,     2,     3,     3,     3,     3,     3,     3,
       3,     3,     3,     3,     1,     1,     1,     1,     1,     5,
       3,     1,     1,     1,     1,     1,     3,     4,     1,     3,
       1,     4,     0,     1,     1,     3,     1,     3,     1,     1,
       1,     1,     1,     6,     0,     2,     2,     5,     7,     0,
       2,     4,     3,     1,     1,     5,     1,     2,     3,     3,
       3
};

/* YYDEFACT[STATE-NAME] -- Default reduction number in state STATE-NUM.
   Performed when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       0,     0,     0,     0,     0,     0,     2,     3,     5,     6,
       8,     7,     0,     0,     0,     0,     1,     4,     0,     0,
       0,     0,     0,    13,     9,    10,     0,     0,     0,     0,
       0,     0,     0,   206,   194,    11,     0,     0,     0,     0,
      48,     0,     0,     0,   205,   207,   193,     0,    14,    17,
      20,     0,    21,    22,    23,     0,    24,    12,    15,    16,
       0,    29,    30,     0,    50,     0,   208,   209,   112,     0,
       0,     0,   190,   191,   192,     0,   171,   189,   174,   172,
     173,   188,     0,     0,   195,   114,   120,   119,   121,   122,
     164,   165,   166,   175,   168,   167,    18,    19,     0,     0,
       0,    31,     0,    41,    42,     0,     0,     0,     0,   210,
       0,     0,     0,     0,   196,    35,    36,     0,     0,     0,
       0,   182,   153,     0,   117,   115,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    33,    32,    34,     0,
      43,     0,   112,     0,    50,    28,   113,     0,     0,   182,
       0,     0,     0,     0,   183,   184,     0,   170,   118,   116,
     156,   161,   163,   162,   160,   159,   158,   157,   149,   150,
     152,   151,   147,   148,   154,   155,   145,   144,   139,   140,
     143,   141,   142,   146,     0,   188,   176,   180,   178,     0,
      27,    25,     0,    45,    46,    47,    44,    51,    49,     0,
      52,    54,    53,    57,     0,     0,    37,     0,     0,     0,
       0,     0,     0,     0,   181,   186,     0,     0,   177,    26,
       0,    55,     0,    58,    56,   112,     0,    39,   197,   199,
       0,     0,     0,     0,   185,     0,   169,   179,     0,     0,
       0,     0,     0,    68,    69,     0,    38,     0,     0,     0,
       0,     0,   124,   126,   125,   187,     0,     0,    59,    60,
       0,    70,    40,   203,   204,   200,     0,   198,   132,     0,
       0,     0,    62,     0,     0,     0,   182,     0,   131,   133,
     132,     0,   129,   127,     0,    61,     0,     0,     0,    64,
       0,     0,   201,   202,     0,     0,   134,     0,     0,   128,
       0,     0,     0,    66,     0,    73,     0,     0,   137,   130,
       0,     0,     0,     0,     0,    75,     0,   135,     0,   123,
       0,   137,    67,     0,     0,     0,     0,     0,   138,    63,
      65,    72,    74,     0,     0,     0,     0,    83,    99,    78,
       0,    81,     0,    77,    76,     0,     0,    79,     0,     0,
       0,   136,    85,     0,     0,    71,    82,    89,     0,    97,
      87,     0,    84,     0,    92,     0,     0,    80,    86,     0,
      96,    95,     0,    93,    88,     0,     0,     0,    90,     0,
     101,   102,     0,    94,     0,    98,     0,    92,     0,     0,
       0,     0,   103,    91,     0,   100,   104,   109,   105,   106,
     110,   111,   108,     0,   107
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     5,     6,     7,     8,    24,    25,    60,    57,    58,
      59,    99,     9,    28,    61,    62,   157,   114,   115,   116,
     266,    40,   103,   104,   216,    65,   106,   108,   220,   221,
     222,   223,   224,   305,   244,   309,   165,   263,   264,   335,
     346,   358,   359,   360,   361,   373,   379,   417,   392,   393,
     387,   368,   402,   411,   412,   418,   419,   422,   109,   110,
      84,   125,    85,    86,    87,   272,   273,   319,    88,   298,
     299,   300,   339,    89,    90,    91,   209,    92,   176,   236,
      93,    10,    46,    47,    94,    95,   268,   285,   286,    11,
      31,    32,    33,    44
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -230
static const yytype_int16 yypact[] =
{
      16,   -25,    20,    30,    37,   140,    16,  -230,  -230,  -230,
    -230,  -230,   116,    73,    85,    92,  -230,  -230,   105,   121,
     192,   170,   113,  -230,   121,  -230,   -66,   117,   218,   131,
     133,   223,   224,  -230,  -230,  -230,   171,    96,   121,   139,
     244,   121,   121,   145,  -230,  -230,  -230,    68,  -230,  -230,
    -230,   161,  -230,  -230,  -230,   153,  -230,  -230,  -230,  -230,
      47,   121,  -230,   121,   155,   248,   121,   121,  -230,     7,
     156,   158,  -230,  -230,  -230,   -23,  -230,  -230,  -230,  -230,
    -230,   159,   203,   203,  -230,   200,  -230,  -230,  -230,   397,
    -230,    93,  -230,  -230,  -230,  -230,  -230,  -230,   202,   -35,
       7,  -230,    74,   121,  -230,   121,   163,   172,   255,  -230,
      68,   232,   257,   175,  -230,  -230,  -230,   208,   203,   239,
      68,    87,   196,   268,   200,    68,   203,   203,   203,   203,
     203,   203,   203,   203,   203,   203,   203,   203,   203,   203,
     203,   203,   203,   203,   203,   203,   203,   203,   203,   203,
     194,    62,   183,   193,   128,   238,  -230,  -230,  -230,   118,
    -230,   -40,  -230,   229,   155,  -230,  -230,    96,   259,    87,
     195,   302,   198,   199,  -230,   397,     8,  -230,  -230,  -230,
     397,   397,   397,   397,   397,   397,   397,   397,   415,   432,
     197,   197,   442,   442,   442,   442,   107,   107,   132,   132,
     196,   196,   196,   196,   233,  -230,  -230,  -230,   370,   204,
    -230,  -230,   205,  -230,  -230,  -230,  -230,  -230,  -230,   209,
    -230,   229,     0,  -230,   201,   226,  -230,   128,    21,    62,
     206,    68,   -14,   203,  -230,   246,   251,   203,  -230,  -230,
     341,  -230,   264,  -230,  -230,   229,   344,   346,  -230,   262,
      68,   265,   121,   203,   397,   311,  -230,   397,   325,   369,
     294,   295,   338,   344,  -230,   337,  -230,   110,   299,   317,
     380,    -2,   327,  -230,   397,  -230,   328,   366,  -230,  -230,
     353,  -230,  -230,  -230,  -230,   330,   364,  -230,   410,    62,
     371,   121,   460,   141,    33,   110,    87,   -21,  -230,  -230,
     410,   463,   422,  -230,   485,  -230,   451,   505,   203,  -230,
     454,   456,  -230,   471,   441,    68,  -230,   203,   203,  -230,
     475,   514,   481,   397,   443,   464,   203,   440,   241,   397,
     519,   203,   515,   121,   507,   498,   336,  -230,   203,  -230,
     203,   241,  -230,   -33,   469,   531,   450,   452,   397,   397,
    -230,  -230,  -230,   492,   490,    68,   154,   476,   -37,  -230,
      88,  -230,   455,  -230,  -230,   500,   477,  -230,   457,   490,
     -39,  -230,  -230,    84,   504,  -230,  -230,  -230,   465,   -32,
    -230,   503,  -230,   483,    56,   -39,   517,  -230,  -230,   486,
    -230,  -230,    22,  -230,  -230,   547,   164,    56,   521,   508,
    -230,  -230,   489,  -230,   473,  -230,   470,    56,   490,    28,
      89,   -34,  -230,  -230,   -39,  -230,  -230,   157,   506,  -230,
    -230,  -230,  -230,   -39,  -230
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -230,  -230,  -230,   576,  -230,  -230,   559,   -19,   -62,  -137,
    -230,  -230,  -230,  -230,   184,   109,  -230,   484,  -230,  -230,
    -230,  -230,  -230,   482,  -230,  -230,   423,  -230,  -136,  -230,
    -230,   367,  -230,  -230,  -230,  -230,  -230,  -230,   323,  -230,
    -230,  -230,   230,  -229,   221,  -230,  -230,  -208,   185,   207,
    -230,  -230,  -230,  -230,   180,  -230,   173,  -230,  -154,  -230,
    -107,   474,  -230,  -230,  -230,  -230,   303,  -230,  -230,   293,
    -230,  -230,   254,   -81,  -230,  -230,  -230,  -230,  -163,  -230,
    -147,  -230,  -230,  -230,  -230,  -230,  -230,   304,  -230,  -230,
    -230,  -230,   565,  -230
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -1
static const yytype_uint16 yytable[] =
{
      26,   122,   123,   166,   206,    26,   228,   113,   218,    36,
     314,   154,   119,   173,   366,   377,   252,   211,   179,     1,
       2,     3,     4,    49,    50,    51,    52,    53,   253,    54,
      55,   111,   242,   290,   378,    36,   357,   171,   156,   357,
     175,    37,    36,   385,   102,   180,   181,   182,   183,   184,
     185,   186,   187,   188,   189,   190,   191,   192,   193,   194,
     195,   196,   197,   198,   199,   200,   201,   202,   203,   217,
     155,   208,   112,    36,   219,    69,    70,   415,   351,    56,
     386,    12,   249,   233,   102,   241,   161,   120,   175,   315,
     247,   261,    13,   310,    69,    70,   233,   397,   311,    71,
      72,    73,    74,   397,    14,   226,    72,    73,    74,   260,
      15,    75,    49,    50,    51,    52,    53,   234,    54,    55,
     390,   283,    36,   284,   251,    72,    73,    74,    77,   391,
     248,   398,    76,   313,    77,   205,    78,   413,    79,    80,
      16,    81,   301,   269,    49,    50,    51,    52,    53,    36,
      54,    76,   254,    77,   100,    78,   257,    79,    80,   381,
      81,   174,   380,   369,   369,   213,   214,   215,    56,    82,
     101,   306,   274,   382,    83,   101,   101,   394,   307,   410,
      19,   159,   410,   308,    29,    30,    96,    97,    82,    18,
      69,    70,    20,    83,    23,   370,   414,   151,   152,    21,
      56,    27,   144,   145,   146,   147,   148,   149,   327,   150,
      69,    70,   363,   364,    22,   175,   400,   401,   420,   421,
      34,    72,    73,    74,    38,    66,    67,   323,    39,   146,
     147,   148,   149,   271,   150,    43,   328,   329,    41,    30,
      42,    72,    73,    74,    48,   336,    63,    76,   362,    77,
     341,    78,    68,    79,    80,    64,    81,   348,    98,   349,
     107,   105,   117,   124,   118,   121,   153,    76,   164,    77,
     162,    78,   271,    79,    80,   207,    81,   338,   167,   163,
     168,   169,   170,   172,    82,   138,   139,   140,   141,    83,
     142,   143,   144,   145,   146,   147,   148,   149,   150,   150,
     204,   210,   212,   219,    82,   227,   235,   229,   231,    83,
     232,   245,   238,   239,   343,   240,   250,   126,   127,   128,
     129,   130,   131,   132,   133,   134,   135,   136,   137,   138,
     139,   140,   141,   246,   142,   143,   144,   145,   146,   147,
     148,   149,   255,   150,   126,   127,   128,   129,   130,   131,
     132,   133,   134,   135,   136,   137,   138,   139,   140,   141,
     256,   142,   143,   144,   145,   146,   147,   148,   149,   258,
     150,   259,   262,   265,   267,   275,   270,   177,   126,   127,
     128,   129,   130,   131,   132,   133,   134,   135,   136,   137,
     138,   139,   140,   141,   276,   142,   143,   144,   145,   146,
     147,   148,   149,   277,   150,   278,   279,   280,   287,   282,
     289,   230,   126,   127,   128,   129,   130,   131,   132,   133,
     134,   135,   136,   137,   138,   139,   140,   141,   288,   142,
     143,   144,   145,   146,   147,   148,   149,   292,   150,   291,
     293,   294,   295,   297,   302,   347,   126,   127,   128,   129,
     130,   131,   132,   133,   134,   135,   136,   137,   138,   139,
     140,   141,   237,   142,   143,   144,   145,   146,   147,   148,
     149,   296,   150,   126,   127,   128,   129,   130,   131,   132,
     133,   134,   135,   136,   137,   138,   139,   140,   141,   304,
     142,   143,   144,   145,   146,   147,   148,   149,   317,   150,
     135,   136,   137,   138,   139,   140,   141,   318,   142,   143,
     144,   145,   146,   147,   148,   149,   320,   150,   136,   137,
     138,   139,   140,   141,   321,   142,   143,   144,   145,   146,
     147,   148,   149,   322,   150,   142,   143,   144,   145,   146,
     147,   148,   149,   324,   150,   325,   233,   326,   330,   331,
     332,   337,   334,   333,   340,   345,   342,   344,   352,   353,
     354,   356,   355,   357,   365,   374,   371,   372,   375,   383,
     388,   384,   389,   395,   396,   399,   404,   405,   406,   407,
     408,   423,    17,    35,   158,   160,   281,   225,   367,   243,
     376,   416,   409,   316,   303,   350,   424,    45,   178,   312,
       0,     0,     0,     0,   403
};

#define yypact_value_is_default(yystate) \
  ((yystate) == (-230))

#define yytable_value_is_error(yytable_value) \
  YYID (0)

static const yytype_int16 yycheck[] =
{
      19,    82,    83,   110,   151,    24,   169,    69,   162,    75,
      31,    46,    35,   120,    51,    54,    30,   154,   125,     3,
       4,     5,     6,    16,    17,    18,    19,    20,    42,    22,
      23,    24,    32,    35,    73,    75,    73,   118,   100,    73,
     121,   107,    75,    75,    63,   126,   127,   128,   129,   130,
     131,   132,   133,   134,   135,   136,   137,   138,   139,   140,
     141,   142,   143,   144,   145,   146,   147,   148,   149,   109,
     105,   152,    65,    75,    74,     7,     8,   111,   111,    72,
     112,   106,   229,    75,   103,   221,   105,   110,   169,   110,
     227,   245,    72,    60,     7,     8,    75,    75,    65,    31,
      38,    39,    40,    75,    74,   167,    38,    39,    40,   245,
      73,    43,    16,    17,    18,    19,    20,   109,    22,    23,
      64,    11,    75,    13,   231,    38,    39,    40,    66,    73,
     109,   109,    64,   296,    66,    73,    68,   109,    70,    71,
       0,    73,   289,   250,    16,    17,    18,    19,    20,    75,
      22,    64,   233,    66,   107,    68,   237,    70,    71,    75,
      73,    74,   370,    75,    75,    47,    48,    49,    72,   101,
      61,    30,   253,    89,   106,    66,    67,   385,    37,   408,
     107,   107,   411,    42,    14,    15,    25,    26,   101,    73,
       7,     8,   107,   106,    73,   107,   107,   104,   105,   107,
      72,     9,    95,    96,    97,    98,    99,   100,   315,   102,
       7,     8,    58,    59,   109,   296,    52,    53,    61,    62,
     107,    38,    39,    40,   107,    41,    42,   308,    10,    97,
      98,    99,   100,   252,   102,    12,   317,   318,   107,    15,
     107,    38,    39,    40,    73,   326,   107,    64,   355,    66,
     331,    68,   107,    70,    71,    11,    73,   338,   105,   340,
      12,   106,   106,    63,   106,   106,    64,    64,    13,    66,
     107,    68,   291,    70,    71,    92,    73,    36,    46,   107,
      23,   106,    74,    44,   101,    88,    89,    90,    91,   106,
      93,    94,    95,    96,    97,    98,    99,   100,   102,   102,
     106,   108,    64,    74,   101,    46,    73,   112,   110,   106,
     111,   110,   108,   108,   333,   106,   110,    76,    77,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    91,   107,    93,    94,    95,    96,    97,    98,
      99,   100,    96,   102,    76,    77,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
     109,    93,    94,    95,    96,    97,    98,    99,   100,    28,
     102,   107,    28,    27,   112,    64,   111,   109,    76,    77,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    69,    93,    94,    95,    96,    97,
      98,    99,   100,    34,   102,   111,   111,    69,   109,    72,
      30,   109,    76,    77,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    88,    89,    90,    91,   111,    93,
      94,    95,    96,    97,    98,    99,   100,   109,   102,   112,
      74,    88,   112,    33,    73,   109,    76,    77,    78,    79,
      80,    81,    82,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    92,    93,    94,    95,    96,    97,    98,    99,
     100,   107,   102,    76,    77,    78,    79,    80,    81,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    29,
      93,    94,    95,    96,    97,    98,    99,   100,    35,   102,
      85,    86,    87,    88,    89,    90,    91,    85,    93,    94,
      95,    96,    97,    98,    99,   100,    31,   102,    86,    87,
      88,    89,    90,    91,    73,    93,    94,    95,    96,    97,
      98,    99,   100,    28,   102,    93,    94,    95,    96,    97,
      98,    99,   100,    89,   102,    89,    75,   106,    73,    35,
      69,   111,    88,   110,    35,    57,    41,    50,    89,    28,
     110,    69,   110,    73,    88,    88,   111,    67,   111,    65,
      67,   106,    89,    56,    88,    28,    55,    69,    89,   106,
     110,    75,     6,    24,   100,   103,   263,   164,   358,   222,
     369,   411,   407,   300,   291,   341,   423,    32,   124,   295,
      -1,    -1,    -1,    -1,   397
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     4,     5,     6,   114,   115,   116,   117,   125,
     194,   202,   106,    72,    74,    73,     0,   116,    73,   107,
     107,   107,   109,    73,   118,   119,   120,     9,   126,    14,
      15,   203,   204,   205,   107,   119,    75,   107,   107,    10,
     134,   107,   107,    12,   206,   205,   195,   196,    73,    16,
      17,    18,    19,    20,    22,    23,    72,   121,   122,   123,
     120,   127,   128,   107,    11,   138,   127,   127,   107,     7,
       8,    31,    38,    39,    40,    43,    64,    66,    68,    70,
      71,    73,   101,   106,   173,   175,   176,   177,   181,   186,
     187,   188,   190,   193,   197,   198,    25,    26,   105,   124,
     107,   128,   120,   135,   136,   106,   139,    12,   140,   171,
     172,    24,    65,   121,   130,   131,   132,   106,   106,    35,
     110,   106,   186,   186,    63,   174,    76,    77,    78,    79,
      80,    81,    82,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    93,    94,    95,    96,    97,    98,    99,   100,
     102,   104,   105,    64,    46,   105,   121,   129,   130,   107,
     136,   120,   107,   107,    13,   149,   173,    46,    23,   106,
      74,   186,    44,   173,    74,   186,   191,   109,   174,   173,
     186,   186,   186,   186,   186,   186,   186,   186,   186,   186,
     186,   186,   186,   186,   186,   186,   186,   186,   186,   186,
     186,   186,   186,   186,   106,    73,   193,    92,   186,   189,
     108,   122,    64,    47,    48,    49,   137,   109,   171,    74,
     141,   142,   143,   144,   145,   139,   121,    46,   191,   112,
     109,   110,   111,    75,   109,    73,   192,    92,   108,   108,
     106,   141,    32,   144,   147,   110,   107,   122,   109,   193,
     110,   173,    30,    42,   186,    96,   109,   186,    28,   107,
     141,   171,    28,   150,   151,    27,   133,   112,   199,   173,
     111,   120,   178,   179,   186,    64,    69,    34,   111,   111,
      69,   151,    72,    11,    13,   200,   201,   109,   111,    30,
      35,   112,   109,    74,    88,   112,   107,    33,   182,   183,
     184,   193,    73,   179,    29,   146,    30,    37,    42,   148,
      60,    65,   200,   191,    31,   110,   182,    35,    85,   180,
      31,    73,    28,   186,    89,    89,   106,   173,   186,   186,
      73,    35,    69,   110,    88,   152,   186,   111,    36,   185,
      35,   186,    41,   120,    50,    57,   153,   109,   186,   186,
     185,   111,    89,    28,   110,   110,    69,    73,   154,   155,
     156,   157,   173,    58,    59,    88,    51,   155,   164,    75,
     107,   111,    67,   158,    88,   111,   157,    54,    73,   159,
     160,    75,    89,    65,   106,    75,   112,   163,    67,    89,
      64,    73,   161,   162,   160,    56,    88,    75,   109,    28,
      52,    53,   165,   162,    55,    69,    89,   106,   110,   161,
     156,   166,   167,   109,   107,   111,   167,   160,   168,   169,
      61,    62,   170,    75,   169
};

#define yyerrok		(yyerrstatus = 0)
#define yyclearin	(yychar = YYEMPTY)
#define YYEMPTY		(-2)
#define YYEOF		0

#define YYACCEPT	goto yyacceptlab
#define YYABORT		goto yyabortlab
#define YYERROR		goto yyerrorlab


/* Like YYERROR except do call yyerror.  This remains here temporarily
   to ease the transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  However,
   YYFAIL appears to be in use.  Nevertheless, it is formally deprecated
   in Bison 2.4.2's NEWS entry, where a plan to phase it out is
   discussed.  */

#define YYFAIL		goto yyerrlab
#if defined YYFAIL
  /* This is here to suppress warnings from the GCC cpp's
     -Wunused-macros.  Normally we don't worry about that warning, but
     some users do, and we want to make it easy for users to remove
     YYFAIL uses, which will produce warnings from Bison 2.5.  */
#endif

#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)					\
do								\
  if (yychar == YYEMPTY && yylen == 1)				\
    {								\
      yychar = (Token);						\
      yylval = (Value);						\
      YYPOPSTACK (1);						\
      goto yybackup;						\
    }								\
  else								\
    {								\
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;							\
    }								\
while (YYID (0))


#define YYTERROR	1
#define YYERRCODE	256


/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#define YYRHSLOC(Rhs, K) ((Rhs)[K])
#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)				\
    do									\
      if (YYID (N))                                                    \
	{								\
	  (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;	\
	  (Current).first_column = YYRHSLOC (Rhs, 1).first_column;	\
	  (Current).last_line    = YYRHSLOC (Rhs, N).last_line;		\
	  (Current).last_column  = YYRHSLOC (Rhs, N).last_column;	\
	}								\
      else								\
	{								\
	  (Current).first_line   = (Current).last_line   =		\
	    YYRHSLOC (Rhs, 0).last_line;				\
	  (Current).first_column = (Current).last_column =		\
	    YYRHSLOC (Rhs, 0).last_column;				\
	}								\
    while (YYID (0))
#endif


/* YY_LOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

#ifndef YY_LOCATION_PRINT
# if defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL
#  define YY_LOCATION_PRINT(File, Loc)			\
     fprintf (File, "%d.%d-%d.%d",			\
	      (Loc).first_line, (Loc).first_column,	\
	      (Loc).last_line,  (Loc).last_column)
# else
#  define YY_LOCATION_PRINT(File, Loc) ((void) 0)
# endif
#endif


/* YYLEX -- calling `yylex' with the right arguments.  */

#ifdef YYLEX_PARAM
# define YYLEX yylex (YYLEX_PARAM)
#else
# define YYLEX yylex ()
#endif

/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)			\
do {						\
  if (yydebug)					\
    YYFPRINTF Args;				\
} while (YYID (0))

# define YY_SYMBOL_PRINT(Title, Type, Value, Location)			  \
do {									  \
  if (yydebug)								  \
    {									  \
      YYFPRINTF (stderr, "%s ", Title);					  \
      yy_symbol_print (stderr,						  \
		  Type, Value, Location); \
      YYFPRINTF (stderr, "\n");						  \
    }									  \
} while (YYID (0))


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep, YYLTYPE const * const yylocationp)
#else
static void
yy_symbol_value_print (yyoutput, yytype, yyvaluep, yylocationp)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
    YYLTYPE const * const yylocationp;
#endif
{
  if (!yyvaluep)
    return;
  YYUSE (yylocationp);
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# else
  YYUSE (yyoutput);
# endif
  switch (yytype)
    {
      default:
	break;
    }
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep, YYLTYPE const * const yylocationp)
#else
static void
yy_symbol_print (yyoutput, yytype, yyvaluep, yylocationp)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
    YYLTYPE const * const yylocationp;
#endif
{
  if (yytype < YYNTOKENS)
    YYFPRINTF (yyoutput, "token %s (", yytname[yytype]);
  else
    YYFPRINTF (yyoutput, "nterm %s (", yytname[yytype]);

  YY_LOCATION_PRINT (yyoutput, *yylocationp);
  YYFPRINTF (yyoutput, ": ");
  yy_symbol_value_print (yyoutput, yytype, yyvaluep, yylocationp);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_stack_print (yytype_int16 *yybottom, yytype_int16 *yytop)
#else
static void
yy_stack_print (yybottom, yytop)
    yytype_int16 *yybottom;
    yytype_int16 *yytop;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)				\
do {								\
  if (yydebug)							\
    yy_stack_print ((Bottom), (Top));				\
} while (YYID (0))


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_reduce_print (YYSTYPE *yyvsp, YYLTYPE *yylsp, int yyrule)
#else
static void
yy_reduce_print (yyvsp, yylsp, yyrule)
    YYSTYPE *yyvsp;
    YYLTYPE *yylsp;
    int yyrule;
#endif
{
  int yynrhs = yyr2[yyrule];
  int yyi;
  unsigned long int yylno = yyrline[yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
	     yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr, yyrhs[yyprhs[yyrule] + yyi],
		       &(yyvsp[(yyi + 1) - (yynrhs)])
		       , &(yylsp[(yyi + 1) - (yynrhs)])		       );
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (yydebug)				\
    yy_reduce_print (yyvsp, yylsp, Rule); \
} while (YYID (0))

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef	YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif


#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static YYSIZE_T
yystrlen (const char *yystr)
#else
static YYSIZE_T
yystrlen (yystr)
    const char *yystr;
#endif
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static char *
yystpcpy (char *yydest, const char *yysrc)
#else
static char *
yystpcpy (yydest, yysrc)
    char *yydest;
    const char *yysrc;
#endif
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
	switch (*++yyp)
	  {
	  case '\'':
	  case ',':
	    goto do_not_strip_quotes;

	  case '\\':
	    if (*++yyp != '\\')
	      goto do_not_strip_quotes;
	    /* Fall through.  */
	  default:
	    if (yyres)
	      yyres[yyn] = *yyp;
	    yyn++;
	    break;

	  case '"':
	    if (yyres)
	      yyres[yyn] = '\0';
	    return yyn;
	  }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into *YYMSG, which is of size *YYMSG_ALLOC, an error message
   about the unexpected token YYTOKEN for the state stack whose top is
   YYSSP.

   Return 0 if *YYMSG was successfully written.  Return 1 if *YYMSG is
   not large enough to hold the message.  In that case, also set
   *YYMSG_ALLOC to the required number of bytes.  Return 2 if the
   required number of bytes is too large to store.  */
static int
yysyntax_error (YYSIZE_T *yymsg_alloc, char **yymsg,
                yytype_int16 *yyssp, int yytoken)
{
  YYSIZE_T yysize0 = yytnamerr (0, yytname[yytoken]);
  YYSIZE_T yysize = yysize0;
  YYSIZE_T yysize1;
  enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
  /* Internationalized format string. */
  const char *yyformat = 0;
  /* Arguments of yyformat. */
  char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
  /* Number of reported tokens (one for the "unexpected", one per
     "expected"). */
  int yycount = 0;

  /* There are many possibilities here to consider:
     - Assume YYFAIL is not used.  It's too flawed to consider.  See
       <http://lists.gnu.org/archive/html/bison-patches/2009-12/msg00024.html>
       for details.  YYERROR is fine as it does not invoke this
       function.
     - If this state is a consistent state with a default action, then
       the only way this function was invoked is if the default action
       is an error action.  In that case, don't check for expected
       tokens because there are none.
     - The only way there can be no lookahead present (in yychar) is if
       this state is a consistent state with a default action.  Thus,
       detecting the absence of a lookahead is sufficient to determine
       that there is no unexpected or expected token to report.  In that
       case, just report a simple "syntax error".
     - Don't assume there isn't a lookahead just because this state is a
       consistent state with a default action.  There might have been a
       previous inconsistent state, consistent state with a non-default
       action, or user semantic action that manipulated yychar.
     - Of course, the expected token list depends on states to have
       correct lookahead information, and it depends on the parser not
       to perform extra reductions after fetching a lookahead from the
       scanner and before detecting a syntax error.  Thus, state merging
       (from LALR or IELR) and default reductions corrupt the expected
       token list.  However, the list is correct for canonical LR with
       one exception: it will still contain any token that will not be
       accepted due to an error action in a later state.
  */
  if (yytoken != YYEMPTY)
    {
      int yyn = yypact[*yyssp];
      yyarg[yycount++] = yytname[yytoken];
      if (!yypact_value_is_default (yyn))
        {
          /* Start YYX at -YYN if negative to avoid negative indexes in
             YYCHECK.  In other words, skip the first -YYN actions for
             this state because they are default actions.  */
          int yyxbegin = yyn < 0 ? -yyn : 0;
          /* Stay within bounds of both yycheck and yytname.  */
          int yychecklim = YYLAST - yyn + 1;
          int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
          int yyx;

          for (yyx = yyxbegin; yyx < yyxend; ++yyx)
            if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR
                && !yytable_value_is_error (yytable[yyx + yyn]))
              {
                if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
                  {
                    yycount = 1;
                    yysize = yysize0;
                    break;
                  }
                yyarg[yycount++] = yytname[yyx];
                yysize1 = yysize + yytnamerr (0, yytname[yyx]);
                if (! (yysize <= yysize1
                       && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
                  return 2;
                yysize = yysize1;
              }
        }
    }

  switch (yycount)
    {
# define YYCASE_(N, S)                      \
      case N:                               \
        yyformat = S;                       \
      break
      YYCASE_(0, YY_("syntax error"));
      YYCASE_(1, YY_("syntax error, unexpected %s"));
      YYCASE_(2, YY_("syntax error, unexpected %s, expecting %s"));
      YYCASE_(3, YY_("syntax error, unexpected %s, expecting %s or %s"));
      YYCASE_(4, YY_("syntax error, unexpected %s, expecting %s or %s or %s"));
      YYCASE_(5, YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
# undef YYCASE_
    }

  yysize1 = yysize + yystrlen (yyformat);
  if (! (yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
    return 2;
  yysize = yysize1;

  if (*yymsg_alloc < yysize)
    {
      *yymsg_alloc = 2 * yysize;
      if (! (yysize <= *yymsg_alloc
             && *yymsg_alloc <= YYSTACK_ALLOC_MAXIMUM))
        *yymsg_alloc = YYSTACK_ALLOC_MAXIMUM;
      return 1;
    }

  /* Avoid sprintf, as that infringes on the user's name space.
     Don't have undefined behavior even if the translation
     produced a string with the wrong number of "%s"s.  */
  {
    char *yyp = *yymsg;
    int yyi = 0;
    while ((*yyp = *yyformat) != '\0')
      if (*yyp == '%' && yyformat[1] == 's' && yyi < yycount)
        {
          yyp += yytnamerr (yyp, yyarg[yyi++]);
          yyformat += 2;
        }
      else
        {
          yyp++;
          yyformat++;
        }
  }
  return 0;
}
#endif /* YYERROR_VERBOSE */

/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep, YYLTYPE *yylocationp)
#else
static void
yydestruct (yymsg, yytype, yyvaluep, yylocationp)
    const char *yymsg;
    int yytype;
    YYSTYPE *yyvaluep;
    YYLTYPE *yylocationp;
#endif
{
  YYUSE (yyvaluep);
  YYUSE (yylocationp);

  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  switch (yytype)
    {

      default:
	break;
    }
}


/* Prevent warnings from -Wmissing-prototypes.  */
#ifdef YYPARSE_PARAM
#if defined __STDC__ || defined __cplusplus
int yyparse (void *YYPARSE_PARAM);
#else
int yyparse ();
#endif
#else /* ! YYPARSE_PARAM */
#if defined __STDC__ || defined __cplusplus
int yyparse (void);
#else
int yyparse ();
#endif
#endif /* ! YYPARSE_PARAM */


/* The lookahead symbol.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;

/* Location data for the lookahead symbol.  */
YYLTYPE yylloc;

/* Number of syntax errors so far.  */
int yynerrs;


/*----------.
| yyparse.  |
`----------*/

#ifdef YYPARSE_PARAM
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void *YYPARSE_PARAM)
#else
int
yyparse (YYPARSE_PARAM)
    void *YYPARSE_PARAM;
#endif
#else /* ! YYPARSE_PARAM */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void)
#else
int
yyparse ()

#endif
#endif
{
    int yystate;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus;

    /* The stacks and their tools:
       `yyss': related to states.
       `yyvs': related to semantic values.
       `yyls': related to locations.

       Refer to the stacks thru separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* The state stack.  */
    yytype_int16 yyssa[YYINITDEPTH];
    yytype_int16 *yyss;
    yytype_int16 *yyssp;

    /* The semantic value stack.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs;
    YYSTYPE *yyvsp;

    /* The location stack.  */
    YYLTYPE yylsa[YYINITDEPTH];
    YYLTYPE *yyls;
    YYLTYPE *yylsp;

    /* The locations where the error started and ended.  */
    YYLTYPE yyerror_range[3];

    YYSIZE_T yystacksize;

  int yyn;
  int yyresult;
  /* Lookahead token as an internal (translated) token number.  */
  int yytoken;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;
  YYLTYPE yyloc;

#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N), yylsp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  yytoken = 0;
  yyss = yyssa;
  yyvs = yyvsa;
  yyls = yylsa;
  yystacksize = YYINITDEPTH;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY; /* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */
  yyssp = yyss;
  yyvsp = yyvs;
  yylsp = yyls;

#if defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL
  /* Initialize the default location before parsing starts.  */
  yylloc.first_line   = yylloc.last_line   = 1;
  yylloc.first_column = yylloc.last_column = 1;
#endif

  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
	/* Give user a chance to reallocate the stack.  Use copies of
	   these so that the &'s don't force the real ones into
	   memory.  */
	YYSTYPE *yyvs1 = yyvs;
	yytype_int16 *yyss1 = yyss;
	YYLTYPE *yyls1 = yyls;

	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if yyoverflow is a macro.  */
	yyoverflow (YY_("memory exhausted"),
		    &yyss1, yysize * sizeof (*yyssp),
		    &yyvs1, yysize * sizeof (*yyvsp),
		    &yyls1, yysize * sizeof (*yylsp),
		    &yystacksize);

	yyls = yyls1;
	yyss = yyss1;
	yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
	goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
	yystacksize = YYMAXDEPTH;

      {
	yytype_int16 *yyss1 = yyss;
	union yyalloc *yyptr =
	  (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
	if (! yyptr)
	  goto yyexhaustedlab;
	YYSTACK_RELOCATE (yyss_alloc, yyss);
	YYSTACK_RELOCATE (yyvs_alloc, yyvs);
	YYSTACK_RELOCATE (yyls_alloc, yyls);
#  undef YYSTACK_RELOCATE
	if (yyss1 != yyssa)
	  YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;
      yylsp = yyls + yysize - 1;

      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
		  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
	YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid lookahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = YYLEX;
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token.  */
  yychar = YYEMPTY;

  yystate = yyn;
  *++yyvsp = yylval;
  *++yylsp = yylloc;
  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     `$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];

  /* Default location.  */
  YYLLOC_DEFAULT (yyloc, (yylsp - yylen), yylen);
  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 2:

/* Line 1806 of yacc.c  */
#line 180 "IT.y"
    {(yylsp[(1) - (1)]); // this is needed to make bison set up
							     	     // the location variable yylloc
								  ProgramDef *program = new ProgramDef((yyvsp[(1) - (1)].nodeList));
								  program->Print(0);		
								}
    break;

  case 3:

/* Line 1806 of yacc.c  */
#line 185 "IT.y"
    { ((yyval.nodeList) = new List<Node*>)->Append((yyvsp[(1) - (1)].node)); }
    break;

  case 4:

/* Line 1806 of yacc.c  */
#line 186 "IT.y"
    { ((yyval.nodeList) = (yyvsp[(1) - (2)].nodeList))->Append((yyvsp[(2) - (2)].node)); }
    break;

  case 9:

/* Line 1806 of yacc.c  */
#line 190 "IT.y"
    { (yyval.node) = new TupleDef(new Identifier((yylsp[(2) - (4)]), (yyvsp[(2) - (4)].stringConstant)), (yyvsp[(4) - (4)].varList)); }
    break;

  case 10:

/* Line 1806 of yacc.c  */
#line 191 "IT.y"
    { ((yyval.varList) = new List<VariableDef*>)->AppendAll((yyvsp[(1) - (1)].varList)); }
    break;

  case 11:

/* Line 1806 of yacc.c  */
#line 192 "IT.y"
    { ((yyval.varList) = (yyvsp[(1) - (2)].varList))->AppendAll((yyvsp[(2) - (2)].varList)); }
    break;

  case 12:

/* Line 1806 of yacc.c  */
#line 193 "IT.y"
    { (yyval.varList) = VariableDef::DecomposeDefs((yyvsp[(1) - (3)].idList), (yyvsp[(3) - (3)].type)); }
    break;

  case 13:

/* Line 1806 of yacc.c  */
#line 194 "IT.y"
    { ((yyval.idList) = new List<Identifier*>)->Append(new Identifier((yylsp[(1) - (1)]), (yyvsp[(1) - (1)].stringConstant))); }
    break;

  case 14:

/* Line 1806 of yacc.c  */
#line 195 "IT.y"
    { ((yyval.idList) = (yyvsp[(1) - (3)].idList))->Append(new Identifier((yylsp[(3) - (3)]), (yyvsp[(3) - (3)].stringConstant))); }
    break;

  case 17:

/* Line 1806 of yacc.c  */
#line 197 "IT.y"
    { (yyval.type) = Type::intType; }
    break;

  case 18:

/* Line 1806 of yacc.c  */
#line 198 "IT.y"
    { (yyval.type) = Type::floatType; }
    break;

  case 19:

/* Line 1806 of yacc.c  */
#line 199 "IT.y"
    { (yyval.type) = Type::doubleType; }
    break;

  case 20:

/* Line 1806 of yacc.c  */
#line 200 "IT.y"
    { (yyval.type) = Type::charType; }
    break;

  case 21:

/* Line 1806 of yacc.c  */
#line 201 "IT.y"
    { (yyval.type) = Type::boolType; }
    break;

  case 22:

/* Line 1806 of yacc.c  */
#line 202 "IT.y"
    { (yyval.type) = Type::epochType; }
    break;

  case 23:

/* Line 1806 of yacc.c  */
#line 203 "IT.y"
    { (yyval.type) = Type::rangeType; }
    break;

  case 24:

/* Line 1806 of yacc.c  */
#line 204 "IT.y"
    { (yyval.type) = new NamedType(new Identifier((yylsp[(1) - (1)]), (yyvsp[(1) - (1)].stringConstant))); }
    break;

  case 25:

/* Line 1806 of yacc.c  */
#line 206 "IT.y"
    { StaticArrayType *sa = new StaticArrayType((yylsp[(1) - (4)]), (yyvsp[(4) - (4)].type), (yyvsp[(2) - (4)].intList)->NumElements());
								  sa->setLengths((yyvsp[(2) - (4)].intList)); 
								  (yyval.type) = sa; }
    break;

  case 26:

/* Line 1806 of yacc.c  */
#line 209 "IT.y"
    { ((yyval.intList) = (yyvsp[(1) - (4)].intList))->Append((yyvsp[(3) - (4)].intConstant)); }
    break;

  case 27:

/* Line 1806 of yacc.c  */
#line 210 "IT.y"
    { ((yyval.intList) = new List<int>)->Append((yyvsp[(2) - (3)].intConstant)); }
    break;

  case 28:

/* Line 1806 of yacc.c  */
#line 215 "IT.y"
    { (yyval.node) = new TaskDef(new Identifier((yylsp[(2) - (8)]), (yyvsp[(2) - (8)].stringConstant)), 
										new DefineSection((yyvsp[(4) - (8)].varList), (yylsp[(4) - (8)])), (yyvsp[(5) - (8)].envConfig), (yyvsp[(6) - (8)].initInstr), (yyvsp[(7) - (8)].compute), (yyvsp[(8) - (8)].partition)); }
    break;

  case 29:

/* Line 1806 of yacc.c  */
#line 220 "IT.y"
    { (yyval.varList) = (yyvsp[(3) - (3)].varList); }
    break;

  case 30:

/* Line 1806 of yacc.c  */
#line 221 "IT.y"
    { ((yyval.varList) = new List<VariableDef*>)->AppendAll((yyvsp[(1) - (1)].varList)); }
    break;

  case 31:

/* Line 1806 of yacc.c  */
#line 222 "IT.y"
    { ((yyval.varList) = (yyvsp[(1) - (2)].varList))->AppendAll((yyvsp[(2) - (2)].varList)); }
    break;

  case 32:

/* Line 1806 of yacc.c  */
#line 223 "IT.y"
    { (yyval.varList) = VariableDef::DecomposeDefs((yyvsp[(1) - (3)].idList), (yyvsp[(3) - (3)].type)); }
    break;

  case 37:

/* Line 1806 of yacc.c  */
#line 226 "IT.y"
    { (yyval.type) = new ListType((yylsp[(1) - (3)]), (yyvsp[(3) - (3)].type)); }
    break;

  case 38:

/* Line 1806 of yacc.c  */
#line 228 "IT.y"
    { (yyval.type) = new ArrayType((yylsp[(1) - (5)]), (yyvsp[(4) - (5)].type), (yyvsp[(1) - (5)].intConstant)); }
    break;

  case 41:

/* Line 1806 of yacc.c  */
#line 233 "IT.y"
    { (yyval.envConfig) = new EnvironmentConfig((yyvsp[(3) - (3)].envLinks), (yylsp[(1) - (3)])); }
    break;

  case 42:

/* Line 1806 of yacc.c  */
#line 234 "IT.y"
    { (yyval.envLinks) = (yyvsp[(1) - (1)].envLinks); }
    break;

  case 43:

/* Line 1806 of yacc.c  */
#line 235 "IT.y"
    { ((yyval.envLinks) = (yyvsp[(1) - (2)].envLinks))->AppendAll((yyvsp[(2) - (2)].envLinks)); }
    break;

  case 44:

/* Line 1806 of yacc.c  */
#line 236 "IT.y"
    { (yyval.envLinks) = EnvironmentLink::decomposeLinks((yyvsp[(1) - (3)].idList), (yyvsp[(3) - (3)].linkageType)); }
    break;

  case 45:

/* Line 1806 of yacc.c  */
#line 237 "IT.y"
    { (yyval.linkageType) = TypeLink; }
    break;

  case 46:

/* Line 1806 of yacc.c  */
#line 238 "IT.y"
    { (yyval.linkageType) = TypeCreate; }
    break;

  case 47:

/* Line 1806 of yacc.c  */
#line 239 "IT.y"
    { (yyval.linkageType) = TypeCreateIfNotLinked; }
    break;

  case 48:

/* Line 1806 of yacc.c  */
#line 243 "IT.y"
    { (yyval.initInstr) = NULL; }
    break;

  case 49:

/* Line 1806 of yacc.c  */
#line 244 "IT.y"
    { (yyval.initInstr) = new InitializeInstr((yyvsp[(2) - (4)].idList), (yyvsp[(4) - (4)].stmtList), (yylsp[(1) - (4)])); }
    break;

  case 50:

/* Line 1806 of yacc.c  */
#line 245 "IT.y"
    { (yyval.idList) = new List<Identifier*>; }
    break;

  case 51:

/* Line 1806 of yacc.c  */
#line 246 "IT.y"
    { (yyval.idList) = (yyvsp[(2) - (3)].idList); }
    break;

  case 52:

/* Line 1806 of yacc.c  */
#line 250 "IT.y"
    { (yyval.compute) = new ComputeSection((yyvsp[(3) - (3)].metaStageList), (yylsp[(1) - (3)])); }
    break;

  case 53:

/* Line 1806 of yacc.c  */
#line 251 "IT.y"
    { ((yyval.metaStageList) = new List<MetaComputeStage*>)->Append(new MetaComputeStage((yyvsp[(1) - (1)].stageList), NULL)); }
    break;

  case 54:

/* Line 1806 of yacc.c  */
#line 252 "IT.y"
    { ((yyval.metaStageList) = new List<MetaComputeStage*>)->Append((yyvsp[(1) - (1)].metaStage)); }
    break;

  case 55:

/* Line 1806 of yacc.c  */
#line 253 "IT.y"
    { ((yyval.metaStageList) = new List<MetaComputeStage*>)->Append((yyvsp[(1) - (2)].metaStage)); (yyval.metaStageList)->AppendAll((yyvsp[(2) - (2)].metaStageList)); }
    break;

  case 56:

/* Line 1806 of yacc.c  */
#line 254 "IT.y"
    { (yyval.metaStage) = new MetaComputeStage((yyvsp[(1) - (2)].stageList), (yyvsp[(2) - (2)].repeat)); }
    break;

  case 57:

/* Line 1806 of yacc.c  */
#line 255 "IT.y"
    { ((yyval.stageList) = new List<ComputeStage*>)->Append((yyvsp[(1) - (1)].stage)); }
    break;

  case 58:

/* Line 1806 of yacc.c  */
#line 256 "IT.y"
    { ((yyval.stageList) = (yyvsp[(1) - (2)].stageList))->Append((yyvsp[(2) - (2)].stage)); }
    break;

  case 59:

/* Line 1806 of yacc.c  */
#line 257 "IT.y"
    { (yyval.stage) = new ComputeStage((yyvsp[(1) - (4)].stageHeader), (yyvsp[(3) - (4)].metaStageList)); }
    break;

  case 60:

/* Line 1806 of yacc.c  */
#line 258 "IT.y"
    { (yyval.stage) = new ComputeStage((yyvsp[(1) - (4)].stageHeader), (yyvsp[(3) - (4)].stmtList)); }
    break;

  case 61:

/* Line 1806 of yacc.c  */
#line 260 "IT.y"
    { (yyval.stageHeader) = new StageHeader(new Identifier((yylsp[(1) - (6)]), (yyvsp[(1) - (6)].stringConstant)), (yyvsp[(4) - (6)].characterConstant), (yyvsp[(6) - (6)].expr)); }
    break;

  case 62:

/* Line 1806 of yacc.c  */
#line 261 "IT.y"
    { (yyval.expr) = NULL; }
    break;

  case 63:

/* Line 1806 of yacc.c  */
#line 262 "IT.y"
    { (yyval.expr) = new RangeExpr(new Identifier((yylsp[(3) - (5)]), (yyvsp[(3) - (5)].stringConstant)), (yyvsp[(5) - (5)].expr), NULL, (yylsp[(1) - (5)])); }
    break;

  case 64:

/* Line 1806 of yacc.c  */
#line 263 "IT.y"
    { (yyval.repeat) = new RepeatControl(new Identifier((yylsp[(4) - (5)]), (yyvsp[(4) - (5)].stringConstant)), (yyvsp[(5) - (5)].expr), (yylsp[(1) - (5)])); }
    break;

  case 65:

/* Line 1806 of yacc.c  */
#line 264 "IT.y"
    { (yyval.expr) = new RangeExpr(new Identifier((yylsp[(2) - (5)]), (yyvsp[(2) - (5)].stringConstant)), (yyvsp[(4) - (5)].expr), (yyvsp[(5) - (5)].expr), (yylsp[(1) - (5)])); }
    break;

  case 66:

/* Line 1806 of yacc.c  */
#line 265 "IT.y"
    { (yyval.expr) = (yyvsp[(2) - (2)].expr); }
    break;

  case 67:

/* Line 1806 of yacc.c  */
#line 266 "IT.y"
    { (yyval.expr) = new SubpartitionRangeExpr((yyvsp[(3) - (4)].characterConstant), (yylsp[(1) - (4)])); }
    break;

  case 68:

/* Line 1806 of yacc.c  */
#line 270 "IT.y"
    { (yyval.partition) = new PartitionSection((yyvsp[(2) - (4)].idList), (yyvsp[(4) - (4)].partSpecList), (yylsp[(1) - (4)])); }
    break;

  case 69:

/* Line 1806 of yacc.c  */
#line 271 "IT.y"
    { ((yyval.partSpecList) = new List<PartitionSpec*>)->Append((yyvsp[(1) - (1)].partSpec)); }
    break;

  case 70:

/* Line 1806 of yacc.c  */
#line 272 "IT.y"
    { ((yyval.partSpecList) = (yyvsp[(1) - (2)].partSpecList))->Append((yyvsp[(2) - (2)].partSpec)); }
    break;

  case 71:

/* Line 1806 of yacc.c  */
#line 274 "IT.y"
    { (yyval.partSpec) = new PartitionSpec((yyvsp[(2) - (11)].characterConstant), (yyvsp[(4) - (11)].intConstant), (yyvsp[(9) - (11)].dConSpecList), (yyvsp[(6) - (11)].booleanConstant), (yyvsp[(7) - (11)].sLink), (yyvsp[(10) - (11)].subPartSpec), (yylsp[(1) - (11)])); }
    break;

  case 72:

/* Line 1806 of yacc.c  */
#line 276 "IT.y"
    { (yyval.partSpec) = new PartitionSpec((yyvsp[(2) - (8)].characterConstant), (yyvsp[(7) - (8)].idList), (yylsp[(1) - (8)])); }
    break;

  case 73:

/* Line 1806 of yacc.c  */
#line 277 "IT.y"
    { (yyval.booleanConstant) = false; }
    break;

  case 74:

/* Line 1806 of yacc.c  */
#line 278 "IT.y"
    { (yyval.booleanConstant) = true; }
    break;

  case 75:

/* Line 1806 of yacc.c  */
#line 279 "IT.y"
    { (yyval.sLink) = NULL; }
    break;

  case 76:

/* Line 1806 of yacc.c  */
#line 280 "IT.y"
    { (yyval.sLink) = new SpaceLinkage(LinkTypePartition, (yyvsp[(3) - (4)].characterConstant), (yylsp[(1) - (4)])); }
    break;

  case 77:

/* Line 1806 of yacc.c  */
#line 281 "IT.y"
    { (yyval.sLink) = new SpaceLinkage(LinkTypeSubpartition, (yyvsp[(3) - (4)].characterConstant), (yylsp[(1) - (4)])); }
    break;

  case 78:

/* Line 1806 of yacc.c  */
#line 282 "IT.y"
    { ((yyval.dConSpecList) = new List<DataConfigurationSpec*>)->AppendAll((yyvsp[(1) - (1)].dConSpecList)); }
    break;

  case 79:

/* Line 1806 of yacc.c  */
#line 283 "IT.y"
    { ((yyval.dConSpecList) = (yyvsp[(1) - (2)].dConSpecList))->AppendAll((yyvsp[(2) - (2)].dConSpecList)); }
    break;

  case 80:

/* Line 1806 of yacc.c  */
#line 284 "IT.y"
    { (yyval.dConSpecList) = DataConfigurationSpec::decomposeDataConfig((yyvsp[(1) - (4)].vDimsList), (yyvsp[(3) - (4)].pInstrList), (yyvsp[(4) - (4)].sLink)); }
    break;

  case 81:

/* Line 1806 of yacc.c  */
#line 285 "IT.y"
    { ((yyval.vDimsList) = new List<VarDimensions*>)->Append((yyvsp[(1) - (1)].vDims)); }
    break;

  case 82:

/* Line 1806 of yacc.c  */
#line 286 "IT.y"
    { ((yyval.vDimsList) = (yyvsp[(1) - (3)].vDimsList))->Append((yyvsp[(3) - (3)].vDims)); }
    break;

  case 83:

/* Line 1806 of yacc.c  */
#line 287 "IT.y"
    { (yyval.vDims) = new VarDimensions(new Identifier((yylsp[(1) - (1)]), (yyvsp[(1) - (1)].stringConstant)), NULL); }
    break;

  case 84:

/* Line 1806 of yacc.c  */
#line 288 "IT.y"
    { (yyval.vDims) = new VarDimensions(new Identifier((yylsp[(1) - (4)]), (yyvsp[(1) - (4)].stringConstant)), (yyvsp[(3) - (4)].iConList)); }
    break;

  case 85:

/* Line 1806 of yacc.c  */
#line 289 "IT.y"
    { ((yyval.iConList) = new List<IntConstant*>)->Append(new IntConstant((yylsp[(1) - (1)]), (yyvsp[(1) - (1)].intConstant))); }
    break;

  case 86:

/* Line 1806 of yacc.c  */
#line 290 "IT.y"
    { ((yyval.iConList) = (yyvsp[(1) - (3)].iConList))->Append(new IntConstant((yylsp[(3) - (3)]), (yyvsp[(3) - (3)].intConstant))); }
    break;

  case 87:

/* Line 1806 of yacc.c  */
#line 291 "IT.y"
    { ((yyval.pInstrList) = new List<PartitionInstr*>)->Append((yyvsp[(1) - (1)].pInstr)); }
    break;

  case 88:

/* Line 1806 of yacc.c  */
#line 292 "IT.y"
    { ((yyval.pInstrList) = (yyvsp[(1) - (3)].pInstrList))->Append((yyvsp[(3) - (3)].pInstr)); }
    break;

  case 89:

/* Line 1806 of yacc.c  */
#line 293 "IT.y"
    { (yyval.pInstr) = new PartitionInstr((yylsp[(1) - (1)])); }
    break;

  case 90:

/* Line 1806 of yacc.c  */
#line 294 "IT.y"
    { (yyval.pInstr) = new PartitionInstr(
								    	new Identifier((yylsp[(1) - (4)]), (yyvsp[(1) - (4)].stringConstant)), (yyvsp[(3) - (4)].pArgList), false, NULL, (yylsp[(1) - (4)])); }
    break;

  case 91:

/* Line 1806 of yacc.c  */
#line 297 "IT.y"
    { (yyval.pInstr) = new PartitionInstr(new Identifier((yylsp[(1) - (8)]), (yyvsp[(1) - (8)].stringConstant)), (yyvsp[(3) - (8)].pArgList), true, (yyvsp[(7) - (8)].pArgList), (yylsp[(1) - (8)])); }
    break;

  case 92:

/* Line 1806 of yacc.c  */
#line 298 "IT.y"
    { (yyval.pArgList) = new List<PartitionArg*>; }
    break;

  case 93:

/* Line 1806 of yacc.c  */
#line 299 "IT.y"
    { ((yyval.pArgList) = new List<PartitionArg*>)->Append((yyvsp[(1) - (1)].pArg)); }
    break;

  case 94:

/* Line 1806 of yacc.c  */
#line 300 "IT.y"
    { ((yyval.pArgList) = (yyvsp[(1) - (3)].pArgList))->Append((yyvsp[(3) - (3)].pArg)); }
    break;

  case 95:

/* Line 1806 of yacc.c  */
#line 301 "IT.y"
    { (yyval.pArg) = new PartitionArg(new Identifier((yylsp[(1) - (1)]), (yyvsp[(1) - (1)].stringConstant))); }
    break;

  case 96:

/* Line 1806 of yacc.c  */
#line 302 "IT.y"
    { (yyval.pArg) = new PartitionArg(new IntConstant((yylsp[(1) - (1)]), (yyvsp[(1) - (1)].intConstant))); }
    break;

  case 97:

/* Line 1806 of yacc.c  */
#line 303 "IT.y"
    { (yyval.sLink) = NULL; }
    break;

  case 98:

/* Line 1806 of yacc.c  */
#line 304 "IT.y"
    { (yyval.sLink) = new SpaceLinkage(LinkTypeUndefined, (yyvsp[(4) - (4)].characterConstant), (yylsp[(2) - (4)])); }
    break;

  case 99:

/* Line 1806 of yacc.c  */
#line 305 "IT.y"
    { (yyval.subPartSpec) = NULL; }
    break;

  case 100:

/* Line 1806 of yacc.c  */
#line 307 "IT.y"
    { (yyval.subPartSpec) = new SubpartitionSpec((yyvsp[(3) - (10)].intConstant), (yyvsp[(6) - (10)].booleanConstant), (yyvsp[(9) - (10)].dConSpecList), (yylsp[(1) - (10)])); }
    break;

  case 101:

/* Line 1806 of yacc.c  */
#line 308 "IT.y"
    { (yyval.booleanConstant) = true; }
    break;

  case 102:

/* Line 1806 of yacc.c  */
#line 309 "IT.y"
    { (yyval.booleanConstant) = false; }
    break;

  case 103:

/* Line 1806 of yacc.c  */
#line 310 "IT.y"
    { ((yyval.dConSpecList) = new List<DataConfigurationSpec*>)->AppendAll((yyvsp[(1) - (1)].dConSpecList)); }
    break;

  case 104:

/* Line 1806 of yacc.c  */
#line 311 "IT.y"
    { ((yyval.dConSpecList) = (yyvsp[(1) - (2)].dConSpecList))->AppendAll((yyvsp[(2) - (2)].dConSpecList)); }
    break;

  case 105:

/* Line 1806 of yacc.c  */
#line 312 "IT.y"
    { (yyval.dConSpecList) = DataConfigurationSpec::decomposeDataConfig((yyvsp[(1) - (3)].vDimsList), (yyvsp[(3) - (3)].pInstrList), NULL);}
    break;

  case 106:

/* Line 1806 of yacc.c  */
#line 313 "IT.y"
    { ((yyval.pInstrList) = new List<PartitionInstr*>)->Append((yyvsp[(1) - (1)].pInstr)); }
    break;

  case 107:

/* Line 1806 of yacc.c  */
#line 314 "IT.y"
    { ((yyval.pInstrList) = (yyvsp[(1) - (3)].pInstrList))->Append((yyvsp[(3) - (3)].pInstr)); }
    break;

  case 108:

/* Line 1806 of yacc.c  */
#line 315 "IT.y"
    { (yyval.pInstr) = (yyvsp[(1) - (2)].pInstr); (yyval.pInstr)->SetOrder((yyvsp[(2) - (2)].order)); }
    break;

  case 109:

/* Line 1806 of yacc.c  */
#line 316 "IT.y"
    { (yyval.order) = RandomOrder; }
    break;

  case 110:

/* Line 1806 of yacc.c  */
#line 317 "IT.y"
    { (yyval.order) = AscendingOrder; }
    break;

  case 111:

/* Line 1806 of yacc.c  */
#line 318 "IT.y"
    { (yyval.order) = DescendingOrder; }
    break;

  case 112:

/* Line 1806 of yacc.c  */
#line 322 "IT.y"
    {BeginCode();}
    break;

  case 113:

/* Line 1806 of yacc.c  */
#line 322 "IT.y"
    { EndCode(); (yyval.stmtList) = (yyvsp[(2) - (2)].stmtList); }
    break;

  case 114:

/* Line 1806 of yacc.c  */
#line 323 "IT.y"
    { ((yyval.stmtList) = new List<Stmt*>)->Append((yyvsp[(1) - (1)].stmt)); }
    break;

  case 115:

/* Line 1806 of yacc.c  */
#line 324 "IT.y"
    { ((yyval.stmtList) = new List<Stmt*>)->Append((yyvsp[(1) - (2)].stmt)); }
    break;

  case 116:

/* Line 1806 of yacc.c  */
#line 325 "IT.y"
    { ((yyval.stmtList) = new List<Stmt*>)->Append((yyvsp[(1) - (3)].stmt)); (yyval.stmtList)->AppendAll((yyvsp[(3) - (3)].stmtList)); }
    break;

  case 122:

/* Line 1806 of yacc.c  */
#line 330 "IT.y"
    { (yyval.stmt) = (yyvsp[(1) - (1)].expr); }
    break;

  case 123:

/* Line 1806 of yacc.c  */
#line 332 "IT.y"
    { (yyval.stmt) = new SLoopStmt((yyvsp[(8) - (11)].id), (yyvsp[(10) - (11)].expr), (yyvsp[(11) - (11)].expr), new StmtBlock((yyvsp[(5) - (11)].stmtList)), (yylsp[(1) - (11)])); }
    break;

  case 124:

/* Line 1806 of yacc.c  */
#line 333 "IT.y"
    { (yyval.stmt) = new PLoopStmt((yyvsp[(6) - (6)].rangeCondList), new StmtBlock((yyvsp[(3) - (6)].stmtList)), (yylsp[(1) - (6)])); }
    break;

  case 125:

/* Line 1806 of yacc.c  */
#line 334 "IT.y"
    { (yyval.stmt) = new WhileStmt((yyvsp[(6) - (6)].expr), new StmtBlock((yyvsp[(3) - (6)].stmtList)), (yylsp[(1) - (6)])); }
    break;

  case 126:

/* Line 1806 of yacc.c  */
#line 335 "IT.y"
    { ((yyval.rangeCondList) = new List<IndexRangeCondition*>)->Append((yyvsp[(1) - (1)].rangeCond)); }
    break;

  case 127:

/* Line 1806 of yacc.c  */
#line 336 "IT.y"
    { ((yyval.rangeCondList) = (yyvsp[(1) - (3)].rangeCondList))->Append((yyvsp[(3) - (3)].rangeCond)); }
    break;

  case 128:

/* Line 1806 of yacc.c  */
#line 337 "IT.y"
    { (yyval.rangeCond) = new IndexRangeCondition((yyvsp[(1) - (4)].idList), new Identifier((yylsp[(3) - (4)]), (yyvsp[(3) - (4)].stringConstant)), (yyvsp[(4) - (4)].expr), Join((yylsp[(1) - (4)]), (yylsp[(4) - (4)]))); }
    break;

  case 129:

/* Line 1806 of yacc.c  */
#line 338 "IT.y"
    { (yyval.expr) = NULL; }
    break;

  case 130:

/* Line 1806 of yacc.c  */
#line 339 "IT.y"
    { (yyval.expr) = (yyvsp[(2) - (2)].expr); }
    break;

  case 131:

/* Line 1806 of yacc.c  */
#line 340 "IT.y"
    { List<ConditionalStmt*> *lca = new List<ConditionalStmt*>;
								  lca->Append(new ConditionalStmt((yyvsp[(3) - (8)].expr), new StmtBlock((yyvsp[(6) - (8)].stmtList)), (yylsp[(1) - (8)])));
								  lca->AppendAll((yyvsp[(8) - (8)].condStmtList));	
								  (yyval.stmt) = new IfStmt(lca, (yylsp[(1) - (8)])); }
    break;

  case 132:

/* Line 1806 of yacc.c  */
#line 344 "IT.y"
    { (yyval.condStmtList) = new List<ConditionalStmt*>; }
    break;

  case 133:

/* Line 1806 of yacc.c  */
#line 345 "IT.y"
    { ((yyval.condStmtList) = new List<ConditionalStmt*>)->Append((yyvsp[(1) - (1)].condStmt)); }
    break;

  case 134:

/* Line 1806 of yacc.c  */
#line 346 "IT.y"
    { ((yyval.condStmtList) = new List<ConditionalStmt*>)->Append((yyvsp[(1) - (2)].condStmt)); (yyval.condStmtList)->AppendAll((yyvsp[(2) - (2)].condStmtList)); }
    break;

  case 135:

/* Line 1806 of yacc.c  */
#line 347 "IT.y"
    { (yyval.condStmt) = new ConditionalStmt(NULL, new StmtBlock((yyvsp[(3) - (4)].stmtList)), (yylsp[(1) - (4)])); }
    break;

  case 136:

/* Line 1806 of yacc.c  */
#line 348 "IT.y"
    { (yyval.condStmt) = new ConditionalStmt((yyvsp[(4) - (8)].expr), new StmtBlock((yyvsp[(7) - (8)].stmtList)), (yylsp[(1) - (8)])); }
    break;

  case 137:

/* Line 1806 of yacc.c  */
#line 350 "IT.y"
    { (yyval.expr) = NULL; }
    break;

  case 138:

/* Line 1806 of yacc.c  */
#line 351 "IT.y"
    { (yyval.expr) = (yyvsp[(2) - (2)].expr); }
    break;

  case 139:

/* Line 1806 of yacc.c  */
#line 352 "IT.y"
    { (yyval.expr) = new ArithmaticExpr((yyvsp[(1) - (3)].expr), ADD, (yyvsp[(3) - (3)].expr), (yylsp[(2) - (3)])); }
    break;

  case 140:

/* Line 1806 of yacc.c  */
#line 353 "IT.y"
    { (yyval.expr) = new ArithmaticExpr((yyvsp[(1) - (3)].expr), SUBTRACT, (yyvsp[(3) - (3)].expr), (yylsp[(2) - (3)])); }
    break;

  case 141:

/* Line 1806 of yacc.c  */
#line 354 "IT.y"
    { (yyval.expr) = new ArithmaticExpr((yyvsp[(1) - (3)].expr), MULTIPLY, (yyvsp[(3) - (3)].expr), (yylsp[(2) - (3)])); }
    break;

  case 142:

/* Line 1806 of yacc.c  */
#line 355 "IT.y"
    { (yyval.expr) = new ArithmaticExpr((yyvsp[(1) - (3)].expr), DIVIDE, (yyvsp[(3) - (3)].expr), (yylsp[(2) - (3)])); }
    break;

  case 143:

/* Line 1806 of yacc.c  */
#line 356 "IT.y"
    { (yyval.expr) = new ArithmaticExpr((yyvsp[(1) - (3)].expr), MODULUS, (yyvsp[(3) - (3)].expr), (yylsp[(2) - (3)])); }
    break;

  case 144:

/* Line 1806 of yacc.c  */
#line 357 "IT.y"
    { (yyval.expr) = new ArithmaticExpr((yyvsp[(1) - (3)].expr), LEFT_SHIFT, (yyvsp[(3) - (3)].expr), (yylsp[(2) - (3)])); }
    break;

  case 145:

/* Line 1806 of yacc.c  */
#line 358 "IT.y"
    { (yyval.expr) = new ArithmaticExpr((yyvsp[(1) - (3)].expr), RIGHT_SHIFT, (yyvsp[(3) - (3)].expr), (yylsp[(2) - (3)])); }
    break;

  case 146:

/* Line 1806 of yacc.c  */
#line 359 "IT.y"
    { (yyval.expr) = new ArithmaticExpr((yyvsp[(1) - (3)].expr), POWER, (yyvsp[(3) - (3)].expr), (yylsp[(2) - (3)])); }
    break;

  case 147:

/* Line 1806 of yacc.c  */
#line 360 "IT.y"
    { (yyval.expr) = new LogicalExpr((yyvsp[(1) - (3)].expr), LT, (yyvsp[(3) - (3)].expr), (yylsp[(2) - (3)])); }
    break;

  case 148:

/* Line 1806 of yacc.c  */
#line 361 "IT.y"
    { (yyval.expr) = new LogicalExpr((yyvsp[(1) - (3)].expr), GT, (yyvsp[(3) - (3)].expr), (yylsp[(2) - (3)])); }
    break;

  case 149:

/* Line 1806 of yacc.c  */
#line 362 "IT.y"
    { (yyval.expr) = new LogicalExpr((yyvsp[(1) - (3)].expr), OR, (yyvsp[(3) - (3)].expr), (yylsp[(2) - (3)])); }
    break;

  case 150:

/* Line 1806 of yacc.c  */
#line 363 "IT.y"
    { (yyval.expr) = new LogicalExpr((yyvsp[(1) - (3)].expr), AND, (yyvsp[(3) - (3)].expr), (yylsp[(2) - (3)])); }
    break;

  case 151:

/* Line 1806 of yacc.c  */
#line 364 "IT.y"
    { (yyval.expr) = new LogicalExpr((yyvsp[(1) - (3)].expr), EQ, (yyvsp[(3) - (3)].expr), (yylsp[(2) - (3)])); }
    break;

  case 152:

/* Line 1806 of yacc.c  */
#line 365 "IT.y"
    { (yyval.expr) = new LogicalExpr((yyvsp[(1) - (3)].expr), NE, (yyvsp[(3) - (3)].expr), (yylsp[(2) - (3)])); }
    break;

  case 153:

/* Line 1806 of yacc.c  */
#line 366 "IT.y"
    { (yyval.expr) = new LogicalExpr(NULL, NOT, (yyvsp[(2) - (2)].expr), (yylsp[(1) - (2)])); }
    break;

  case 154:

/* Line 1806 of yacc.c  */
#line 367 "IT.y"
    { (yyval.expr) = new LogicalExpr((yyvsp[(1) - (3)].expr), GTE, (yyvsp[(3) - (3)].expr), (yylsp[(2) - (3)])); }
    break;

  case 155:

/* Line 1806 of yacc.c  */
#line 368 "IT.y"
    { (yyval.expr) = new LogicalExpr((yyvsp[(1) - (3)].expr), LTE, (yyvsp[(3) - (3)].expr), (yylsp[(2) - (3)])); }
    break;

  case 156:

/* Line 1806 of yacc.c  */
#line 369 "IT.y"
    { (yyval.expr) = new AssignmentExpr((yyvsp[(1) - (3)].expr), (yyvsp[(3) - (3)].expr), (yylsp[(2) - (3)])); }
    break;

  case 157:

/* Line 1806 of yacc.c  */
#line 370 "IT.y"
    { (yyval.expr) = new ReductionExpr((yyvsp[(1) - (3)].expr), SUM, (yyvsp[(3) - (3)].expr), (yylsp[(2) - (3)])); }
    break;

  case 158:

/* Line 1806 of yacc.c  */
#line 371 "IT.y"
    { (yyval.expr) = new ReductionExpr((yyvsp[(1) - (3)].expr), PRODUCT, (yyvsp[(3) - (3)].expr), (yylsp[(2) - (3)])); }
    break;

  case 159:

/* Line 1806 of yacc.c  */
#line 372 "IT.y"
    { (yyval.expr) = new ReductionExpr((yyvsp[(1) - (3)].expr), MAX, (yyvsp[(3) - (3)].expr), (yylsp[(2) - (3)])); }
    break;

  case 160:

/* Line 1806 of yacc.c  */
#line 373 "IT.y"
    { (yyval.expr) = new ReductionExpr((yyvsp[(1) - (3)].expr), MIN, (yyvsp[(3) - (3)].expr), (yylsp[(2) - (3)])); }
    break;

  case 161:

/* Line 1806 of yacc.c  */
#line 374 "IT.y"
    { (yyval.expr) = new ReductionExpr((yyvsp[(1) - (3)].expr), AVG, (yyvsp[(3) - (3)].expr), (yylsp[(2) - (3)])); }
    break;

  case 162:

/* Line 1806 of yacc.c  */
#line 375 "IT.y"
    { (yyval.expr) = new ReductionExpr((yyvsp[(1) - (3)].expr), MAX_ENTRY, (yyvsp[(3) - (3)].expr), (yylsp[(2) - (3)])); }
    break;

  case 163:

/* Line 1806 of yacc.c  */
#line 376 "IT.y"
    { (yyval.expr) = new ReductionExpr((yyvsp[(1) - (3)].expr), MIN_ENTRY, (yyvsp[(3) - (3)].expr), (yylsp[(2) - (3)])); }
    break;

  case 169:

/* Line 1806 of yacc.c  */
#line 382 "IT.y"
    { (yyval.expr) = new EpochExpr((yyvsp[(1) - (5)].expr), (yyvsp[(4) - (5)].epoch)); }
    break;

  case 170:

/* Line 1806 of yacc.c  */
#line 383 "IT.y"
    { (yyval.expr) = (yyvsp[(2) - (3)].expr); }
    break;

  case 171:

/* Line 1806 of yacc.c  */
#line 384 "IT.y"
    { (yyval.expr) = new IntConstant((yylsp[(1) - (1)]), (yyvsp[(1) - (1)].intConstant)); }
    break;

  case 172:

/* Line 1806 of yacc.c  */
#line 385 "IT.y"
    { (yyval.expr) = new DoubleConstant((yylsp[(1) - (1)]), (yyvsp[(1) - (1)].doubleConstant)); }
    break;

  case 173:

/* Line 1806 of yacc.c  */
#line 386 "IT.y"
    { (yyval.expr) = new BoolConstant((yylsp[(1) - (1)]), (yyvsp[(1) - (1)].booleanConstant)); }
    break;

  case 174:

/* Line 1806 of yacc.c  */
#line 387 "IT.y"
    { (yyval.expr) = new CharacterConstant((yylsp[(1) - (1)]), (yyvsp[(1) - (1)].characterConstant)); }
    break;

  case 175:

/* Line 1806 of yacc.c  */
#line 388 "IT.y"
    { (yyval.expr) = new FieldAccess(NULL, (yyvsp[(1) - (1)].id), (yylsp[(1) - (1)])); }
    break;

  case 176:

/* Line 1806 of yacc.c  */
#line 389 "IT.y"
    { (yyval.expr) = new FieldAccess((yyvsp[(1) - (3)].expr), (yyvsp[(3) - (3)].id), (yylsp[(2) - (3)])); }
    break;

  case 177:

/* Line 1806 of yacc.c  */
#line 390 "IT.y"
    { (yyval.expr) = new ArrayAccess((yyvsp[(1) - (4)].expr), (yyvsp[(3) - (4)].expr), Join((yylsp[(2) - (4)]), (yylsp[(4) - (4)]))); }
    break;

  case 179:

/* Line 1806 of yacc.c  */
#line 392 "IT.y"
    { (yyval.expr) = new SubRangeExpr((yyvsp[(1) - (3)].expr), (yyvsp[(3) - (3)].expr), Join((yylsp[(1) - (3)]), (yylsp[(3) - (3)]))); }
    break;

  case 180:

/* Line 1806 of yacc.c  */
#line 393 "IT.y"
    { (yyval.expr) = new SubRangeExpr(NULL, NULL, (yylsp[(1) - (1)])); }
    break;

  case 181:

/* Line 1806 of yacc.c  */
#line 394 "IT.y"
    { (yyval.expr) = new FunctionCall(new Identifier((yylsp[(1) - (4)]), (yyvsp[(1) - (4)].stringConstant)), (yyvsp[(3) - (4)].exprList), Join((yylsp[(1) - (4)]), (yylsp[(4) - (4)]))); }
    break;

  case 182:

/* Line 1806 of yacc.c  */
#line 395 "IT.y"
    { (yyval.exprList) = new List<Expr*>;}
    break;

  case 183:

/* Line 1806 of yacc.c  */
#line 396 "IT.y"
    { ((yyval.exprList) = new List<Expr*>)->Append(new StringConstant((yylsp[(1) - (1)]), (yyvsp[(1) - (1)].stringConstant))); }
    break;

  case 184:

/* Line 1806 of yacc.c  */
#line 397 "IT.y"
    { ((yyval.exprList) = new List<Expr*>)->Append((yyvsp[(1) - (1)].expr)); }
    break;

  case 185:

/* Line 1806 of yacc.c  */
#line 398 "IT.y"
    { ((yyval.exprList) = (yyvsp[(1) - (3)].exprList))->Append((yyvsp[(3) - (3)].expr)); }
    break;

  case 186:

/* Line 1806 of yacc.c  */
#line 399 "IT.y"
    { (yyval.epoch) = new EpochValue(new Identifier((yylsp[(1) - (1)]), (yyvsp[(1) - (1)].stringConstant)), 0); }
    break;

  case 187:

/* Line 1806 of yacc.c  */
#line 400 "IT.y"
    { (yyval.epoch) = new EpochValue(new Identifier((yylsp[(1) - (3)]), (yyvsp[(1) - (3)].stringConstant)), (yyvsp[(3) - (3)].intConstant)); }
    break;

  case 188:

/* Line 1806 of yacc.c  */
#line 401 "IT.y"
    { (yyval.id) = new Identifier((yylsp[(1) - (1)]), (yyvsp[(1) - (1)].stringConstant)); }
    break;

  case 189:

/* Line 1806 of yacc.c  */
#line 402 "IT.y"
    { (yyval.id) = new DimensionIdentifier((yylsp[(1) - (1)]), (yyvsp[(1) - (1)].intConstant)); }
    break;

  case 190:

/* Line 1806 of yacc.c  */
#line 403 "IT.y"
    { (yyval.id) = new Identifier((yylsp[(1) - (1)]), Identifier::RangeId); }
    break;

  case 191:

/* Line 1806 of yacc.c  */
#line 404 "IT.y"
    { (yyval.id) = new Identifier((yylsp[(1) - (1)]), Identifier::LocalId); }
    break;

  case 192:

/* Line 1806 of yacc.c  */
#line 405 "IT.y"
    { (yyval.id) = new Identifier((yylsp[(1) - (1)]), Identifier::IndexId); }
    break;

  case 193:

/* Line 1806 of yacc.c  */
#line 409 "IT.y"
    { (yyval.node) = new CoordinatorDef(new Identifier((yylsp[(3) - (6)]), (yyvsp[(3) - (6)].stringConstant)), (yyvsp[(6) - (6)].stmtList), (yylsp[(1) - (6)])); }
    break;

  case 194:

/* Line 1806 of yacc.c  */
#line 410 "IT.y"
    {BeginProgram();}
    break;

  case 195:

/* Line 1806 of yacc.c  */
#line 410 "IT.y"
    { EndProgram(); (yyval.stmtList) = (yyvsp[(2) - (2)].stmtList); }
    break;

  case 196:

/* Line 1806 of yacc.c  */
#line 411 "IT.y"
    { (yyval.expr) = new ObjectCreate((yyvsp[(2) - (2)].type), new List<Expr*>, (yylsp[(1) - (2)])); }
    break;

  case 197:

/* Line 1806 of yacc.c  */
#line 412 "IT.y"
    { (yyval.expr) = new ObjectCreate((yyvsp[(2) - (5)].type), (yyvsp[(4) - (5)].exprList), (yylsp[(1) - (5)])); }
    break;

  case 198:

/* Line 1806 of yacc.c  */
#line 413 "IT.y"
    { (yyval.expr) = new TaskInvocation(new Identifier((yylsp[(3) - (7)]), (yyvsp[(3) - (7)].stringConstant)), (yyvsp[(5) - (7)].id), (yyvsp[(6) - (7)].invokeArgsList), (yylsp[(1) - (7)])); }
    break;

  case 199:

/* Line 1806 of yacc.c  */
#line 414 "IT.y"
    { ((yyval.invokeArgsList) = new List<OptionalInvocationParams*>); }
    break;

  case 200:

/* Line 1806 of yacc.c  */
#line 415 "IT.y"
    { ((yyval.invokeArgsList) = new List<OptionalInvocationParams*>)->Append((yyvsp[(2) - (2)].invokeArgs)); }
    break;

  case 201:

/* Line 1806 of yacc.c  */
#line 416 "IT.y"
    { (yyval.invokeArgsList) = new List<OptionalInvocationParams*>;
								  (yyval.invokeArgsList)->Append((yyvsp[(2) - (4)].invokeArgs)); (yyval.invokeArgsList)->Append((yyvsp[(4) - (4)].invokeArgs));}
    break;

  case 202:

/* Line 1806 of yacc.c  */
#line 418 "IT.y"
    { (yyval.invokeArgs) = new OptionalInvocationParams((yyvsp[(1) - (3)].id), (yyvsp[(3) - (3)].exprList), Join((yylsp[(1) - (3)]), (yylsp[(3) - (3)]))); }
    break;

  case 203:

/* Line 1806 of yacc.c  */
#line 419 "IT.y"
    { (yyval.id) = new Identifier((yylsp[(1) - (1)]), OptionalInvocationParams::InitializeSection); }
    break;

  case 204:

/* Line 1806 of yacc.c  */
#line 420 "IT.y"
    { (yyval.id) = new Identifier((yylsp[(1) - (1)]), OptionalInvocationParams::PartitionSection); }
    break;

  case 205:

/* Line 1806 of yacc.c  */
#line 425 "IT.y"
    { (yyval.node) = new FunctionDef(new Identifier((yylsp[(2) - (5)]), (yyvsp[(2) - (5)].stringConstant)), (yyvsp[(4) - (5)].fnHeader), (yyvsp[(5) - (5)].stmtList)); }
    break;

  case 206:

/* Line 1806 of yacc.c  */
#line 426 "IT.y"
    { (yyval.fnHeader) = new FunctionHeader(new List<VariableDef*>, (yyvsp[(1) - (1)].varList)); }
    break;

  case 207:

/* Line 1806 of yacc.c  */
#line 427 "IT.y"
    { (yyval.fnHeader) = new FunctionHeader((yyvsp[(1) - (2)].varList), (yyvsp[(2) - (2)].varList)); }
    break;

  case 208:

/* Line 1806 of yacc.c  */
#line 428 "IT.y"
    { (yyval.varList) = (yyvsp[(3) - (3)].varList); }
    break;

  case 209:

/* Line 1806 of yacc.c  */
#line 429 "IT.y"
    { (yyval.varList) = (yyvsp[(3) - (3)].varList); }
    break;

  case 210:

/* Line 1806 of yacc.c  */
#line 430 "IT.y"
    { (yyval.stmtList) = (yyvsp[(3) - (3)].stmtList); }
    break;



/* Line 1806 of yacc.c  */
#line 3437 "y.tab.c"
      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;
  *++yylsp = yyloc;

  /* Now `shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*------------------------------------.
| yyerrlab -- here on detecting error |
`------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYEMPTY : YYTRANSLATE (yychar);

  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
# define YYSYNTAX_ERROR yysyntax_error (&yymsg_alloc, &yymsg, \
                                        yyssp, yytoken)
      {
        char const *yymsgp = YY_("syntax error");
        int yysyntax_error_status;
        yysyntax_error_status = YYSYNTAX_ERROR;
        if (yysyntax_error_status == 0)
          yymsgp = yymsg;
        else if (yysyntax_error_status == 1)
          {
            if (yymsg != yymsgbuf)
              YYSTACK_FREE (yymsg);
            yymsg = (char *) YYSTACK_ALLOC (yymsg_alloc);
            if (!yymsg)
              {
                yymsg = yymsgbuf;
                yymsg_alloc = sizeof yymsgbuf;
                yysyntax_error_status = 2;
              }
            else
              {
                yysyntax_error_status = YYSYNTAX_ERROR;
                yymsgp = yymsg;
              }
          }
        yyerror (yymsgp);
        if (yysyntax_error_status == 2)
          goto yyexhaustedlab;
      }
# undef YYSYNTAX_ERROR
#endif
    }

  yyerror_range[1] = yylloc;

  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
	 error, discard it.  */

      if (yychar <= YYEOF)
	{
	  /* Return failure if at end of input.  */
	  if (yychar == YYEOF)
	    YYABORT;
	}
      else
	{
	  yydestruct ("Error: discarding",
		      yytoken, &yylval, &yylloc);
	  yychar = YYEMPTY;
	}
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  yyerror_range[1] = yylsp[1-yylen];
  /* Do not reclaim the symbols of the rule which action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;	/* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
	{
	  yyn += YYTERROR;
	  if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
	    {
	      yyn = yytable[yyn];
	      if (0 < yyn)
		break;
	    }
	}

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
	YYABORT;

      yyerror_range[1] = *yylsp;
      yydestruct ("Error: popping",
		  yystos[yystate], yyvsp, yylsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  *++yyvsp = yylval;

  yyerror_range[2] = yylloc;
  /* Using YYLLOC is tempting, but would change the location of
     the lookahead.  YYLOC is available though.  */
  YYLLOC_DEFAULT (yyloc, yyerror_range, 2);
  *++yylsp = yyloc;

  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#if !defined(yyoverflow) || YYERROR_VERBOSE
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval, &yylloc);
    }
  /* Do not reclaim the symbols of the rule which action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
		  yystos[*yyssp], yyvsp, yylsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  /* Make sure YYID is used.  */
  return YYID (yyresult);
}



/* Line 2067 of yacc.c  */
#line 433 "IT.y"


/*----------------------------------------------------- Epilogue --------------------------------------------------------------*/

/* The closing %% above marks the end of the Rules section and the beginning
 * of the User Subroutines section. All text from here to the end of the
 * file is copied verbatim to the end of the generated y.tab.c file.
 * This section is where you put definitions of helper functions.
 */

/* Function: InitParser
 * --------------------
 * This function will be called before any calls to yyparse().  It is designed
 * to give you an opportunity to do anything that must be done to initialize
 * the parser (set global variables, configure starting state, etc.). One
 * thing it already does for you is assign the value of the global variable
 * yydebug that controls whether yacc prints debugging information about
 * parser actions (shift/reduce) and contents of state stack during parser.
 * If set to false, no information is printed. Setting it to true will give
 * you a running trail that might be helpful when debugging your parser.
 */
void InitParser() {
	PrintDebug("parser", "Initializing parser");
   	yydebug = false;
}


