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
     Dimensionality = 280,
     Dimension_No = 281,
     Single = 282,
     Double = 283,
     Format = 284,
     Real = 285,
     Integer = 286,
     Boolean = 287,
     Character = 288,
     String = 289,
     Type_Name = 290,
     Variable_Name = 291,
     Space = 292,
     Space_ID = 293,
     V_Dimension = 294,
     Activate = 295,
     For = 296,
     If = 297,
     Repeat = 298,
     Else = 299,
     From = 300,
     In = 301,
     Step = 302,
     Foreach = 303,
     Range = 304,
     Local = 305,
     Index = 306,
     C_Sub_Partition = 307,
     While = 308,
     Do = 309,
     Sequence = 310,
     To = 311,
     Of = 312,
     Link = 313,
     Create = 314,
     Link_or_Create = 315,
     Dynamic = 316,
     P_Sub_Partition = 317,
     Ordered = 318,
     Unordered = 319,
     Replicated = 320,
     Padding = 321,
     Relative_To = 322,
     Divides = 323,
     Sub_Partitions = 324,
     Partitions = 325,
     Unpartitioned = 326,
     Ascends = 327,
     Descends = 328,
     New_Line = 329,
     R_MIN_ENTRY = 330,
     R_MAX_ENTRY = 331,
     R_MIN = 332,
     R_MAX = 333,
     R_MULT = 334,
     R_ADD = 335,
     O_OR = 336,
     O_AND = 337,
     O_NE = 338,
     O_EQ = 339,
     O_GTE = 340,
     O_LTE = 341,
     O_SB_RANGE = 342,
     O_RSH = 343,
     O_LSH = 344,
     At = 345,
     Field = 346
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
#define Dimensionality 280
#define Dimension_No 281
#define Single 282
#define Double 283
#define Format 284
#define Real 285
#define Integer 286
#define Boolean 287
#define Character 288
#define String 289
#define Type_Name 290
#define Variable_Name 291
#define Space 292
#define Space_ID 293
#define V_Dimension 294
#define Activate 295
#define For 296
#define If 297
#define Repeat 298
#define Else 299
#define From 300
#define In 301
#define Step 302
#define Foreach 303
#define Range 304
#define Local 305
#define Index 306
#define C_Sub_Partition 307
#define While 308
#define Do 309
#define Sequence 310
#define To 311
#define Of 312
#define Link 313
#define Create 314
#define Link_or_Create 315
#define Dynamic 316
#define P_Sub_Partition 317
#define Ordered 318
#define Unordered 319
#define Replicated 320
#define Padding 321
#define Relative_To 322
#define Divides 323
#define Sub_Partitions 324
#define Partitions 325
#define Unpartitioned 326
#define Ascends 327
#define Descends 328
#define New_Line 329
#define R_MIN_ENTRY 330
#define R_MAX_ENTRY 331
#define R_MIN 332
#define R_MAX 333
#define R_MULT 334
#define R_ADD 335
#define O_OR 336
#define O_AND 337
#define O_NE 338
#define O_EQ 339
#define O_GTE 340
#define O_LTE 341
#define O_SB_RANGE 342
#define O_RSH 343
#define O_LSH 344
#define At 345
#define Field 346




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
{

/* Line 293 of yacc.c  */
#line 25 "IT.y"

        int i;
        char c;
        double r;
	char *s;



/* Line 293 of yacc.c  */
#line 314 "y.tab.c"
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
#line 339 "y.tab.c"

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
#define YYFINAL  18
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   602

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  111
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  105
/* YYNRULES -- Number of rules.  */
#define YYNRULES  219
/* YYNRULES -- Number of states.  */
#define YYNSTATES  424

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   346

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    99,     2,     2,     2,    96,     2,     2,
     104,   107,    97,    94,    75,    95,   102,    98,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,   105,   110,
      87,    76,    88,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,   103,     2,   106,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,   108,     2,   109,     2,     2,     2,     2,
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
      89,    90,    91,    92,    93,   100,   101
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     5,     7,    10,    12,    14,    16,    18,
      23,    25,    28,    32,    34,    38,    40,    42,    44,    47,
      50,    52,    54,    56,    58,    60,    65,    70,    74,    77,
      81,    87,    91,    93,    96,   100,   102,   104,   106,   108,
     112,   118,   119,   122,   126,   128,   131,   135,   137,   139,
     141,   142,   146,   149,   150,   154,   158,   160,   162,   165,
     168,   170,   173,   178,   185,   186,   192,   198,   200,   202,
     207,   209,   212,   223,   232,   233,   238,   240,   241,   246,
     248,   250,   253,   255,   258,   263,   265,   269,   271,   276,
     278,   282,   284,   288,   290,   296,   297,   302,   303,   305,
     309,   311,   313,   314,   319,   320,   331,   333,   335,   337,
     340,   344,   346,   350,   353,   354,   356,   358,   364,   367,
     372,   373,   376,   377,   380,   382,   385,   389,   391,   394,
     396,   398,   400,   402,   414,   421,   428,   430,   434,   439,
     440,   443,   452,   453,   455,   458,   463,   472,   476,   480,
     484,   488,   492,   496,   500,   504,   508,   512,   516,   520,
     523,   527,   531,   535,   539,   543,   545,   547,   549,   551,
     553,   559,   563,   565,   567,   569,   571,   573,   577,   582,
     584,   588,   590,   592,   594,   596,   598,   600,   602,   607,
     608,   610,   612,   616,   618,   622,   624,   626,   628,   630,
     632,   639,   640,   643,   646,   651,   653,   661,   662,   665,
     670,   674,   676,   678,   681,   686,   688,   691,   695,   699
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
     112,     0,    -1,   113,    -1,   114,    -1,   114,   113,    -1,
     115,    -1,   123,    -1,   210,    -1,   201,    -1,     4,    35,
     105,   116,    -1,   117,    -1,   117,   116,    -1,   118,   105,
     119,    -1,    36,    -1,    36,    75,   118,    -1,   120,    -1,
     121,    -1,    16,    -1,    18,    27,    -1,    18,    28,    -1,
      17,    -1,    19,    -1,    20,    -1,    22,    -1,    35,    -1,
      23,   122,    57,   120,    -1,   103,    31,   106,   122,    -1,
     103,    31,   106,    -1,   124,   125,    -1,     5,    34,   105,
      -1,   126,   134,   138,   141,   150,    -1,     9,   105,   127,
      -1,   128,    -1,   128,   127,    -1,   118,   105,   129,    -1,
     119,    -1,   130,    -1,   131,    -1,   132,    -1,    24,    57,
     119,    -1,    25,    23,    57,   120,   133,    -1,    -1,    29,
      35,    -1,    10,   105,   135,    -1,   136,    -1,   136,   135,
      -1,   118,   105,   137,    -1,    58,    -1,    59,    -1,    60,
      -1,    -1,   139,   105,   178,    -1,    11,   140,    -1,    -1,
     104,   118,   107,    -1,    12,   105,   142,    -1,   144,    -1,
     143,    -1,   143,   142,    -1,   144,   148,    -1,   145,    -1,
     145,   144,    -1,   146,   108,   149,   109,    -1,    34,   104,
      37,    38,   107,   147,    -1,    -1,    40,    42,   192,    46,
     192,    -1,    43,   105,    45,    34,   176,    -1,   142,    -1,
     178,    -1,    13,   140,   105,   151,    -1,   152,    -1,   152,
     151,    -1,    37,    38,    87,    25,    88,   153,   155,   108,
     157,   109,    -1,    37,    38,    87,    71,    88,   108,   118,
     109,    -1,    -1,    87,   154,    88,   153,    -1,    61,    -1,
      -1,    68,    37,    38,   156,    -1,    70,    -1,    69,    -1,
     158,   169,    -1,   159,    -1,   159,   158,    -1,   160,   105,
     163,   168,    -1,   161,    -1,   161,    75,   160,    -1,    36,
      -1,    36,    87,   162,    88,    -1,    39,    -1,    39,    75,
     162,    -1,   164,    -1,   164,    75,   163,    -1,    65,    -1,
      36,   104,   166,   107,   165,    -1,    -1,    66,   104,   166,
     107,    -1,    -1,   167,    -1,   167,    75,   166,    -1,    36,
      -1,    31,    -1,    -1,   110,    67,    37,    38,    -1,    -1,
      62,    87,    25,    88,    87,   170,    88,   108,   171,   109,
      -1,    63,    -1,    64,    -1,   172,    -1,   172,   171,    -1,
     160,   105,   173,    -1,   174,    -1,   174,    75,   173,    -1,
     164,   175,    -1,    -1,    72,    -1,    73,    -1,    41,   192,
      46,   192,   177,    -1,    53,   192,    -1,    48,    37,    38,
      52,    -1,    -1,    47,   192,    -1,    -1,   179,   180,    -1,
     182,    -1,   182,   181,    -1,   182,   181,   180,    -1,    74,
      -1,    74,   181,    -1,   184,    -1,   183,    -1,   188,    -1,
     192,    -1,    54,    46,    55,   108,   180,   109,    41,   192,
      46,   192,   177,    -1,    54,   108,   180,   109,    41,   185,
      -1,    54,   108,   180,   109,    53,   192,    -1,   186,    -1,
     186,   110,   185,    -1,   118,    46,    36,   187,    -1,    -1,
      84,   192,    -1,    42,   104,   192,   107,   108,   180,   109,
     189,    -1,    -1,   190,    -1,   191,   189,    -1,    44,   108,
     180,   109,    -1,    44,    42,   104,   192,   107,   108,   180,
     109,    -1,   192,    94,   192,    -1,   192,    95,   192,    -1,
     192,    97,   192,    -1,   192,    98,   192,    -1,   192,    87,
     192,    -1,   192,    88,   192,    -1,   192,    76,   192,    -1,
     192,    96,   192,    -1,   192,    83,   192,    -1,   192,    84,
     192,    -1,   192,    86,   192,    -1,   192,    85,   192,    -1,
      99,   192,    -1,   192,    89,   192,    -1,   192,    90,   192,
      -1,   192,    93,   192,    -1,   192,    92,   192,    -1,   194,
     196,   194,    -1,   193,    -1,   194,    -1,   197,    -1,   206,
      -1,   204,    -1,   192,   100,   104,   199,   107,    -1,   104,
     192,   107,    -1,    31,    -1,    30,    -1,    32,    -1,    33,
      -1,   200,    -1,   194,   102,   200,    -1,   194,   103,   195,
     106,    -1,   192,    -1,   192,    91,   192,    -1,    91,    -1,
      82,    -1,    81,    -1,    80,    -1,    79,    -1,    78,    -1,
      77,    -1,   194,   104,   198,   107,    -1,    -1,    34,    -1,
     192,    -1,   192,    75,   198,    -1,   200,    -1,   200,    95,
      31,    -1,    36,    -1,    26,    -1,    49,    -1,    50,    -1,
      51,    -1,     3,   104,    36,   107,   105,   202,    -1,    -1,
     203,   180,    -1,     7,   205,    -1,   119,   104,   198,   107,
      -1,   130,    -1,     8,   104,    34,   110,   200,   207,   107,
      -1,    -1,   110,   208,    -1,   110,   208,   110,   208,    -1,
     209,   105,   198,    -1,    11,    -1,    13,    -1,   211,   215,
      -1,     6,    36,   105,   212,    -1,   214,    -1,   213,   214,
      -1,    14,   105,   127,    -1,    15,   105,   127,    -1,    12,
     105,   178,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,    70,    70,    73,    74,    75,    75,    75,    75,    76,
      77,    78,    79,    80,    81,    82,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    93,    94,    97,    98,
      99,   103,   104,   105,   106,   107,   107,   108,   108,   109,
     110,   112,   112,   115,   116,   116,   117,   118,   119,   120,
     123,   124,   125,   126,   127,   130,   131,   132,   133,   134,
     135,   136,   137,   138,   140,   141,   142,   143,   144,   147,
     148,   149,   150,   152,   154,   155,   156,   157,   158,   159,
     159,   160,   161,   162,   163,   164,   165,   166,   167,   168,
     169,   170,   171,   172,   173,   175,   176,   177,   178,   179,
     180,   180,   181,   182,   183,   184,   186,   186,   187,   188,
     189,   190,   191,   192,   193,   193,   193,   195,   196,   197,
     198,   198,   200,   200,   201,   202,   203,   204,   204,   205,
     206,   207,   208,   209,   211,   212,   213,   214,   215,   216,
     216,   217,   218,   219,   220,   221,   222,   224,   225,   226,
     227,   228,   229,   230,   231,   232,   233,   234,   235,   236,
     237,   238,   239,   240,   241,   242,   243,   244,   245,   246,
     247,   248,   249,   249,   249,   249,   250,   251,   252,   253,
     254,   255,   256,   256,   256,   257,   257,   257,   258,   259,
     259,   259,   259,   260,   261,   262,   263,   263,   263,   263,
     266,   267,   267,   268,   269,   270,   271,   272,   272,   273,
     274,   275,   275,   277,   278,   279,   279,   280,   281,   282
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
  "T_Array", "T_List", "Dimensionality", "Dimension_No", "Single",
  "Double", "Format", "Real", "Integer", "Boolean", "Character", "String",
  "Type_Name", "Variable_Name", "Space", "Space_ID", "V_Dimension",
  "Activate", "For", "If", "Repeat", "Else", "From", "In", "Step",
  "Foreach", "Range", "Local", "Index", "C_Sub_Partition", "While", "Do",
  "Sequence", "To", "Of", "Link", "Create", "Link_or_Create", "Dynamic",
  "P_Sub_Partition", "Ordered", "Unordered", "Replicated", "Padding",
  "Relative_To", "Divides", "Sub_Partitions", "Partitions",
  "Unpartitioned", "Ascends", "Descends", "New_Line", "','", "'='",
  "R_MIN_ENTRY", "R_MAX_ENTRY", "R_MIN", "R_MAX", "R_MULT", "R_ADD",
  "O_OR", "O_AND", "O_NE", "O_EQ", "'<'", "'>'", "O_GTE", "O_LTE",
  "O_SB_RANGE", "O_RSH", "O_LSH", "'+'", "'-'", "'%'", "'*'", "'/'", "'!'",
  "At", "Field", "'.'", "'['", "'('", "':'", "']'", "')'", "'{'", "'}'",
  "';'", "$accept", "program", "components", "component", "tuple",
  "element_defs", "element_def", "names", "static_type", "scalar_type",
  "static_array", "static_dims", "task", "task_header", "task_body",
  "define", "definitions", "definition", "type", "dynamic_type", "list",
  "dynamic_array", "format", "environment", "linkages", "linkage", "mode",
  "initialize", "init_header", "arguments", "compute", "meta_stages",
  "meta_stage", "stage_sequence", "compute_stage", "stage_header",
  "activation_command", "repeat_control", "stage_body", "partition",
  "partition_specs", "partition_spec", "attributes", "attribute",
  "divides", "parent_config", "data_dist", "main_dist", "data_spec",
  "var_list", "var", "dimensions", "instr_list", "instr", "padding",
  "partition_args", "partition_arg", "relativity", "sub_dist", "nature",
  "data_sub_dist", "data_sub_spec", "ordered_instr_list", "ordered_instr",
  "order", "repeat_loop", "step_expr", "code", "$@1", "stmt_block",
  "new_lines", "stmt", "sequencial_loop", "parallel_loop", "index_ranges",
  "index_range", "restrictions", "if_else_block", "else_block", "else",
  "else_if", "expr", "constant", "field", "array_index", "reduction",
  "function_call", "args", "epoch", "id", "coordinator", "meta_code",
  "$@2", "create_obj", "object_type", "task_call", "optional_secs",
  "optional_sec", "section", "function", "function_header", "in_out",
  "input", "output", "function_body", 0
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
     333,   334,   335,   336,   337,   338,   339,    60,    62,   340,
     341,   342,   343,   344,    43,    45,    37,    42,    47,    33,
     345,   346,    46,    91,    40,    58,    93,    41,   123,   125,
      59
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,   111,   112,   113,   113,   114,   114,   114,   114,   115,
     116,   116,   117,   118,   118,   119,   119,   120,   120,   120,
     120,   120,   120,   120,   120,   121,   122,   122,   123,   124,
     125,   126,   127,   127,   128,   129,   129,   130,   130,   131,
     132,   133,   133,   134,   135,   135,   136,   137,   137,   137,
     138,   138,   139,   140,   140,   141,   142,   142,   142,   143,
     144,   144,   145,   146,   147,   147,   148,   149,   149,   150,
     151,   151,   152,   152,   153,   153,   154,   155,   155,   156,
     156,   157,   158,   158,   159,   160,   160,   161,   161,   162,
     162,   163,   163,   164,   164,   165,   165,   166,   166,   166,
     167,   167,   168,   168,   169,   169,   170,   170,   171,   171,
     172,   173,   173,   174,   175,   175,   175,   176,   176,   176,
     177,   177,   179,   178,   180,   180,   180,   181,   181,   182,
     182,   182,   182,   183,   184,   184,   185,   185,   186,   187,
     187,   188,   189,   189,   189,   190,   191,   192,   192,   192,
     192,   192,   192,   192,   192,   192,   192,   192,   192,   192,
     192,   192,   192,   192,   192,   192,   192,   192,   192,   192,
     192,   192,   193,   193,   193,   193,   194,   194,   194,   195,
     195,   195,   196,   196,   196,   196,   196,   196,   197,   198,
     198,   198,   198,   199,   199,   200,   200,   200,   200,   200,
     201,   203,   202,   204,   205,   205,   206,   207,   207,   207,
     208,   209,   209,   210,   211,   212,   212,   213,   214,   215
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     1,     2,     1,     1,     1,     1,     4,
       1,     2,     3,     1,     3,     1,     1,     1,     2,     2,
       1,     1,     1,     1,     1,     4,     4,     3,     2,     3,
       5,     3,     1,     2,     3,     1,     1,     1,     1,     3,
       5,     0,     2,     3,     1,     2,     3,     1,     1,     1,
       0,     3,     2,     0,     3,     3,     1,     1,     2,     2,
       1,     2,     4,     6,     0,     5,     5,     1,     1,     4,
       1,     2,    10,     8,     0,     4,     1,     0,     4,     1,
       1,     2,     1,     2,     4,     1,     3,     1,     4,     1,
       3,     1,     3,     1,     5,     0,     4,     0,     1,     3,
       1,     1,     0,     4,     0,    10,     1,     1,     1,     2,
       3,     1,     3,     2,     0,     1,     1,     5,     2,     4,
       0,     2,     0,     2,     1,     2,     3,     1,     2,     1,
       1,     1,     1,    11,     6,     6,     1,     3,     4,     0,
       2,     8,     0,     1,     2,     4,     8,     3,     3,     3,
       3,     3,     3,     3,     3,     3,     3,     3,     3,     2,
       3,     3,     3,     3,     3,     1,     1,     1,     1,     1,
       5,     3,     1,     1,     1,     1,     1,     3,     4,     1,
       3,     1,     1,     1,     1,     1,     1,     1,     4,     0,
       1,     1,     3,     1,     3,     1,     1,     1,     1,     1,
       6,     0,     2,     2,     4,     1,     7,     0,     2,     4,
       3,     1,     1,     2,     4,     1,     2,     3,     3,     3
};

/* YYDEFACT[STATE-NAME] -- Default reduction number in state STATE-NUM.
   Performed when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       0,     0,     0,     0,     0,     0,     2,     3,     5,     6,
       0,     8,     7,     0,     0,     0,     0,     0,     1,     4,
       0,    28,     0,     0,   213,     0,     0,    29,     0,     0,
       0,    50,   122,     0,    13,     9,    10,     0,     0,     0,
     214,     0,   215,     0,    31,    32,     0,    53,     0,     0,
     219,     0,   201,     0,    11,     0,     0,     0,   216,     0,
      33,     0,    43,    44,     0,    52,     0,     0,   122,     0,
       0,   196,   173,   172,   174,   175,   195,     0,   197,   198,
     199,     0,     0,     0,   123,   124,   130,   129,   131,   132,
     165,   166,   167,   176,   169,   168,   200,     0,    14,    17,
      20,     0,    21,    22,    23,     0,    24,    12,    15,    16,
     217,   218,     0,     0,    35,    34,    36,    37,    38,     0,
      45,     0,     0,    53,    30,    51,     0,   205,   203,     0,
       0,     0,     0,   159,     0,   127,   125,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   187,   186,   185,   184,   183,   182,
       0,     0,   189,     0,   202,    18,    19,     0,     0,     0,
       0,    47,    48,    49,    46,    54,     0,    55,    57,    56,
      60,     0,     0,   189,     0,     0,     0,     0,   171,   128,
     126,   153,   155,   156,   158,   157,   151,   152,   160,   161,
     163,   162,   147,   148,   154,   149,   150,     0,   177,   181,
     179,     0,   190,   191,     0,   164,     0,     0,    39,     0,
       0,    58,     0,    59,    61,   122,     0,     0,     0,     0,
       0,     0,     0,   193,     0,   178,   189,   188,    27,    25,
      41,     0,     0,    67,     0,    68,     0,    69,    70,   204,
     207,     0,     0,     0,     0,   170,     0,   180,   192,    26,
       0,    40,     0,     0,    62,     0,    71,     0,     0,     0,
       0,     0,   134,   136,   135,   194,    42,    64,     0,     0,
     211,   212,   208,     0,   206,   142,     0,     0,     0,     0,
      63,     0,     0,     0,    66,     0,     0,     0,   189,     0,
     141,   143,   142,     0,   139,   137,     0,     0,     0,   118,
      74,     0,   209,   210,     0,     0,   144,     0,     0,   138,
       0,     0,     0,     0,    77,     0,     0,     0,   120,   140,
       0,   120,   119,    76,     0,     0,     0,     0,     0,   145,
       0,   133,    65,   117,    74,     0,     0,    73,     0,   121,
      75,     0,    87,     0,   104,    82,     0,    85,     0,    80,
      79,    78,     0,    72,     0,    81,    83,     0,     0,     0,
      89,     0,     0,     0,    93,   102,    91,    86,   146,     0,
      88,     0,    97,     0,    84,     0,    90,     0,   101,   100,
       0,    98,     0,    92,     0,    95,    97,     0,   106,   107,
       0,     0,    94,    99,   103,     0,    97,     0,     0,     0,
       0,   108,    96,     0,   105,   109,   114,   110,   111,   115,
     116,   113,     0,   112
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     5,     6,     7,     8,    35,    36,    43,   107,   108,
     109,   168,     9,    10,    21,    22,    44,    45,   115,   116,
     117,   118,   261,    31,    62,    63,   174,    48,    49,    65,
      67,   177,   178,   179,   180,   181,   290,   223,   244,   124,
     247,   248,   324,   334,   336,   361,   353,   354,   355,   356,
     357,   371,   375,   376,   402,   390,   391,   384,   365,   400,
     410,   411,   417,   418,   421,   294,   341,    50,    51,    84,
     136,    85,    86,    87,   272,   273,   319,    88,   300,   301,
     302,    89,    90,    91,   211,   163,    92,   214,   232,    93,
      11,    96,    97,    94,   128,    95,   268,   282,   283,    12,
      13,    40,    41,    42,    24
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -381
static const yytype_int16 yypact[] =
{
     146,   -87,    -5,     7,    18,    79,  -381,   146,  -381,  -381,
      35,  -381,  -381,    89,    55,     2,    17,    26,  -381,  -381,
      27,  -381,   116,    30,  -381,    29,   107,  -381,    21,   107,
      40,   143,  -381,    51,    84,  -381,   107,    63,    64,    65,
    -381,   157,  -381,    69,  -381,   107,   107,    71,   164,    72,
    -381,    67,  -381,   107,  -381,   516,   107,   107,  -381,   250,
    -381,    73,  -381,   107,   107,  -381,    74,   179,  -381,   250,
      90,  -381,  -381,  -381,  -381,  -381,  -381,    91,  -381,  -381,
    -381,   -30,   183,   183,  -381,   119,  -381,  -381,  -381,   431,
    -381,     8,  -381,  -381,  -381,  -381,  -381,    67,  -381,  -381,
    -381,    23,  -381,  -381,  -381,    94,  -381,  -381,  -381,  -381,
    -381,  -381,   141,   176,  -381,  -381,  -381,  -381,  -381,    24,
    -381,    93,   169,    71,  -381,  -381,   100,  -381,  -381,   172,
     183,   152,    67,   108,   309,   119,    67,   183,   183,   183,
     183,   183,   183,   183,   183,   183,   183,   183,   183,   183,
     183,   183,   183,   114,  -381,  -381,  -381,  -381,  -381,  -381,
      -4,    97,   131,    -4,  -381,  -381,  -381,   180,   163,   516,
     166,  -381,  -381,  -381,  -381,  -381,   117,  -381,   169,   181,
     169,   118,   123,   131,   121,   334,   130,   120,  -381,  -381,
    -381,   431,   460,   476,   490,   490,   499,   499,   499,   499,
     240,   240,    44,    44,   108,   108,   108,    -4,  -381,  -381,
     408,   133,  -381,   385,   134,    -7,   136,   167,  -381,   167,
     206,  -381,   139,  -381,  -381,   169,   209,   140,    -4,   142,
      67,   -28,   144,   153,   183,  -381,   131,  -381,    94,  -381,
     223,   215,   210,  -381,   147,  -381,   219,  -381,   209,  -381,
     148,    67,   151,   107,   183,  -381,   230,   431,  -381,  -381,
     228,  -381,   171,   237,  -381,   177,  -381,    81,   173,   170,
     235,   242,  -381,   200,   431,  -381,  -381,   241,   -27,     6,
    -381,  -381,   201,   178,  -381,   245,   183,   254,   107,   244,
    -381,   183,   275,   183,  -381,   225,   234,    81,   131,   -35,
    -381,  -381,   245,   208,   246,  -381,   183,   231,   294,   431,
     255,   233,  -381,  -381,   239,    67,  -381,   183,   183,  -381,
     263,   183,   292,   284,   296,   107,   183,   256,   286,   431,
     183,   286,  -381,  -381,   266,   329,   259,   268,   359,  -381,
     183,  -381,   431,  -381,   255,   330,   351,  -381,   280,   431,
    -381,    50,   302,   281,   338,   351,   303,   316,    67,  -381,
    -381,  -381,   372,  -381,   325,  -381,  -381,   -25,   351,   304,
     339,   327,   400,   332,  -381,   323,   362,  -381,  -381,   372,
    -381,   350,    -2,   373,  -381,   -25,  -381,   352,  -381,  -381,
     343,   383,   425,  -381,    61,   397,    -2,   426,  -381,  -381,
     377,   363,  -381,  -381,  -381,   368,    -2,   351,   379,   382,
     380,   351,  -381,   -25,  -381,  -381,    41,  -381,   413,  -381,
    -381,  -381,   -25,  -381
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -381,  -381,   483,  -381,  -381,   473,  -381,   -26,   -54,  -111,
    -381,   272,  -381,  -381,  -381,  -381,   -33,  -381,  -381,   442,
    -381,  -381,  -381,  -381,   449,  -381,  -381,  -381,  -381,   390,
    -381,  -172,  -381,   342,  -381,  -381,  -381,  -381,  -381,  -381,
     282,  -381,   193,  -381,  -381,  -381,  -381,   185,  -381,  -359,
    -381,   162,   158,  -380,  -381,  -378,  -381,  -381,  -381,  -381,
     156,  -381,   137,  -381,  -381,  -381,   211,   -65,  -381,   -93,
     440,  -381,  -381,  -381,   293,  -381,  -381,  -381,   287,  -381,
    -381,   -81,  -381,   435,  -381,  -381,  -381,  -164,  -381,  -152,
    -381,  -381,  -381,  -381,  -381,  -381,  -381,   305,  -381,  -381,
    -381,  -381,  -381,   559,  -381
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -1
static const yytype_uint16 yytable[] =
{
      37,   133,   134,   125,   164,   114,   221,   314,   208,   377,
      37,   373,    60,   253,   291,   126,   131,    14,   403,   227,
      61,   292,    71,   110,   111,   254,   293,    98,   408,   388,
      15,   295,    76,   416,   389,    38,    39,    61,   121,   187,
     374,    16,   416,   190,    20,    78,    79,    80,   409,   185,
     165,   166,   409,   243,    17,   233,   191,   192,   193,   194,
     195,   196,   197,   198,   199,   200,   201,   202,   203,   204,
     205,   206,   258,   315,    69,    70,   250,   296,   132,    18,
     210,   213,   171,   172,   173,   154,   155,   156,   157,   158,
     159,    25,   280,    71,   281,   160,   161,    72,    73,    74,
      75,    23,   213,    76,    69,    70,   239,    26,   240,    77,
     160,   161,   162,   419,   420,   218,    78,    79,    80,   359,
     360,    81,    27,    71,   398,   399,    30,    72,    73,    74,
      75,    28,    29,    76,   313,    32,    33,   252,    69,    70,
     150,   151,   152,    34,   153,    46,    78,    79,    80,     1,
       2,     3,     4,   257,    47,   213,    52,    71,   269,    53,
     245,    72,    73,    74,    75,   212,    82,    76,    55,    56,
      57,    83,    39,   274,    59,    64,    66,    68,   119,   122,
      78,    79,    80,    99,   100,   101,   102,   103,   209,   104,
      69,    70,   123,   135,   129,   130,    82,   167,   169,   170,
     175,    83,   106,   176,   183,   303,   184,   186,   153,    71,
     307,   216,   309,    72,    73,    74,    75,   213,   207,    76,
     217,   220,   327,   219,   222,   320,   225,   271,   226,   231,
      82,   228,    78,    79,    80,    83,   328,   329,   230,   235,
     331,   237,   238,   241,   242,   338,   246,   249,   256,   342,
     251,   255,   260,   262,   317,   263,   264,   265,   267,   349,
     270,   275,   271,   276,   279,   369,    99,   100,   101,   102,
     103,   278,   104,   105,   112,   113,   286,   321,   277,   285,
     284,   289,    82,   298,   137,   106,   306,    83,   287,   299,
     304,   138,   139,   140,   141,   142,   143,   144,   145,   337,
     146,   147,   148,   149,   150,   151,   152,   137,   153,   330,
     288,   297,   308,   310,   138,   139,   140,   141,   142,   143,
     144,   145,   311,   146,   147,   148,   149,   150,   151,   152,
     318,   153,   322,   340,   148,   149,   150,   151,   152,   137,
     153,   325,   323,   326,   332,   333,   138,   139,   140,   141,
     142,   143,   144,   145,   344,   146,   147,   148,   149,   150,
     151,   152,   137,   153,   335,   339,   345,   346,   351,   138,
     139,   140,   141,   142,   143,   144,   145,   347,   146,   147,
     148,   149,   150,   151,   152,   137,   153,   352,   358,   362,
     363,   368,   138,   139,   140,   141,   142,   143,   144,   145,
     364,   146,   147,   148,   149,   150,   151,   152,   367,   153,
     137,   370,   372,   378,   379,   380,   188,   138,   139,   140,
     141,   142,   143,   144,   145,   381,   146,   147,   148,   149,
     150,   151,   152,   383,   153,   137,   382,   385,   387,   394,
     392,   229,   138,   139,   140,   141,   142,   143,   144,   145,
     395,   146,   147,   148,   149,   150,   151,   152,   396,   153,
     236,   137,   397,   401,   404,   405,   348,   406,   138,   139,
     140,   141,   142,   143,   144,   145,   407,   146,   147,   148,
     149,   150,   151,   152,   137,   153,   412,   413,   422,   414,
      19,   138,   139,   140,   141,   142,   143,   144,   145,   234,
     146,   147,   148,   149,   150,   151,   152,   137,   153,    54,
     259,   127,   120,   182,   138,   139,   140,   141,   142,   143,
     144,   145,   224,   146,   147,   148,   149,   150,   151,   152,
     266,   153,    99,   100,   101,   102,   103,   350,   104,   105,
     366,   386,   343,   393,   139,   140,   141,   142,   143,   144,
     145,   106,   146,   147,   148,   149,   150,   151,   152,   423,
     153,   140,   141,   142,   143,   144,   145,   415,   146,   147,
     148,   149,   150,   151,   152,   189,   153,   142,   143,   144,
     145,   305,   146,   147,   148,   149,   150,   151,   152,   316,
     153,   146,   147,   148,   149,   150,   151,   152,   215,   153,
      58,     0,   312
};

#define yypact_value_is_default(yystate) \
  ((yystate) == (-381))

#define yytable_value_is_error(yytable_value) \
  YYID (0)

static const yytype_int16 yycheck[] =
{
      26,    82,    83,    68,    97,    59,   178,    42,   160,   368,
      36,    36,    45,    41,    41,    69,    46,   104,   396,   183,
      46,    48,    26,    56,    57,    53,    53,    53,   406,    31,
      35,    25,    36,   413,    36,    14,    15,    63,    64,   132,
      65,    34,   422,   136,     9,    49,    50,    51,   407,   130,
      27,    28,   411,   225,    36,   207,   137,   138,   139,   140,
     141,   142,   143,   144,   145,   146,   147,   148,   149,   150,
     151,   152,   236,   108,     7,     8,   228,    71,   108,     0,
     161,   162,    58,    59,    60,    77,    78,    79,    80,    81,
      82,    36,    11,    26,    13,   102,   103,    30,    31,    32,
      33,    12,   183,    36,     7,     8,   217,   105,   219,    42,
     102,   103,   104,    72,    73,   169,    49,    50,    51,    69,
      70,    54,   105,    26,    63,    64,    10,    30,    31,    32,
      33,   105,   105,    36,   298,   105,   107,   230,     7,     8,
      96,    97,    98,    36,   100,   105,    49,    50,    51,     3,
       4,     5,     6,   234,    11,   236,   105,    26,   251,    75,
     225,    30,    31,    32,    33,    34,    99,    36,   105,   105,
     105,   104,    15,   254,   105,   104,    12,   105,   105,   105,
      49,    50,    51,    16,    17,    18,    19,    20,    91,    22,
       7,     8,    13,    74,   104,   104,    99,   103,    57,    23,
     107,   104,    35,    34,   104,   286,    34,    55,   100,    26,
     291,    31,   293,    30,    31,    32,    33,   298,   104,    36,
      57,   104,   315,    57,    43,   306,   108,   253,   105,   109,
      99,   110,    49,    50,    51,   104,   317,   318,   108,   106,
     321,   107,   106,    37,   105,   326,    37,   107,    95,   330,
     108,   107,    29,    38,    46,    45,   109,    38,   110,   340,
     109,    31,   288,    35,    87,   358,    16,    17,    18,    19,
      20,    34,    22,    23,    24,    25,    41,    46,   107,   109,
     107,    40,    99,   105,    76,    35,    42,   104,    46,    44,
      36,    83,    84,    85,    86,    87,    88,    89,    90,   325,
      92,    93,    94,    95,    96,    97,    98,    76,   100,    46,
     110,   110,    37,    88,    83,    84,    85,    86,    87,    88,
      89,    90,    88,    92,    93,    94,    95,    96,    97,    98,
      84,   100,    38,    47,    94,    95,    96,    97,    98,    76,
     100,   108,    87,   104,    52,    61,    83,    84,    85,    86,
      87,    88,    89,    90,    88,    92,    93,    94,    95,    96,
      97,    98,    76,   100,    68,   109,    37,   108,    38,    83,
      84,    85,    86,    87,    88,    89,    90,   109,    92,    93,
      94,    95,    96,    97,    98,    76,   100,    36,   108,    87,
     109,    75,    83,    84,    85,    86,    87,    88,    89,    90,
      62,    92,    93,    94,    95,    96,    97,    98,   105,   100,
      76,    39,    87,   109,    75,    88,   107,    83,    84,    85,
      86,    87,    88,    89,    90,    25,    92,    93,    94,    95,
      96,    97,    98,   110,   100,    76,   104,    75,    88,    87,
      67,   107,    83,    84,    85,    86,    87,    88,    89,    90,
     107,    92,    93,    94,    95,    96,    97,    98,    75,   100,
      75,    76,    37,    66,    38,    88,   107,   104,    83,    84,
      85,    86,    87,    88,    89,    90,   108,    92,    93,    94,
      95,    96,    97,    98,    76,   100,   107,   105,    75,   109,
       7,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    93,    94,    95,    96,    97,    98,    76,   100,    36,
     238,    69,    63,   123,    83,    84,    85,    86,    87,    88,
      89,    90,   180,    92,    93,    94,    95,    96,    97,    98,
     248,   100,    16,    17,    18,    19,    20,   344,    22,    23,
     355,   379,   331,   385,    84,    85,    86,    87,    88,    89,
      90,    35,    92,    93,    94,    95,    96,    97,    98,   422,
     100,    85,    86,    87,    88,    89,    90,   411,    92,    93,
      94,    95,    96,    97,    98,   135,   100,    87,    88,    89,
      90,   288,    92,    93,    94,    95,    96,    97,    98,   302,
     100,    92,    93,    94,    95,    96,    97,    98,   163,   100,
      41,    -1,   297
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     4,     5,     6,   112,   113,   114,   115,   123,
     124,   201,   210,   211,   104,    35,    34,    36,     0,   113,
       9,   125,   126,    12,   215,    36,   105,   105,   105,   105,
      10,   134,   105,   107,    36,   116,   117,   118,    14,    15,
     212,   213,   214,   118,   127,   128,   105,    11,   138,   139,
     178,   179,   105,    75,   116,   105,   105,   105,   214,   105,
     127,   118,   135,   136,   104,   140,    12,   141,   105,     7,
       8,    26,    30,    31,    32,    33,    36,    42,    49,    50,
      51,    54,    99,   104,   180,   182,   183,   184,   188,   192,
     193,   194,   197,   200,   204,   206,   202,   203,   118,    16,
      17,    18,    19,    20,    22,    23,    35,   119,   120,   121,
     127,   127,    24,    25,   119,   129,   130,   131,   132,   105,
     135,   118,   105,    13,   150,   178,   119,   130,   205,   104,
     104,    46,   108,   192,   192,    74,   181,    76,    83,    84,
      85,    86,    87,    88,    89,    90,    92,    93,    94,    95,
      96,    97,    98,   100,    77,    78,    79,    80,    81,    82,
     102,   103,   104,   196,   180,    27,    28,   103,   122,    57,
      23,    58,    59,    60,   137,   107,    34,   142,   143,   144,
     145,   146,   140,   104,    34,   192,    55,   180,   107,   181,
     180,   192,   192,   192,   192,   192,   192,   192,   192,   192,
     192,   192,   192,   192,   192,   192,   192,   104,   200,    91,
     192,   195,    34,   192,   198,   194,    31,    57,   119,    57,
     104,   142,    43,   148,   144,   108,   105,   198,   110,   107,
     108,   109,   199,   200,    91,   106,    75,   107,   106,   120,
     120,    37,   105,   142,   149,   178,    37,   151,   152,   107,
     200,   108,   180,    41,    53,   107,    95,   192,   198,   122,
      29,   133,    38,    45,   109,    38,   151,   110,   207,   180,
     109,   118,   185,   186,   192,    31,    35,   107,    34,    87,
      11,    13,   208,   209,   107,   109,    41,    46,   110,    40,
     147,    41,    48,    53,   176,    25,    71,   110,   105,    44,
     189,   190,   191,   192,    36,   185,    42,   192,    37,   192,
      88,    88,   208,   198,    42,   108,   189,    46,    84,   187,
     192,    46,    38,    87,   153,   108,   104,   180,   192,   192,
      46,   192,    52,    61,   154,    68,   155,   118,   192,   109,
      47,   177,   192,   177,    88,    37,   108,   109,   107,   192,
     153,    38,    36,   157,   158,   159,   160,   161,   108,    69,
      70,   156,    87,   109,    62,   169,   158,   105,    75,   180,
      39,   162,    87,    36,    65,   163,   164,   160,   109,    75,
      88,    25,   104,   110,   168,    75,   162,    88,    31,    36,
     166,   167,    67,   163,    87,   107,    75,    37,    63,    64,
     170,    66,   165,   166,    38,    88,   104,   108,   166,   160,
     171,   172,   107,   105,   109,   171,   164,   173,   174,    72,
      73,   175,    75,   173
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
#line 70 "IT.y"
    {(yylsp[(1) - (1)]); // this is needed to make bison set up
							     	     // the location variable yylloc
								}
    break;

  case 122:

/* Line 1806 of yacc.c  */
#line 200 "IT.y"
    {BeginCode();}
    break;

  case 123:

/* Line 1806 of yacc.c  */
#line 200 "IT.y"
    {EndCode();}
    break;

  case 201:

/* Line 1806 of yacc.c  */
#line 267 "IT.y"
    {BeginProgram();}
    break;

  case 202:

/* Line 1806 of yacc.c  */
#line 267 "IT.y"
    {EndProgram();}
    break;



/* Line 1806 of yacc.c  */
#line 2107 "y.tab.c"
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
#line 285 "IT.y"


/*------------------------------------------ Epilogue ----------------------------------------------*/

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


