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

/* Line 2068 of yacc.c  */
#line 25 "IT.y"

        int i;
        char c;
        double r;
	char *s;



/* Line 2068 of yacc.c  */
#line 241 "y.tab.h"
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

