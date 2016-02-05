%{
#include "symtab.h"
#include "token.h"
#include "string.h"
#include "stdio.h"

#define B_TRUE 1
#define B_FALSE 0
#define DIM_LENGTH 3
#define DIMENSION_LENGTH 9

%}

OPERATOR          ([-+/*%=.,;:!<>()[\]{}])

%%

\/\/.*          	;
\\[[:space:]]+$		;
[:{][[:space:]]+	return yytext[0];
[[:space:]]+[}]		return '}';
\n+			return New_Line;
[[:space:]]     	;

Program			return Program;
Tuple			return Tuple;
Task			return Task;
Define			return S_Define;
Environment		return S_Environment;
Initialize		return S_Initialize;
Compute			return S_Compute;
Partition		return S_Partition;

Integer			return T_Integer;
Character		return T_Character;
Real			return T_Real;
Boolean			return T_Boolean;
Epoch			return T_Epoch;
Index			return T_Index;
Range			return T_Range;
array			return T_Array;
list			return T_List;

dimension[0-9]+		{
				char *dimensionNo = yytext + DIMENSION_LENGTH;
				yylval.i = atoi(dimensionNo);
				return A_Dimension;
			}
dimension		return A_Dimension;

Precision		return Precision;
Single			return P_Single;
Double			return P_Double;

link			return Link;
create			return Create;
"link-or-create"	return Link_or_Create;	

"<="			return O_LTE;
">="			return O_GTE;
"=="			return O_EQ;
"!="			return O_NE;
">>"			return O_RSH;
"<<"			return O_LSH;
"||"			return O_OR;
"&&"			return O_AND;
".."			return O_SB_RANGE;

"+="			return R_ADD;
"*="			return R_MULT;
"max="			return R_MAX;
"maxEntry="		return R_MAX_ENTRY;
"min="			return R_MIN;
"minEntry="		return R_MIN_ENTRY;	

Repeat			return Repeat;
Activate		return Activate;
do			return Do;
while			return While;
for			return For;
if			return If;
else			return Else;
in			return In;
from			return From;
to			return To;
step			return Step;
foreach			return Foreach;
sequence		return Sequence;
at			return At;
"sub-partition"		return Sub_Partition;
"local-range"		return Local_Range;
range			return Range;
of			return Of;

"Sub-partition"		return P_Sub_Partition;
unordered		return Unordered;
ordered			return Ordered;
replicated		return Replicated;
padding			return Padding;
divides			return Divides;
"sub-partitions"	return Sub_Partitions;
partitions		return Partitions;
"un-partitioned"	return Unpartitioned;
ascends			return Ascends;
descends		return Descends;

Space			return Space;
[0-9]D			yylval.i = yytext[0]; return S_Dimension;
dim[0-9]+		{	
				char *dimensionality = yytext + DIM_LENGTH;
				yylval.i = atoi(dimensionality);
				return V_Dimension;
			}

TRUE			yylval.i = B_TRUE; return Boolean;
FALSE			yylval.i = B_FALSE; return Boolean;
\'.\'			yylval.c = yytext[1]; return Character;
\".*\"			{	
				yylval.s = (char *) malloc(strlen(yytext) + 1);
                                strcpy(yylval.s, yytext);		
				return String;
			} 
[0-9]+\.[0-9]+  	yylval.r = atof(yytext); return Real;
[0-9]+          	yylval.i = atoi(yytext); return Integer;
[A-Z][0-9a-zA-Z_]+	{
				yylval.n = (char *) malloc(strlen(yytext) + 1);
			 	strcpy(yylval.n, yytext);
                  		return Type_Name;

			}
[a-z][0-9a-zA-Z_]*	{	
				yylval.n = (char *) malloc(strlen(yytext) + 1);
			 	strcpy(yylval.n, yytext);
                  		return Variable_Name;
			}
[A-Z]			yylval.c = yytext[0]; return Space_ID;

{OPERATOR}          	return yytext[0];

.               	fprintf(stderr,"Unexpected character %c (%2X)\n",yytext[0],yytext[0]);

%%
