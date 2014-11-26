/*------------------------------------------ Prolog -----------------------------------------------*/
%{
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
%}

/*---------------------------------------- Declarations -------------------------------------------*/

/* yylval 
 * ------
 * Here we define the type of the yylval global variable that is used by
 * the scanner to store attibute information about the token just scanned
 * and thus communicate that information to the parser. 
 */
%union {
        int i;
        char c;
        double r;
	char *s;
}

/* Tokens
 * ------
 * Here we tell yacc about all the token types that we are using.
 * Yacc will assign unique numbers to these and export the #define
 * in the generated y.tab.h header file.
 */

%token Program Tuple Task Function New Execute
%token S_Define S_Environment S_Initialize S_Compute S_Partition S_Arguments S_Results
%token T_Integer T_Character T_Real T_Boolean T_Epoch T_Index T_Range T_Array T_List
%token Dimensionality Dimension_No Single Double Format
%token Real Integer Boolean Character String
%token Type_Name Variable_Name
%token Space Space_ID V_Dimension
%token Activate For If Repeat Else From In Step Foreach Range Local Index
%token C_Sub_Partition While Do Sequence To Of
%token Link Create Link_or_Create
%token Dynamic P_Sub_Partition Ordered Unordered Replicated Padding Relative_To 
%token Divides Sub_Partitions Partitions Unpartitioned Ascends Descends
%token New_Line

%left ','
%right '=' R_ADD R_MULT R_MAX R_MIN R_MAX_ENTRY R_MIN_ENTRY
%left O_OR
%left O_AND
%left O_EQ O_NE
%left '<' '>' O_LTE O_GTE
%left O_SB_RANGE
%left O_LSH O_RSH
%left '+' '-'
%left '%' '*' '/'
%right '!'
%nonassoc At
%left Field
%left '.' '[' '('

/*---------------------------------------- Grammer Rules -------------------------------------------*/     
%%
program		: components					{@1; // this is needed to make bison set up
							     	     // the location variable yylloc
								};
components	: component
		| component components
component	: tuple | task | function | coordinator		;
tuple 		: Tuple Type_Name ':' element_defs		;
element_defs	: element_def 
		| element_def element_defs			;
element_def	: names ':' static_type				;
names		: Variable_Name
		| Variable_Name ',' names			;
static_type	: scalar_type | static_array			;
scalar_type	: T_Integer
		| T_Real Single
		| T_Real Double
		| T_Character
		| T_Boolean
		| T_Epoch
		| T_Range
		| Type_Name					;
static_array	: T_Array static_dims 
		  Of scalar_type				;
static_dims	: '['Integer']' static_dims		
		| '['Integer']'					;


task 		: task_header task_body				;
task_header	: Task String ':'				;
task_body	: define environment 
		  initialize compute partition			;


define		: S_Define ':' definitions			;
definitions	: definition
		| definition definitions			;
definition	: names ':' type				;
type		: static_type | dynamic_type			;
dynamic_type	: list | dynamic_array				;
list		: T_List Of static_type				;
dynamic_array	: Dimensionality T_Array Of 
		  scalar_type format				;
format		: | Format Type_Name				;


environment	: S_Environment ':' linkages			;
linkages 	: linkage | linkage linkages			;
linkage		: names ':' mode				;
mode		: Link
		| Create
		| Link_or_Create				;


initialize	: 
		| init_header ':' code			 	;
init_header	: S_Initialize arguments			;
arguments	: 
		| '(' names ')'					; 


compute		: S_Compute ':'	meta_stages			;
meta_stages	: stage_sequence
		| meta_stage 
		| meta_stage meta_stages			;
meta_stage	: stage_sequence repeat_control			;
stage_sequence	: compute_stage 
		| compute_stage stage_sequence			;
compute_stage	: stage_header '{' stage_body '}'		;
stage_header	: String '(' Space Space_ID ')'
		  activation_command				;
activation_command : 
		| Activate If expr In expr			;	 
repeat_control	: Repeat ':' From String repeat_loop 		;
stage_body	: meta_stages					
		| code						; 


partition	: S_Partition arguments	':' partition_specs	;
partition_specs	: partition_spec 
		| partition_spec partition_specs		;
partition_spec  : Space Space_ID '<' Dimensionality '>'
		  attributes divides '{' data_dist'}'		
		| Space Space_ID '<' Unpartitioned '>' 
		  '{' names '}'					;
attributes	:
		| '<' attribute '>' attributes			;
attribute	: Dynamic					;
divides		: 
		| Divides Space Space_ID parent_config		;
parent_config	: Partitions | Sub_Partitions			;
data_dist	: main_dist sub_dist				;
main_dist	: data_spec
		| data_spec main_dist				;
data_spec	: var_list ':' instr_list relativity		;
var_list	: var 
		| var ',' var_list				;
var		: Variable_Name
		| Variable_Name '<' dimensions '>'		;
dimensions	: V_Dimension 
		| V_Dimension ',' dimensions			;
instr_list	: instr
		| instr ',' instr_list				;
instr		: Replicated
		| Variable_Name '(' partition_args ')' 
		  padding					;
padding		:
		| Padding '(' partition_args ')'		;
partition_args	: 
		| partition_arg
		| partition_arg ',' partition_args		;
partition_arg	: Variable_Name | Integer			;	 	
relativity	:
		| ';' Relative_To Space Space_ID  		;			
sub_dist	:
		| P_Sub_Partition '<' Dimensionality'>'
		  '<' nature '>' '{' data_sub_dist '}'		;
nature		: Ordered | Unordered				;
data_sub_dist	: data_sub_spec 				
		| data_sub_spec data_sub_dist			;
data_sub_spec	: var_list ':' ordered_instr_list		;
ordered_instr_list : ordered_instr
		| ordered_instr ',' ordered_instr_list		;
ordered_instr	: instr order					;	
order		: | Ascends | Descends				;			

repeat_loop	: For expr In expr step_expr
		| While expr
		| Foreach Space Space_ID C_Sub_Partition	;
step_expr	: | Step expr					;	

code		: {BeginCode();}  stmt_block			{EndCode();};
stmt_block	: stmt
		| stmt new_lines 
		| stmt new_lines stmt_block			;
new_lines	: New_Line | New_Line new_lines			; 
stmt		: parallel_loop 
		| sequencial_loop
		| if_else_block					
		| expr						;
sequencial_loop	: Do In Sequence '{' stmt_block '}'
		For expr In expr step_expr			;
parallel_loop	: Do '{' stmt_block '}' For index_ranges	
		| Do '{' stmt_block '}' While expr		;
index_ranges	: index_range
		| index_range ';' index_ranges			;
index_range	: names In Variable_Name restrictions		;
restrictions	: | O_AND expr					;
if_else_block	: If '(' expr ')' '{' stmt_block '}' else_block	;
else_block	:
		| else
		| else_if else_block				;
else		: Else '{' stmt_block '}'			;
else_if		: Else If '(' expr ')' '{' stmt_block '}'	;
	
expr		: expr '+' expr
		| expr '-' expr
		| expr '*' expr
		| expr '/' expr
		| expr '<' expr
		| expr '>' expr
		| expr '=' expr
		| expr '%' expr
		| expr O_OR expr
		| expr O_AND expr
		| expr O_EQ expr
		| expr O_NE expr
		| '!' expr
		| expr O_GTE expr
		| expr O_LTE expr
		| expr O_LSH expr
		| expr O_RSH expr
		| field reduction field
		| constant
		| field
		| function_call
		| task_call
		| create_obj
		| expr At '(' epoch ')'				
		| '(' expr ')' 					;
constant	: Integer | Real | Boolean | Character		;	
field		: id %prec Field
		| field '.' id
		| field '[' array_index ']'			;
array_index	: expr 
		| expr O_SB_RANGE expr 
		| O_SB_RANGE				 	;
reduction	: R_ADD | R_MULT | R_MAX 
		| R_MIN | R_MAX_ENTRY | R_MIN_ENTRY		;
function_call	: field '(' args ')'				;
args		: | String | expr | expr ',' args		;		
epoch		: id 
		| id '-' Integer				; 
id		: Variable_Name 
		| Dimension_No | Range | Local | Index		;


coordinator	: Program '(' Variable_Name ')' ':' meta_code	;
meta_code	: {BeginProgram();} stmt_block 			{EndProgram();};
create_obj	: New object_type				;
object_type	: static_type '(' args ')' 
		| dynamic_type					;
task_call	: Execute '(' String ';' id optional_secs ')'	;
optional_secs	: | ';' optional_sec
		| ';' optional_sec ';' optional_sec		;
optional_sec	: section ':' args				;
section		: S_Initialize | S_Partition			;

function	: function_header function_body			;
function_header : Function Variable_Name ':' in_out		;
in_out		: output | input output				;
input		: S_Arguments ':' definitions			;
output		: S_Results ':' definitions			;
function_body	: S_Compute ':' code				;				 
					
		
%%

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

