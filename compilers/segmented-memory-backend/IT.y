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
#include "utils/utility.h"
#include "syntax/errors.h"

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
        int 				intConstant;
        bool 				booleanConstant;
	float				floatConstant;
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
	InitializerArg			*objInitArg;
	List<InitializerArg*>		*objInitArgList;
	List<Expr*>			*exprList;
	Stmt				*stmt;
	List<Stmt*>			*stmtList;
	ConditionalStmt			*condStmt;
	List<ConditionalStmt*>		*condStmtList;
	IndexRangeCondition		*rangeCond;
	SLoopAttribute                  *sloopAttr;
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
%token Single Double Format
%token T_Space
%token Activate For If Repeat Else From In Step Foreach Range Local Index
%token C_Sub_Partition While Do Sequence To Of
%token Link Create Link_or_Create
%token Dynamic P_Sub_Partition Ordered Unordered Replicated Padding Relative_To 
%token Divides Sub_Partitions Partitions Unpartitioned Ascends Descends
%token New_Line

%token <intConstant> Integer Dimensionality Dimension_No V_Dimension
%token <characterConstant> Character Space_ID
%token <floatConstant> Real_Single 
%token <doubleConstant> Real_Double 
%token <booleanConstant> Boolean
%token <stringConstant> Type_Name Variable_Name String

%left ','
%right '=' 
%right REDUCE
%left O_OR
%left O_AND
%left '|'
%left '^'
%left '&'
%left O_EQ O_NE
%left '<' '>' O_LTE O_GTE
%left O_SB_RANGE
%left O_LSH O_RSH
%left '+' '-'
%left '%' '*' '/' O_POWER
%right '!'
%nonassoc At
%left Field
%left '.' '[' '('

/* Non-terminal types
 * ------------------
 * In order for yacc to assign/access the correct field of $$, $1, we
 * must to declare which field is appropriate for the non-terminal.
 */
%type <nodeList> 	components
%type <node>		component tuple task function coordinator
%type <varList>		element_defs element_def definitions definition define input output
%type <idList>		names arguments
%type <expr>		expr field constant array_index function_call task_invocation create_obj 
%type <expr>		step_expr restrictions repeat_loop activation_command arg
%type <exprList>	args
%type <objInitArg>	obj_arg
%type <objInitArgList>	obj_args	
%type <type>		type scalar_type static_array static_type dynamic_type list dynamic_array
%type <intList>		static_dims
%type <id>		id section
%type <epoch>		epoch
%type <invokeArgs>	optional_sec
%type <invokeArgsList> 	optional_secs
%type <stmt>		stmt parallel_loop sequencial_loop if_else_block
%type <stmtList>	stmt_block code meta_code function_body
%type <condStmtList>	else_block
%type <condStmt>	else else_if
%type <rangeCond>	index_range
%type <rangeCondList>	index_ranges
%type <sloopAttr>       sloop_attr
%type <fnHeader>	in_out
%type <initInstr>	initialize
%type <linkageType>	mode
%type <envLinks>	linkage linkages
%type <envConfig>	environment
%type <repeat>		repeat_control
%type <stageHeader>	stage_header
%type <stage>		compute_stage
%type <stageList>	stage_sequence
%type <metaStage>	meta_stage
%type <metaStageList>	meta_stages
%type <compute>		compute
%type <order>		order
%type <pInstr>		instr ordered_instr
%type <pInstrList>	instr_list ordered_instr_list	
%type <pArg>		partition_arg
%type <pArgList>	partition_args
%type <sLink>		relativity divides
%type <booleanConstant>	nature dynamic
%type <iConList>	dimensions				
%type <vDims>		var
%type <vDimsList>	var_list
%type <dConSpecList>	data_spec main_dist data_sub_spec data_sub_dist
%type <subPartSpec>	sub_dist
%type <partSpec>	partition_spec
%type <partSpecList>	partition_specs
%type <partition>	partition

/*----------------------------------------------------- Grammer Rules ------------------------------------------------------------*/     
%%
program		: components					{@1; // this is needed to make bison set up
							     	     // the location variable yylloc
								  ProgramDef *program = new ProgramDef($1);
								  ProgramDef::program = program;				
								};
components	: component					{ ($$ = new List<Node*>)->Append($1); }
		| components component				{ ($$ = $1)->Append($2); };	
component	: tuple | task | function | coordinator		

/* ----------------------------------------------------- Tuple Definition -------------------------------------------------------- */
tuple 		: Tuple Type_Name ':' element_defs		{ $$ = new TupleDef(new Identifier(@2, $2), $4); };
element_defs	: element_def					{ ($$ = new List<VariableDef*>)->AppendAll($1); } 
		| element_defs element_def			{ ($$ = $1)->AppendAll($2); };
element_def	: names ':' static_type				{ $$ = VariableDef::DecomposeDefs($1, $3); };
names		: Variable_Name					{ ($$ = new List<Identifier*>)->Append(new Identifier(@1, $1)); }
		| names ',' Variable_Name			{ ($$ = $1)->Append(new Identifier(@3, $3)); };
static_type	: scalar_type | static_array			;
scalar_type	: T_Integer					{ $$ = Type::intType; }
		| T_Real Single					{ $$ = Type::floatType; }
		| T_Real Double					{ $$ = Type::doubleType; }
		| T_Character					{ $$ = Type::charType; }
		| T_Boolean					{ $$ = Type::boolType; }
		| T_Epoch					{ $$ = Type::epochType; }
		| T_Range					{ $$ = Type::rangeType; }
		| Type_Name					{ $$ = new NamedType(new Identifier(@1, $1)); };
static_array	: T_Array static_dims 
		  Of scalar_type				{ StaticArrayType *sa = new StaticArrayType(@1, $4, $2->NumElements());
								  sa->setLengths($2); 
								  $$ = sa; };
static_dims	: static_dims '['Integer']'			{ ($$ = $1)->Append($3); }
		| '['Integer']'					{ ($$ = new List<int>)->Append($2); };


/* ----------------------------------------------------- Task Definition -------------------------------------------------------- */
task 		: Task String ':' define environment 
		  initialize compute partition			{ $$ = new TaskDef(new Identifier(@2, $2), 
										new DefineSection($4, @4), $5, $6, $7, $8); };


/* ----------------------------------------------------- Define Section ---------------------------------------------------------- */
define		: S_Define ':' definitions			{ $$ = $3; };
definitions	: definition					{ ($$ = new List<VariableDef*>)->AppendAll($1); }
		| definitions definition			{ ($$ = $1)->AppendAll($2); };
definition	: names ':' type				{ $$ = VariableDef::DecomposeDefs($1, $3); };
type		: static_type | dynamic_type			;
dynamic_type	: list | dynamic_array				;
list		: T_List Of static_type				{ $$ = new ListType(@1, $3); };
dynamic_array	: Dimensionality T_Array Of 
		  static_type format				{ $$ = new ArrayType(@1, $4, $1); };
format		: | Format Type_Name				;


/* --------------------------------------------------- Environment Section -------------------------------------------------------- */
environment	: S_Environment ':' linkages			{ $$ = new EnvironmentConfig($3, @1); };
linkages 	: linkage 					{ $$ = $1; }
		| linkages linkage				{ ($$ = $1)->AppendAll($2); };
linkage		: names ':' mode				{ $$ = EnvironmentLink::decomposeLinks($1, $3); };
mode		: Link						{ $$ = TypeLink; }
		| Create					{ $$ = TypeCreate; }
		| Link_or_Create				{ $$ = TypeCreateIfNotLinked; };


/* ----------------------------------------------------- Initialize Section -------------------------------------------------------- */
initialize	:						{ $$ = NULL; } 
		| S_Initialize arguments ':' code		{ $$ = new InitializeInstr($2, $4, @1); };
arguments	: 						{ $$ = new List<Identifier*>; }
		| '(' names ')'					{ $$ = $2; }; 


/* ---------------------------------------------- Meta Compute Section ------------------------------------------------------------ */
compute		: S_Compute ':'	meta_stages			{ $$ = new ComputeSection($3, @1); };
meta_stages	: stage_sequence				{ ($$ = new List<MetaComputeStage*>)->Append(new MetaComputeStage($1, NULL)); }
		| meta_stage 					{ ($$ = new List<MetaComputeStage*>)->Append($1); }
		| meta_stage meta_stages			{ ($$ = new List<MetaComputeStage*>)->Append($1); $$->AppendAll($2); };
meta_stage	: stage_sequence repeat_control			{ $$ = new MetaComputeStage($1, $2); };
stage_sequence	: compute_stage					{ ($$ = new List<ComputeStage*>)->Append($1); } 
		| stage_sequence compute_stage			{ ($$ = $1)->Append($2); };
compute_stage	: stage_header '{' meta_stages '}'		{ $$ = new ComputeStage($1, $3); }
		| stage_header '{' code '}'			{ $$ = new ComputeStage($1, $3); };
stage_header	: String '(' T_Space Space_ID ')'
		  activation_command				{ $$ = new StageHeader(new Identifier(@1, $1), $4, $6); };
activation_command :						{ $$ = NULL; } 
		| Activate If Variable_Name In expr		{ $$ = new RangeExpr(new Identifier(@3, $3), $5, NULL, false, @1); };	 
repeat_control	: Repeat ':' From String repeat_loop 		{ $$ = new RepeatControl(new Identifier(@4, $4), $5, @1); };
repeat_loop	: For Variable_Name In expr step_expr		{ $$ = new RangeExpr(new Identifier(@2, $2), $4, $5, true,  @1); }
		| While expr					{ $$ = $2; }
		| Foreach T_Space Space_ID C_Sub_Partition	{ $$ = new SubpartitionRangeExpr($3, @1); };


/* ---------------------------------------------- Partition Section ---------------------------------------------------------------- */
partition	: S_Partition arguments	':' partition_specs	{ $$ = new PartitionSection($2, $4, @1); };
partition_specs	: partition_spec 				{ ($$ = new List<PartitionSpec*>)->Append($1); }
		| partition_specs partition_spec		{ ($$ = $1)->Append($2); };
partition_spec  : T_Space Space_ID '<' Dimensionality '>'
		  dynamic divides '{' main_dist sub_dist '}'	{ $$ = new PartitionSpec($2, $4, $9, $6, $7, $10, @1); }		
		| T_Space Space_ID '<' Unpartitioned '>' 
		  '{' names '}'					{ $$ = new PartitionSpec($2, $7, @1); };
dynamic		: 						{ $$ = false; }
		| '<' Dynamic '>'				{ $$ = true; };
divides		: 						{ $$ = NULL; }	
		| Divides T_Space Space_ID Partitions		{ $$ = new SpaceLinkage(LinkTypePartition, $3, @1); }
		| Divides T_Space Space_ID Sub_Partitions		{ $$ = new SpaceLinkage(LinkTypeSubpartition, $3, @1); };
main_dist	: data_spec					{ ($$ = new List<DataConfigurationSpec*>)->AppendAll($1); }
		| main_dist data_spec				{ ($$ = $1)->AppendAll($2); };
data_spec	: var_list ':' instr_list relativity		{ $$ = DataConfigurationSpec::decomposeDataConfig($1, $3, $4); };
var_list	: var						{ ($$ = new List<VarDimensions*>)->Append($1); }	 
		| var_list ',' var				{ ($$ = $1)->Append($3); };
var		: Variable_Name					{ $$ = new VarDimensions(new Identifier(@1, $1), NULL); }
		| Variable_Name '<' dimensions '>'		{ $$ = new VarDimensions(new Identifier(@1, $1), $3); };
dimensions	: V_Dimension					{ ($$ = new List<IntConstant*>)->Append(new IntConstant(@1, $1)); } 
		| dimensions ',' V_Dimension			{ ($$ = $1)->Append(new IntConstant(@3, $3)); };
instr_list	: instr						{ ($$ = new List<PartitionInstr*>)->Append($1); }	
		| instr_list ',' instr				{ ($$ = $1)->Append($3); };
instr		: Replicated					{ $$ = new PartitionInstr(@1); }
		| Variable_Name '(' partition_args ')' 		{ $$ = new PartitionInstr(
								    	new Identifier(@1, $1), $3, false, NULL, @1); }		
		| Variable_Name '(' partition_args ')' 		
		  Padding '(' partition_args ')'		{ $$ = new PartitionInstr(new Identifier(@1, $1), $3, true, $7, @1); };
partition_args	:						{ $$ = new List<PartitionArg*>; } 
		| partition_arg					{ ($$ = new List<PartitionArg*>)->Append($1); }
		| partition_args ',' partition_arg		{ ($$ = $1)->Append($3); };
partition_arg	: Variable_Name					{ $$ = new PartitionArg(new Identifier(@1, $1)); } 
		| Integer					{ $$ = new PartitionArg(new IntConstant(@1, $1)); };	 	
relativity	:						{ $$ = NULL; }
		| ';' Relative_To T_Space Space_ID  		{ $$ = new SpaceLinkage(LinkTypeUndefined, $4, @2); };			
sub_dist	:						{ $$ = NULL; }
		| P_Sub_Partition '<' Dimensionality'>'
		  '<' nature '>' '{' data_sub_dist '}'		{ $$ = new SubpartitionSpec($3, $6, $9, @1); };
nature		: Ordered 					{ $$ = true; }
		| Unordered					{ $$ = false; };
data_sub_dist	: data_sub_spec 				{ ($$ = new List<DataConfigurationSpec*>)->AppendAll($1); }
		| data_sub_dist data_sub_spec			{ ($$ = $1)->AppendAll($2); };
data_sub_spec	: var_list ':' ordered_instr_list		{ $$ = DataConfigurationSpec::decomposeDataConfig($1, $3, NULL);};
ordered_instr_list : ordered_instr				{ ($$ = new List<PartitionInstr*>)->Append($1); }
		| ordered_instr_list ',' ordered_instr		{ ($$ = $1)->Append($3); };
ordered_instr	: instr order					{ $$ = $1; $$->SetOrder($2); };	
order		: 						{ $$ = RandomOrder; }
		| Ascends 					{ $$ = AscendingOrder; }	
		| Descends					{ $$ = DescendingOrder; };			


/* --------------------------------------------------- Code Section ------------------------------------------------------------ */
code		: {BeginCode();}  stmt_block			{ EndCode(); $$ = $2; };
stmt_block	: stmt						{ ($$ = new List<Stmt*>)->Append($1); }
		| stmt new_lines				{ ($$ = new List<Stmt*>)->Append($1); } 
		| stmt new_lines stmt_block			{ ($$ = new List<Stmt*>)->Append($1); $$->AppendAll($3); };
new_lines	: New_Line | New_Line new_lines			; 
stmt		: parallel_loop 
		| sequencial_loop
		| if_else_block					
		| expr						{ $$ = $1; };
sequencial_loop : Do In Sequence '{' stmt_block '}'
                For id In sloop_attr                            { $$ = new SLoopStmt($8, $10, new StmtBlock($5), @1); };
sloop_attr      : field step_expr                               { $$ = new SLoopAttribute($1, $2, NULL); }
                | field O_AND expr                              { $$ = new SLoopAttribute($1, NULL, $3); };
parallel_loop	: Do '{' stmt_block '}' For index_ranges	{ $$ = new PLoopStmt($6, new StmtBlock($3), @1); }
		| Do '{' stmt_block '}' While expr		{ $$ = new WhileStmt($6, new StmtBlock($3), @1); };
index_ranges	: index_range					{ ($$ = new List<IndexRangeCondition*>)->Append($1); }	
		| index_ranges ';' index_range			{ ($$ = $1)->Append($3); };
index_range	: names In Variable_Name restrictions		{ $$ = new IndexRangeCondition($1, new Identifier(@3, $3), -1, $4, Join(@1, @4)); }
		| names In Variable_Name '.' 
		  Dimension_No restrictions		        { $$ = new IndexRangeCondition($1, new Identifier(@3, $3), $5, $6, Join(@1, @6)); };
restrictions	:						{ $$ = NULL; }	 
		| O_AND expr					{ $$ = $2; };
if_else_block	: If '(' expr ')' '{' stmt_block '}' else_block	{ List<ConditionalStmt*> *lca = new List<ConditionalStmt*>;
								  lca->Append(new ConditionalStmt($3, new StmtBlock($6), @1));
								  lca->AppendAll($8);	
								  $$ = new IfStmt(lca, @1); };
else_block	:						{ $$ = new List<ConditionalStmt*>; }
		| else						{ ($$ = new List<ConditionalStmt*>)->Append($1); }
		| else_if else_block				{ ($$ = new List<ConditionalStmt*>)->Append($1); $$->AppendAll($2); };
else		: Else '{' stmt_block '}'			{ $$ = new ConditionalStmt(NULL, new StmtBlock($3), @1); };
else_if		: Else If '(' expr ')' '{' stmt_block '}'	{ $$ = new ConditionalStmt($4, new StmtBlock($7), @1); };
	
step_expr	:						{ $$ = NULL; } 
		| Step expr					{ $$ = $2; };	
expr		: expr '+' expr					{ $$ = new ArithmaticExpr($1, ADD, $3, @2); }
		| expr '-' expr					{ $$ = new ArithmaticExpr($1, SUBTRACT, $3, @2); }
		| expr '*' expr					{ $$ = new ArithmaticExpr($1, MULTIPLY, $3, @2); }
		| expr '/' expr					{ $$ = new ArithmaticExpr($1, DIVIDE, $3, @2); }
		| expr '%' expr					{ $$ = new ArithmaticExpr($1, MODULUS, $3, @2); }
		| expr O_LSH expr				{ $$ = new ArithmaticExpr($1, LEFT_SHIFT, $3, @2); }
		| expr O_RSH expr				{ $$ = new ArithmaticExpr($1, RIGHT_SHIFT, $3, @2); }
		| expr O_POWER expr				{ $$ = new ArithmaticExpr($1, POWER, $3, @2); }
                | expr '&' expr                                 { $$ = new ArithmaticExpr($1, BITWISE_AND, $3, @2); }
                | expr '^' expr                                 { $$ = new ArithmaticExpr($1, BITWISE_XOR, $3, @2); }
                | expr '|' expr                                 { $$ = new ArithmaticExpr($1, BITWISE_OR, $3, @2); }
		| expr '<' expr					{ $$ = new LogicalExpr($1, LT, $3, @2); }
		| expr '>' expr					{ $$ = new LogicalExpr($1, GT, $3, @2); }
		| expr O_OR expr				{ $$ = new LogicalExpr($1, OR, $3, @2); }
		| expr O_AND expr				{ $$ = new LogicalExpr($1, AND, $3, @2); }
		| expr O_EQ expr				{ $$ = new LogicalExpr($1, EQ, $3, @2); }
		| expr O_NE expr				{ $$ = new LogicalExpr($1, NE, $3, @2); }
		| '!' expr					{ $$ = new LogicalExpr(NULL, NOT, $2, @1); }
		| expr O_GTE expr				{ $$ = new LogicalExpr($1, GTE, $3, @2); }
		| expr O_LTE expr				{ $$ = new LogicalExpr($1, LTE, $3, @2); }
		| expr '=' expr					{ $$ = new AssignmentExpr($1, $3, @2); }
		| REDUCE '(' String ',' expr ')'		{ $$ = new ReductionExpr($3, $5, @1); }
		| constant
		| field
		| function_call
		| task_invocation
		| create_obj
		| expr At '(' epoch ')'				{ $$ = new EpochExpr($1, $4); }				
		| '(' expr ')' 					{ $$ = $2; };
constant	: Integer					{ $$ = new IntConstant(@1, $1); } 
		| '-' Integer					{ $$ = new IntConstant(@1, $2 * -1); } 
		| Real_Single 					{ $$ = new FloatConstant(@1, $1); }
		| '-' Real_Single 				{ $$ = new FloatConstant(@1, $2 * -1); }
		| Real_Double 					{ $$ = new DoubleConstant(@1, $1); }
		| '-' Real_Double 				{ $$ = new DoubleConstant(@1, $2 * -1); }
		| Boolean 					{ $$ = new BoolConstant(@1, $1); }
		| Character					{ $$ = new CharacterConstant(@1, $1); };	
field		: id %prec Field				{ $$ = new FieldAccess(NULL, $1, @1); }
		| field '.' id					{ $$ = new FieldAccess($1, $3, @2); }
		| field '[' array_index ']'			{ $$ = new ArrayAccess($1, $3, Join(@2, @4)); };
array_index	: expr 
		| expr O_SB_RANGE expr				{ $$ = new SubRangeExpr($1, $3, Join(@1, @3)); } 
		| O_SB_RANGE				 	{ $$ = new SubRangeExpr(NULL, NULL, @1); };
function_call	: Variable_Name '(' args ')'			{ 
									Identifier *id = new Identifier(@1, $1);	
								  	if (LibraryFunction::isLibraryFunction(id)) {
										$$ = LibraryFunction::getFunctionExpr(id, $3, Join(@1, @4));	
								  	} else {
										$$ = new FunctionCall(id, $3, Join(@1, @4));
								  	} 
								};
arg		: String					{ $$ = new StringConstant(@1, $1); }
		| expr						{ $$ = $1; };
args		:						{ $$ = new List<Expr*>;} 
		| arg 						{ ($$ = new List<Expr*>)->Append($1); }
		| args ',' arg					{ ($$ = $1)->Append($3); };		
epoch		: Variable_Name					{ $$ = new EpochValue(new Identifier(@1, $1), 0); } 
		| Variable_Name '-' Integer			{ $$ = new EpochValue(new Identifier(@1, $1), $3); }; 
id		: Variable_Name 				{ $$ = new Identifier(@1, $1); }
		| Dimension_No 					{ $$ = new DimensionIdentifier(@1, $1); }
		| Range 					{ $$ = new Identifier(@1, Identifier::RangeId); }
		| Local						{ $$ = new Identifier(@1, Identifier::LocalId); } 
		| Index						{ $$ = new Identifier(@1, Identifier::IndexId); };


/* ----------------------------------------- Coordinator Program Definition--------------------------------------------------- */
coordinator	: Program {BeginProgram();} '(' Variable_Name ')' 
		  '{' meta_code '}'				{ $$ = new CoordinatorDef(new Identifier(@4, $4), $7, @1); };
meta_code	: stmt_block 					{ EndProgram(); $$ = $1; };
create_obj	: New dynamic_type				{ $$ = new ObjectCreate($2, new List<InitializerArg*>, @1); }
		| New static_type '(' obj_args ')'		{ $$ = new ObjectCreate($2, $4, @1); };
obj_args	:						{ $$ = new List<InitializerArg*>; }
		| obj_arg					{ ($$ = new List<InitializerArg*>)->Append($1); }
		| obj_args ',' obj_arg				{ ($$ = $1)->Append($3); };
obj_arg		: Variable_Name ':' String			{ $$ = new InitializerArg($1, new StringConstant(@3, $3), Join(@1, @3)); }
		| Variable_Name ':' expr			{ $$ = new InitializerArg($1, $3, Join(@1, @3)); };
task_invocation	: Execute '(' String ';' id optional_secs ')'	{ $$ = new TaskInvocation(new Identifier(@3, $3), $5, $6, @1); };
optional_secs	:						{ ($$ = new List<OptionalInvocationParams*>); } 
		| ';' optional_sec				{ ($$ = new List<OptionalInvocationParams*>)->Append($2); }
		| ';' optional_sec ';' optional_sec		{ $$ = new List<OptionalInvocationParams*>;
								  $$->Append($2); $$->Append($4);};
optional_sec	: section ':' args				{ $$ = new OptionalInvocationParams($1, $3, Join(@1, @3)); };
section		: S_Initialize 					{ $$ = new Identifier(@1, OptionalInvocationParams::InitializeSection); }
		| S_Partition					{ $$ = new Identifier(@1, OptionalInvocationParams::PartitionSection); };


/* ----------------------------------------------- Function Definition ------------------------------------------------------ */
function	: Function Variable_Name 
		  ':' in_out function_body			{ $$ = new FunctionDef(new Identifier(@2, $2), $4, $5); };
in_out		: output					{ $$ = new FunctionHeader(new List<VariableDef*>, $1); } 
		| input output					{ $$ = new FunctionHeader($1, $2); };	
input		: S_Arguments ':' definitions			{ $$ = $3; };
output		: S_Results ':' definitions			{ $$ = $3; };
function_body	: S_Compute ':' code				{ $$ = $3; };				 
					
		
%%

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

