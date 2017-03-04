/*------------------------------------------ Prolog -----------------------------------------------*/
%{
/* Just like lex, the text within this first region delimited by %{ and %}
 * is assumed to be C/C++ code and will be copied verbatim to the y.tab.c
 * file ahead of the definitions of the yyparse() function. Add other header
 * file inclusions or C++ variable declarations/prototypes that are needed
 * by your code here.
 */
#include "src/lex/scanner.h" // for yylex
#include "src/yacc/parser.h"
#include "src/common/errors.h"
#include "../common-libs/utils/utility.h"

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
	FunctionArg			*fnArg;
	List<FunctionArg*>		*fnArgList;
	FunctionDef			*fn;
	TupleDef			*tuple;	
	
	Node				*node;
	List<Node*>			*nodeList;
	Identifier			*id;
	List<Identifier*>		*idList;
	List<int>			*intList;
	List<const char*>		*strList;

	Type				*type;
	NamedType			*nType;
	ArrayType			*aType;
	ListType			*lType;

	Expr				*expr;
	NamedArgument			*namedArg;
	List<NamedArgument*>		*namedArgList;
	NamedMultiArgument		*namedMultArg;
	List<NamedMultiArgument*>	*namedMultArgList;
	List<Expr*>			*exprList;
	Stmt				*stmt;
	List<Stmt*>			*stmtList;
	ConditionalStmt			*condStmt;
	List<ConditionalStmt*>		*condStmtList;
	IndexRangeCondition		*rangeCond;
	SLoopAttribute                  *sloopAttr;
	List<IndexRangeCondition*>	*rangeCondList;

	DefineSection			*defineSection;	
	InitializeSection 		*initSection;
	LinkageType			linkageType;
	List<EnvironmentLink*>		*envLinks;
	EnvironmentSection		*envSection;
	StageDefinition			*stageDef;
	List<StageDefinition*>		*stageDefList;
	StagesSection			*stagesSection;

	RepeatControl			*repeatControl;
	FlowPart			*flowPart;
	List<FlowPart*>			*flowPartList;
	ComputationSection		*computeSection;

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
	PartitionSection		*partitionSection;
}

/* ---------------------------------------------------------------------------------------
 * Tokens Here we tell yacc about all the token types that we are using. Yacc will assign 
 * unique numbers to these and export the #define in the generated y.tab.h header file.
 */

/* High level entity type tokens */
%token Program Tuple Task Function Coordinator
/* object creation token */ 
%token New
/* task invocation token */ 
%token Execute
/* task sections identifier tokens */
%token S_Define S_Environment S_Initialize S_Stages S_Computation S_Partition
/* built in basic type tokens */
%token T_Integer T_Character T_Real T_Boolean T_String
/* other default scalar type tokens */
%token T_Index T_Range
/* collection type tokens */ 
%token T_Array T_List
/* collection element type specifier token*/
%token Of
/* additional type specifiers token */
%token Single Double Format Reduction
/* LPS indicator token */
%token T_Space
/* computation flow control tokens */
%token Repeat Epoch Where
/* looping construct tokens */
%token For Foreach While C_Sub_Partition
/* range iteration tokens */
%token In Step
/* epoch version access token */
%token Current
/* expression and statement related tokens */
%token If Else Range Local Index Do Sequence Reduce Return
/* extern code block importing tokens */
%token Extern Language Header_Includes Library_Links
/* environment linkage type tokens */
%token Link Create Link_or_Create
/* partition tokens */
%token Dynamic P_Sub_Partition Ordered Unordered Replicated Padding Relative_To 
%token Divides Sub_Partitions Partitions Unpartitioned Ascends Descends
/* newline separator token */
%token New_Line

/* typed tokens */
%token <intConstant> Integer Dimensionality Dimension_No V_Dimension
%token <characterConstant> Character Space_ID
%token <floatConstant> Real_Single 
%token <doubleConstant> Real_Double 
%token <booleanConstant> Boolean
%token <stringConstant> Type_Name Variable_Name String Native_Code

/* operator precedence order (bottom to top and left to write) */
%left ','
%right '=' 
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

/*----------------------------------------------------------------------------------------*/

/* Non-terminal types
 * ------------------
 * In order for yacc to assign/access the correct field of $$, $1, we
 * must to declare which field is appropriate for the non-terminal.
 */
%type <nodeList> 	components
%type <node>		component task coordinator

%type <tuple>		tuple
%type <defineSection>	define
%type <varList>		element_defs element_def definitions definition

%type <fn>		function	
%type <fnArg>		function_arg
%type <fnArgList>	function_args

%type <envLinks>	linkage linkages
%type <envSection>	environment

%type <initSection>	initialize
%type <idList>		names arguments

%type <stagesSection>	stages
%type <stageDefList>	stage_list
%type <stageDef>	compute_stage

%type <computeSection>	computation
%type <flowPartList>	compute_flow
%type <flowPart>	flow_part lps_transition repeat_cycle condition_block epoch_block stage_invoke	
%type <repeatControl>	repeat_control

%type <namedArg>	named_arg
%type <namedArgList>	named_args obj_args
%type <expr>		create_obj
%type <namedMultArg>	multi_arg
%type <namedMultArgList>multi_args	
%type <expr>		task_invocation
%type <exprList>	invoke_args
	
%type <expr>		expr field constant array_index function_call 
%type <expr>		step_expr restrictions arg
%type <exprList>	args

%type <type>		numeric_type scalar_type static_array static_type dynamic_type list dynamic_array
%type <intList>		static_dims
%type <id>		id
%type <intConstant>	epoch_lag

%type <stmt>		stmt parallel_loop sequencial_loop if_else_block extern_block reduction return_stmt
%type <stmtList>	stmt_block code meta_code
%type <condStmtList>	else_block
%type <condStmt>	else else_if
%type <rangeCond>	index_range
%type <rangeCondList>	index_ranges
%type <sloopAttr>       sloop_attr

%type <linkageType>	mode
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
%type <partitionSection>partition
%type <strList>		header_includes includes extern_links library_links

/*----------------------------------------------------- Grammer Rules ------------------------------------------------------------*/     
%%
program		: components					{@1; // this is needed to make bison set up
							     	     // the location variable yylloc
								  ProgramDef *program = new ProgramDef($1);
								  ProgramDef::program = program;				
								};
components	: component					{ ($$ = new List<Node*>)->Append($1); }
		| components component				{ ($$ = $1)->Append($2); };	
component	: tuple 					{ $$ = $1; }
		| task 
		| function					{ $$ = $1; } 
		| coordinator		

/* ----------------------------------------------------- Tuple Definition -------------------------------------------------------- */
tuple 		: Tuple Type_Name ':' element_defs		{ $$ = new TupleDef(new Identifier(@2, $2), $4); };
element_defs	: element_def					{ ($$ = new List<VariableDef*>)->AppendAll($1); } 
		| element_defs element_def			{ ($$ = $1)->AppendAll($2); };
element_def	: names ':' static_type				{ $$ = VariableDef::DecomposeDefs($1, $3); };
names		: Variable_Name					{ ($$ = new List<Identifier*>)->Append(new Identifier(@1, $1)); }
		| names ',' Variable_Name			{ ($$ = $1)->Append(new Identifier(@3, $3)); };
static_type	: scalar_type | static_array			;
scalar_type	: numeric_type					{ $$ = $1; }
		| T_String					{ $$ = Type::stringType; }
		| T_Range					{ $$ = Type::rangeType; }
		| Type_Name					{ $$ = new NamedType(new Identifier(@1, $1)); };
numeric_type	: T_Integer					{ $$ = Type::intType; }
		| T_Real Single					{ $$ = Type::floatType; }
		| T_Real Double					{ $$ = Type::doubleType; }
		| T_Character					{ $$ = Type::charType; }
		| T_Boolean					{ $$ = Type::boolType; };
static_array	: T_Array static_dims 
		  Of scalar_type				{ StaticArrayType *sa = new StaticArrayType(@1, $4, $2->NumElements());
								  sa->setLengths($2); 
								  $$ = sa; };
static_dims	: static_dims '['Integer']'			{ ($$ = $1)->Append($3); }
		| '['Integer']'					{ ($$ = new List<int>)->Append($2); };


/* ----------------------------------------------------- Task Definition -------------------------------------------------------- */
task 		: Task String ':' 
		  define 
                  environment 
		  initialize 
		  stages 
		  computation 
		  partition					{ $$ = new TaskDef(new Identifier(@2, $2), 
									$4, $5, $6, $7, $8, $9); };


/* ----------------------------------------------------- Define Section ---------------------------------------------------------- */
define		: S_Define ':' definitions			{ $$ = new DefineSection($3, @1); };
definitions	: definition					{ ($$ = new List<VariableDef*>)->AppendAll($1); }
		| definitions definition			{ ($$ = $1)->AppendAll($2); };
definition	: names ':' dynamic_type			{ $$ = VariableDef::DecomposeDefs($1, $3); }
		| names ':' static_type				{ $$ = VariableDef::DecomposeDefs($1, $3); }
		| names ':' numeric_type Reduction		{ $$ = VariableDef::DecomposeDefs($1, $3);
								  for (int i = 0; i < $$->NumElements(); i++) { 
								  	$$->Nth(i)->flagAsReduction();
								  } };
dynamic_type	: list | dynamic_array				;
list		: T_List Of static_type				{ $$ = new ListType(@1, $3); };
dynamic_array	: Dimensionality T_Array Of 
		  static_type format				{ $$ = new ArrayType(@1, $4, $1); };
format		: | Format Type_Name				;


/* --------------------------------------------------- Environment Section -------------------------------------------------------- */
environment	: S_Environment ':' linkages			{ $$ = new EnvironmentSection($3, @1); };
linkages 	: linkage 					{ $$ = $1; }
		| linkages linkage				{ ($$ = $1)->AppendAll($2); };
linkage		: names ':' mode				{ $$ = EnvironmentLink::decomposeLinks($1, $3); };
mode		: Link						{ $$ = TypeLink; }
		| Create					{ $$ = TypeCreate; }
		| Link_or_Create				{ $$ = TypeCreateIfNotLinked; };


/* ----------------------------------------------------- Initialize Section -------------------------------------------------------- */
initialize	:						{ $$ = NULL; } 
		| S_Initialize arguments ':' code		{ $$ = new InitializeSection($2, $4, @1); };
arguments	: 						{ $$ = new List<Identifier*>; }
		| '(' names ')'					{ $$ = $2; }; 

/* ------------------------------------------------------- Stages Section ---------------------------------------------------------- */
stages		: S_Stages ':' stage_list			{ $$ = new StagesSection($3, @1); };
stage_list	: compute_stage					{ ($$ = new List<StageDefinition*>)->Append($1); }
		| stage_list compute_stage			{ ($$ = $1)->Append($2); } ;
compute_stage	: Variable_Name '(' names ')' 
		  { BeginCode(); } '{' code '}'			{ $$ = new StageDefinition(new Identifier(@1, $1), $3, new StmtBlock($7)); } ;

/* ----------------------------------------------------- Computation Section ------------------------------------------------------- */

computation	: { FlowPart::resetFlowIndexRef(); } 
		  S_Computation ':' compute_flow		{ $$ = new ComputationSection($4, @2); };
compute_flow	: flow_part					{ ($$ = new List<FlowPart*>)->Append($1); }
		| compute_flow flow_part			{ ($$ = $1)->Append($2); };
flow_part 	: lps_transition
		| repeat_cycle
		| condition_block
		| epoch_block
		| stage_invoke					;
lps_transition	: T_Space Space_ID '{' compute_flow '}'		{ $$ = new LpsTransition($2, $4, Join(@1, @5)); };
repeat_cycle	: Repeat repeat_control 
		  '{' compute_flow '}' 				{ $$ = new RepeatCycle($2, $4, Join(@1, @5)); };
repeat_control	: For Variable_Name In expr step_expr		{ $$ =  new ForRepeat(new RangeExpr(
										new Identifier(@2, $2), $4, $5, Join(@2, @4)), @1); }	
		| Foreach C_Sub_Partition			{ $$ = new SubpartitionRepeat(Join(@1, @2)); }
		| While expr					{ $$ = new WhileRepeat($2, Join(@1, @2)); };
condition_block	: Where expr '{' compute_flow '}'		{ $$ = new ConditionalFlowBlock($2, $4, Join(@1, @5)); }
		| Where field In expr '{' compute_flow '}'	{ $$ = new ConditionalFlowBlock(new RangeExpr($2, $4, 
										Join(@2, @4)), $6, Join(@1, @7)); };
epoch_block	: Epoch '{' compute_flow '}'			{ $$ = new EpochBlock($3, Join(@1, @4)); };
stage_invoke	: Variable_Name '(' args ')'			{ $$ = new StageInvocation(new Identifier(@1, $1), $3, Join(@1, @4)); };

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
		| extern_block
		| return_stmt
		| reduction					 	 				
		| expr						{ $$ = $1; };
return_stmt	: Return expr					{ $$ = new ReturnStmt($2, Join(@1, @2)); };
reduction	: Reduce '(' Variable_Name 
			',' String ',' expr ')'          	{ $$ = new ReductionStmt(new Identifier(@3, $3), $5, $7, Join(@1, @8)); }	
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
		| '[' expr O_SB_RANGE expr ']'			{ $$ = new IndexRange($2, $4, false, Join(@1, @5)); } 
		| constant
		| field
		| function_call
		| task_invocation
		| create_obj
		| expr At '(' epoch_lag ')'			{ $$ = new EpochExpr($1, $4); }				
		| '(' expr ')' 					{ $$ = $2; };
constant	: Integer					{ $$ = new IntConstant(@1, $1); } 
		| '-' Integer					{ $$ = new IntConstant(@1, $2 * -1); } 
		| Real_Single 					{ $$ = new FloatConstant(@1, $1); }
		| '-' Real_Single 				{ $$ = new FloatConstant(@1, $2 * -1); }
		| Real_Double 					{ $$ = new DoubleConstant(@1, $1); }
		| '-' Real_Double 				{ $$ = new DoubleConstant(@1, $2 * -1); }
		| Boolean 					{ $$ = new BoolConstant(@1, $1); }
		| Character					{ $$ = new CharConstant(@1, $1); }	
		| String					{ $$ = new StringConstant(@1, $1); };
field		: id %prec Field				{ $$ = new FieldAccess(NULL, $1, @1); }
		| field '.' id					{ $$ = new FieldAccess($1, $3, @2); }
		| field '[' array_index ']'			{ $$ = new ArrayAccess($1, $3, Join(@2, @4)); };
array_index	: expr 
		| expr O_SB_RANGE expr				{ $$ = new IndexRange($1, $3, true, Join(@1, @3)); } 
		| O_SB_RANGE				 	{ $$ = new IndexRange(NULL, NULL, true, @1); };
function_call	: Variable_Name '(' args ')'			{ 
									Identifier *id = new Identifier(@1, $1);	
								  	if (LibraryFunction::isLibraryFunction(id)) {
										$$ = LibraryFunction::getFunctionExpr(id, $3, Join(@1, @4));	
								  	} else {
										$$ = new FunctionCall(id, $3, Join(@1, @4));
								  	} 
								};
arg		: T_Space Space_ID ':' Variable_Name		{ $$ = new ReductionVar($2, $4, Join(@1, @4)); }
		| expr						{ $$ = $1; };
args		:						{ $$ = new List<Expr*>;} 
		| arg 						{ ($$ = new List<Expr*>)->Append($1); }
		| args ',' arg					{ ($$ = $1)->Append($3); };		
epoch_lag	: Current					{ $$ = 0; } 
		| Current '-' Integer				{ $$ = $3; }; 
id		: Variable_Name 				{ $$ = new Identifier(@1, $1); }
		| Dimension_No 					{ $$ = new DimensionIdentifier(@1, $1); }
		| Range 					{ $$ = new Identifier(@1, Identifier::RangeId); }
		| Local						{ $$ = new Identifier(@1, Identifier::LocalId); } 
		| Index						{ $$ = new Identifier(@1, Identifier::IndexId); };


/* ----------------------------------------- Coordinator Program Definition--------------------------------------------------- */
coordinator	: Program {BeginProgram();} '(' Variable_Name ')' 
		  '{' meta_code '}'				{ $$ = new CoordinatorDef(new Identifier(@4, $4), $7, @1); };
meta_code	: stmt_block 					{ EndProgram(); $$ = $1; };
create_obj	: New dynamic_type				{ $$ = new ObjectCreate($2, new List<NamedArgument*>, @1); }
		| New static_type '(' obj_args ')'		{ $$ = new ObjectCreate($2, $4, @1); };
obj_args	:						{ $$ = new List<NamedArgument*>; }
		| named_args					;
named_args	: named_arg					{ ($$ = new List<NamedArgument*>)->Append($1); }
		| named_args named_arg				{ ($$ = $1)->Append($2); };
named_arg	: Variable_Name ':' expr			{ $$ = new NamedArgument($1, $3, Join(@1, @3)); };
task_invocation	: Execute '(' multi_args ')'			{ $$ = new TaskInvocation($3, Join(@1, @4)); };
multi_args	: multi_arg					{ ($$ = new List<NamedMultiArgument*>)->Append($1); }
		| multi_args ';' multi_arg			{ ($$ = $1)->Append($3); };
multi_arg	: Variable_Name ':' invoke_args			{ $$ = new NamedMultiArgument($1, $3, Join(@1, @3)); };
invoke_args	: expr						{ ($$ = new List<Expr*>)->Append($1); }
		| invoke_args ',' expr				{ ($$ = $1)->Append($3); };

/* ----------------------------------------------- Function Definition ------------------------------------------------------ */
function	: Function {BeginCode();} Variable_Name 
		  '(' function_args ')'
		  '{' code '}'					{ $$ = new FunctionDef(new Identifier(@3, $3), $5, new StmtBlock($8)); };
function_args	:						{ $$ = new List<FunctionArg*>; }
		| function_arg					{ ($$ = new List<FunctionArg*>)->Append($1); }
		| function_args ',' function_arg		{ ($$ = $1)->Append($3); };
function_arg	: '&' Variable_Name				{ $$ = new FunctionArg(new Identifier(@2, $2), REFERENCE_TYPE); }
		| Variable_Name					{ $$ = new FunctionArg(new Identifier(@1, $1), VALUE_TYPE);};				 
					
/* ----------------------------------------------- External Code Block ------------------------------------------------------ */
extern_block	: Extern '{'
		  Language String new_lines
		  header_includes extern_links	
		  Native_Code '}'				{ $$ = new ExternCodeBlock($4, $6, $7, $8, @1); };
header_includes : 						{ $$ = NULL; EndFreeString(); }
		| Header_Includes { BeginFreeString(); } 
		  '{' includes '}' new_lines			{ $$ = $4; EndFreeString(); };
includes	: String					{ ($$ = new List<const char*>)->Append($1); }
		| includes ',' String				{ ($$ = $1)->Append($3); };
extern_links	:						{ $$ = NULL; EndFreeString(); }
		| Library_Links { BeginFreeString(); } 
		  '{' library_links '}' new_lines 		{ $$ = $4; EndFreeString(); };
library_links	: String					{ ($$ = new List<const char*>)->Append($1); }
		| library_links ',' String			{ ($$ = $1)->Append($3); };
		
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

