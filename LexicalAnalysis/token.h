#define Program		258
#define Tuple		333			
#define Task		259
#define S_Define	260
#define S_Environment	271
#define S_Initialize	261
#define S_Compute	262
#define S_Partition	263

#define T_Integer	264
#define T_Character	265	
#define T_Real		266
#define T_Boolean	267
#define T_Epoch		268
#define T_Index		272
#define T_Range		273	
#define T_Array		320
#define T_List		321
#define A_Dimension	322	

#define Precision	269
#define P_Single	270
#define P_Double	274

#define Real		275
#define Integer		276
#define Boolean		277
#define Character	278	
#define String		279

#define Type_Name	334
#define Variable_Name	280

#define Space		281
#define Space_ID	297
#define S_Dimension	282
#define V_Dimension	283

#define R_ADD		284
#define R_MULT		285
#define R_MAX		286
#define R_MIN		287
#define R_MAX_ENTRY	288
#define R_MIN_ENTRY	289

#define O_LTE		290		
#define O_GTE		291	
#define O_EQ		292
#define O_NE		293
#define O_LSH		294
#define O_RSH		295
#define O_AND		296
#define O_OR		298
#define O_SB_RANGE	338

#define Activate	323
#define For		300
#define If		301
#define Repeat		302
#define Else		303
#define At		305
#define From		306
#define In		307
#define Step		308
#define Foreach		309
#define Range		310
#define Local_Range	311
#define Sub_Partition	312
#define While		313
#define Do		314
#define Sequence	315
#define To		316
#define Of		317		

#define Link		318
#define Create		319
#define Link_or_Create	324

#define P_Sub_Partition	325 
#define Ordered		326
#define Unordered	327	
#define Replicated	328
#define Padding		329
#define Divides		330
#define Sub_Partitions	331
#define Partitions	332
#define Unpartitioned	335
#define Ascends		336
#define Descends	337

#define New_Line	339

typedef union YYSTYPE { 
	int i; 
	char c; 
	double r;
	char *n;
	char *s;
} YYSTYPE;
YYSTYPE yylval;

