.PHONY: clean

# Set the default target. When you make with no arguments, this will be the target built.
COMPILER = ficc
PRODUCTS = $(COMPILER) 
default: $(PRODUCTS)

# The backend C++ compiler should be different for environment; the default is mpicc
C_COMPILER=g++

# Set up the list of source and object files
SRCS = $(shell find src -name "*.cc")
LIB_SRCS = $(shell find ../common-libs ! -name "hashtable.cc" -name "*.cc")

# OBJS can deal with either .cc or .c files listed in SRCS
OBJS = y.tab.o lex.yy.o						\
	$(patsubst %.cc, %.o, $(filter %.cc,$(SRCS))) 		\
	$(patsubst %.c, %.o, $(filter %.c, $(SRCS)))		\
	$(patsubst %.c, %.o, $(filter %.cc, $(LIB_SRCS)))

SRC_JUNK = src/*.o src/*/*.o 
LIB_JUNK = ../common-libs/*.o ../common-libs/*/*.o
TOOL_JUNK= *.o lex.yy.c dpp.yy.c y.tab.c y.tab.h *.core core y.output

# Define the tools we are going to use
CC= $(C_COMPILER)
LD = $(C_COMPILER)
LEX = flex
YACC = bison

#--------------------------------------------------------------------------------------------------
# Set up the necessary flags for the tools
#--------------------------------------------------------------------------------------------------
# settings for debug mode code generation
#CFLAGS= -g -Wall -fno-inline -Wno-unused -Wno-sign-compare -O0
# settings for optimized code generation
CFLAGS = -O3

# The -d flag tells lex to set up for debugging. Can turn on/off by
# setting value of global yy_flex_debug inside the scanner itself
LEXFLAGS = -d

# The -d flag tells yacc to generate header with token types
# The -v flag writes out a verbose description of the states and conflicts
# The -t flag turns on debugging capability
# The -y flag means imitate yacc's output file naming conventions
YACCFLAGS = -dvty
#--------------------------------------------------------------------------------------------------

# Link with math library and lex library
LIBS = -lc -lm

#--------------------------------------------------------------------------------------------------
# Rules for various parts of the target
#--------------------------------------------------------------------------------------------------
.yy.o: $*.yy.c
	$(CC) $(CFLAGS) -c -o $@ $*.cc

lex.yy.c: IT.l IT.y y.tab.h 
	$(LEX) $(LEXFLAGS) IT.l

y.tab.o: y.tab.c
	$(CC) $(CFLAGS) -c -o y.tab.o y.tab.c

y.tab.h y.tab.c: IT.y
	$(YACC) $(YACCFLAGS) IT.y

.cc.o: $*.cc
	$(CC) $(CFLAGS) -c -o $@ $*.cc
#--------------------------------------------------------------------------------------------------

# rule to build the compiler
$(COMPILER) : $(OBJS)
	$(LD) -o $@ $(OBJS) $(LIBS)
	rm -f $(TOOL_JUNK) 

clean:
	rm -f $(SRC_JUNK) 
	rm -f $(LIB_JUNK)
	rm -f $(TOOL_JUNK)
	rm -f $(PRODUCTS)
	
