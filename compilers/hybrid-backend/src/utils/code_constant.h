#ifndef _H_code_constant
#define _H_code_constant

#include <string>

static const char *stmtSeparator = ";\n";
static const char *stmtTerminator = ";";
static const char *paramSeparator = ", ";
static const char *indent = "\t";
static const char *doubleIndent = "\t\t";
static const char *tripleIndent = "\t\t\t";
static const char *quadIndent = "\t\t\t\t";

static const char *paramIndent = "\n\t\t";
static const char *stmtIndent = "\t";

// How to access variables inside a GPU kernel varies depending on the GPU PPS executing the underlying code.
// This class holds a set of strings that denote what adjustment need to be done to variable references based
// on the PPS executing the code.	 
class GpuCodeConstants {
  public:
	// this represents the suffix to be added for shared memory variable (or array) declaration
	std::string storageSuffix;
	// this represents the suffix to be added when accessing a shared memory variable
        std::string storageIndex;
	// this represents the expression to be added to the initialization expression of for loops that 
	// distribute top level LPUs copied into the GPU card memory from the host
        std::string distrIndex;
	// this represents the expression to be added to the initialization expression of the for loops that
	// are generated inside GPU kernel to distribute LPUs
	std::string localDistrIndex;
	// this represents the expression to be used to increment for loop indices for staged in LPUs
        std::string jumpExpr;
	// this represents the expression to be used to increment for loop indices for LPUs generated inside
	// the kernel
        std::string localJumpExpr;
  public:
	static GpuCodeConstants *getConstantsForGpuLevel();  
	static GpuCodeConstants *getConstantsForSmLevel();  
	static GpuCodeConstants *getConstantsForWarpLevel();  
}; 


#endif
