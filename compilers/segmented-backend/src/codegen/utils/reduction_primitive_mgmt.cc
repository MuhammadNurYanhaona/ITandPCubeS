#include "reduction_primitive_mgmt.h"
#include "space_mapping.h"
#include "code_constant.h"

#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/domain-obj/constant.h"
#include "../../../../common-libs/utils/decorator_utils.h"

#include "../../../../frontend/src/syntax/ast_type.h"
#include "../../../../frontend/src/semantics/task_space.h"
#include "../../../../frontend/src/semantics/computation_flow.h"
#include "../../../../frontend/src/static-analysis/reduction_info.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

const char *getMpiDataTypeStr(Type *type, ReductionOperator op) {
	if (op == MAX_ENTRY || op == MIN_ENTRY) {
        	if (type == Type::intType) return strdup("MPI_2INT");
        	else if (type == Type::floatType) return strdup("MPI_FLOAT_INT");
        	else if (type == Type::doubleType) return strdup("MPI_DOUBLE_INT");
		else {
			std::cout << "Max/Min index reduction is only meaningful for numeric types\n";
			std::exit(EXIT_FAILURE);
		}
	}
	if (type == Type::charType) return strdup("MPI_CHAR");
	else if (type == Type::intType) return strdup("MPI_INT");
	else if (type == Type::floatType) return strdup("MPI_FLOAT");
	else if (type == Type::doubleType) return strdup("MPI_DOUBLE");
	else if (type == Type::boolType) return strdup("MPI_C_BOOL"); 	// not sure if this is the right type to use
	else {
		std::cout << "Reductions on non-built-in types is still not supported\n";
		std::exit(EXIT_FAILURE);
	}
}

const char *getMpiReductionOp(ReductionOperator op) {
	if (op == SUM) return strdup("MPI_SUM");
	if (op == PRODUCT) return strdup("MPI_PRODUCT");
	if (op == MAX) return strdup("MPI_MAX");
	if (op == MIN) return strdup("MPI_MIN");
	if (op == LAND) return strdup("MPI_LAND");
	if (op == LOR) return strdup("MPI_LOR");
	if (op == BAND) return strdup("MPI_BAND");
	if (op == BOR) return strdup("MPI_BOR");
	if (op == MAX_ENTRY) return strdup("MPI_MAXLOC");
	if (op == MIN_ENTRY) return strdup("MPI_MINLOC");
	return NULL;
}

const char *getReductionOpString(ReductionOperator op) {
	if (op == SUM) return strdup("SUM");
	if (op == PRODUCT) return strdup("PRODUCT");
	if (op == MAX) return strdup("MAX");
	if (op == MIN) return strdup("MIN");
	if (op == MAX_ENTRY) return strdup("MAX_ENTRY");
	if (op == MIN_ENTRY) return strdup("MIN_ENTRY");
	if (op == LAND) return strdup("LAND");
	if (op == LOR) return strdup("LOR");
	if (op == BAND) return strdup("BAND");
	if (op == BOR) return strdup("BOR");
	return NULL;
}

// I don't know how to make this ugly function definition any nicer; probably a better programmer will be able 
// to refactor it.
void generateResultResetFn(std::ofstream &programFile, 
                const char *initials, const char *className, Type *resultType, ReductionOperator op) {

	programFile << '\n' << "void " << initials << "::" << className << "::";	
	programFile << "resetPartialResult(reduction::Result *resultVar) {\n";
	if (op == SUM || op == PRODUCT) {
		std::string value = std::string("0");
		if (op == PRODUCT) value = std::string("1");
		programFile << indent;
		if (resultType == Type::charType) {
			programFile << "resultVar->data.charValue = " << value << stmtSeparator;
		} else if (resultType == Type::intType) {
			programFile << "resultVar->data.intValue = " << value << stmtSeparator;
		} else if (resultType == Type::floatType) {
			programFile << "resultVar->data.floatValue = " << value << stmtSeparator;
		} else if (resultType == Type::doubleType) {
			programFile << "resultVar->data.doubleValue = " << value << stmtSeparator;
		} else {
			std::cout << "Sum/Product reduction is not meaningful for type: ";
			std::cout << resultType->getName() << "\n";
			std::exit(EXIT_FAILURE);
		}
	} else if (op == MAX || op == MAX_ENTRY) {
		programFile << indent;
		if (resultType == Type::intType) {
			programFile << "resultVar->data.intValue = INT_MIN" << stmtSeparator;
		} else if (resultType == Type::floatType) {
			programFile << "resultVar->data.floatValue = -FLT_MAX" << stmtSeparator;
		} else if (resultType == Type::doubleType) {
			programFile << "resultVar->data.doubleValue = -DBL_MAX" << stmtSeparator;
		} else {
			std::cout << "MAX/MAX_ENTRY reduction is not meaningful for type: ";
			std::cout << resultType->getName() << "\n";
			std::exit(EXIT_FAILURE);
		}
	} else if (op == MIN || op == MIN_ENTRY) {
		programFile << indent;
		if (resultType == Type::intType) {
			programFile << "resultVar->data.intValue = INT_MAX" << stmtSeparator;
		} else if (resultType == Type::floatType) {
			programFile << "resultVar->data.floatValue = FLT_MAX" << stmtSeparator;
		} else if (resultType == Type::doubleType) {
			programFile << "resultVar->data.doubleValue = DBL_MAX" << stmtSeparator;
		} else {
			std::cout << "MIN/MIN_ENTRY reduction is not meaningful for type: ";
			std::cout << resultType->getName() << "\n";
			std::exit(EXIT_FAILURE);
		}
	} else if (op == LAND) {
		if (resultType == Type::boolType) {
			programFile << indent << "resultVar->data.boolValue = true" << stmtSeparator;
		} else {
			std::cout << "Logical AND reduction is only meaningful for boolean types\n";
			std::exit(EXIT_FAILURE);

		}
	} else if (op == LOR) {
		if (resultType == Type::boolType) {
			programFile << indent << "resultVar->data.boolValue = false" << stmtSeparator;
		} else {
			std::cout << "Logical OR reduction is only meaningful for boolean types\n";
			std::exit(EXIT_FAILURE);
		}
	} else if (op == BAND) {
		if (resultType == Type::charType) {
			programFile << indent << "resultVar->data.charValue = CHAR_MAX" << stmtSeparator;
		} else if (resultType == Type::intType) {
			programFile << indent << "resultVar->data.intValue = INT_MAX" << stmtSeparator;
		} else {
			std::cout << "Bitwise reduction is supported for integer and character types only\n";
			std::exit(EXIT_FAILURE);
		}
	} else if (op == BOR) {
		if (resultType == Type::charType) {
			programFile << indent << "resultVar->data.charValue = 0" << stmtSeparator;
		} else if (resultType == Type::intType) {
			programFile << indent << "resultVar->data.intValue = 0" << stmtSeparator;
		} else {
			std::cout << "Bitwise reduction is supported for integer and character types only\n";
			std::exit(EXIT_FAILURE);
		}
	}
	programFile << "}\n";
}

void generateUpdateCodeForMax(std::ofstream &programFile, std::string propertyName) {
	programFile << indent << "if (intermediateResult->data." << propertyName << " < ";
	programFile << "localPartialResult->data." << propertyName << ") {\n";
	programFile << doubleIndent << "intermediateResult->data." << propertyName;
	programFile << " = localPartialResult->data." << propertyName;
	programFile << stmtSeparator;	
	programFile << indent << "}\n";
}

void generateUpdateCodeForMaxEntry(std::ofstream &programFile, std::string propertyName) {
	programFile << indent << "if (intermediateResult->data." << propertyName << " < ";
	programFile << "localPartialResult->data." << propertyName << ") {\n";
	programFile << doubleIndent << "intermediateResult->data." << propertyName;
	programFile << " = localPartialResult->data." << propertyName;
	programFile << stmtSeparator;
	programFile << doubleIndent << "intermediateResult->index = localPartialResult->index" << stmtSeparator;	
	programFile << indent << "}\n";
}

void generateUpdateCodeForMin(std::ofstream &programFile, std::string propertyName) {
	programFile << indent << "if (intermediateResult->data." << propertyName << " > ";
	programFile << "localPartialResult->data." << propertyName << ") {\n";
	programFile << doubleIndent << "intermediateResult->data." << propertyName;
	programFile << " = localPartialResult->data." << propertyName;
	programFile << stmtSeparator;	
	programFile << indent << "}\n";
}

void generateUpdateCodeForMinEntry(std::ofstream &programFile, std::string propertyName) {
	programFile << indent << "if (intermediateResult->data." << propertyName << " > ";
	programFile << "localPartialResult->data." << propertyName << ") {\n";
	programFile << doubleIndent << "intermediateResult->data." << propertyName;
	programFile << " = localPartialResult->data." << propertyName;
	programFile << stmtSeparator;	
	programFile << doubleIndent << "intermediateResult->index = localPartialResult->index" << stmtSeparator;	
	programFile << indent << "}\n";
}

void generateUpdateCodeForSum(std::ofstream &programFile, std::string propertyName) {
	programFile << indent << "intermediateResult->data." << propertyName;
	programFile << " += localPartialResult->data." << propertyName;
	programFile << stmtSeparator;	
}

void generateUpdateCodeForProduct(std::ofstream &programFile, std::string propertyName) {
	programFile << indent << "intermediateResult->data." << propertyName;
	programFile << " *= localPartialResult->data." << propertyName;
	programFile << stmtSeparator;	
}

void generateUpdateCodeForLand(std::ofstream &programFile, std::string propertyName) {
	programFile << indent << "intermediateResult->data." << propertyName;
	programFile << " = intermediateResult->data." << propertyName;
	programFile << " && localPartialResult->data." << propertyName;
	programFile << stmtSeparator;	
}

void generateUpdateCodeForLor(std::ofstream &programFile, std::string propertyName) {
	programFile << indent << "intermediateResult->data." << propertyName;
	programFile << " = intermediateResult->data." << propertyName;
	programFile << " || localPartialResult->data." << propertyName;
	programFile << stmtSeparator;	
}

void generateUpdateCodeForBand(std::ofstream &programFile, std::string propertyName) {
	programFile << indent << "intermediateResult->data." << propertyName;
	programFile << " = intermediateResult->data." << propertyName;
	programFile << " & localPartialResult->data." << propertyName;
	programFile << stmtSeparator;	
}

void generateUpdateCodeForBor(std::ofstream &programFile, std::string propertyName) {
	programFile << indent << "intermediateResult->data." << propertyName;
	programFile << " = intermediateResult->data." << propertyName;
	programFile << " | localPartialResult->data." << propertyName;
	programFile << stmtSeparator;	
}

void generateIntermediateResultUpdateFnBody(std::ofstream &programFile, Type *varType, ReductionOperator op) {

	std::ostringstream propertyNameStr;
        propertyNameStr << varType->getCType() << "Value";
        std::string propertyName = propertyNameStr.str();

	if (op == MAX) 			generateUpdateCodeForMax(programFile, propertyName);
	else if (op == MAX_ENTRY)	generateUpdateCodeForMaxEntry(programFile, propertyName);
	else if (op == MIN)		generateUpdateCodeForMin(programFile, propertyName);
	else if (op == MIN_ENTRY)	generateUpdateCodeForMinEntry(programFile, propertyName);
	else if (op == SUM)		generateUpdateCodeForSum(programFile, propertyName);
	else if (op == PRODUCT)		generateUpdateCodeForProduct(programFile, propertyName);
	else if (op == LAND)		generateUpdateCodeForLand(programFile, propertyName);
	else if (op == LOR)		generateUpdateCodeForLor(programFile, propertyName);
	else if (op == BAND)		generateUpdateCodeForBand(programFile, propertyName);
	else if (op == BOR)		generateUpdateCodeForBor(programFile, propertyName);
	else {
		std::cout << "User defined reductions haven't been implemented yet";
                std::exit(EXIT_FAILURE);
	}
}

void generateCodeForDataReduction(std::ofstream &programFile, ReductionOperator op, Type *varType) {
	
	programFile << indent << "MPI_Comm mpiComm = segmentGroup->getCommunicator()" << stmtSeparator;
	programFile << indent << "int status = MPI_Allreduce(sendBuffer" << paramSeparator;
	programFile << paramIndent << indent;
	programFile << "receiveBuffer" << paramSeparator;
	programFile << paramIndent << indent;
	programFile << 1 << paramSeparator;

	const char *mpiDataTypeName = getMpiDataTypeStr(varType, op);
	programFile << paramIndent << indent;
	programFile << mpiDataTypeName << paramSeparator;

	const char *mpiReductionOp = getMpiReductionOp(op);
	programFile << paramIndent << indent;
	programFile << mpiReductionOp << paramSeparator;
	
	programFile << paramIndent << indent;
	programFile << "mpiComm)" << stmtSeparator;

	programFile << indent << "if (status != MPI_SUCCESS) {\n";
	programFile << doubleIndent << "std::cout << \"Reduction operation failed\\n\"" << stmtSeparator;
	programFile << doubleIndent << "std::exit(EXIT_FAILURE)" << stmtSeparator;
	programFile << indent << "}\n";
}

void generateIntraSegmentReductionPrimitive(std::ofstream &headerFile, 
                std::ofstream &programFile,
                const char *initials,
                ReductionMetadata *rdMetadata,
                Space *rootLps) {
	
	const char *resultVar = rdMetadata->getResultVar();
	
	std::ostringstream classNameStr;
	classNameStr << "ReductionPrimitive_" << resultVar;
	const char *className = strdup(classNameStr.str().c_str());

	// determine the superclass name the primitive should extend
	std::string superclassName = (rdMetadata->isSingleton()) 
			? "TaskGlobalReductionPrimitive" : "NonTaskGlobalReductionPrimitive";

	// generate a subclass of the intra-segment reduction primitive for the variable in the header file
	headerFile << "class " << className << " : public " << superclassName << " {\n";
	headerFile << "  public: \n";
	headerFile << indent << className << "(int localParticipants)" << stmtSeparator;
	headerFile << indent << "void resetPartialResult(reduction::Result *resultVar)" << stmtSeparator;
	headerFile << "  protected: \n";
	headerFile << indent << "void updateIntermediateResult(reduction::Result *localPartialResult)";
	headerFile << stmtSeparator;
	headerFile << "}" << stmtSeparator << '\n'; 

	// generate the definition of the constructor in the program file
	Type *exprType = rdMetadata->getExprType();
	ReductionOperator op = rdMetadata->getOpCode();
	const char *opStr = getReductionOpString(op);
	programFile << initials << "::" << className << "::" << className << "(";
	programFile << "int localParticipants)";
	programFile << paramIndent << ": " << superclassName << "(";
	programFile << "sizeof(" << exprType->getCType() << ")" << paramSeparator;
	programFile << opStr << paramSeparator << "localParticipants)";
	programFile << " {}\n"; 
	
	// generate the definition of the result reset function in the program file
	generateResultResetFn(programFile, initials, className, exprType, op);

	// generate the definition of intermediate result update function in the program file
	programFile << std::endl;
	programFile << "void " << initials << "::" << className << "::updateIntermediateResult(";
	programFile << paramIndent << "reduction::Result *localPartialResult) {\n";
	generateIntermediateResultUpdateFnBody(programFile, exprType, op);
	programFile << "}\n";
}

void generateCrossSegmentReductionPrimitive(std::ofstream &headerFile,
                std::ofstream &programFile, 
                const char *initials,
                ReductionMetadata *rdMetadata, 
                Space *rootLps) {
	
	const char *resultVar = rdMetadata->getResultVar();
	
	std::ostringstream classNameStr;
	classNameStr << "ReductionPrimitive_" << resultVar;
	const char *className = strdup(classNameStr.str().c_str());
	
	// determine the superclass name the primitive should extend
	std::string superclassName = (rdMetadata->isSingleton()) 
			? "TaskGlobalMpiReductionPrimitive" : "NonTaskGlobalMpiReductionPrimitive";

	// generate a subclass of the MPI reduction primitive for the variable in the header file
	headerFile << "class " << className << " : public " << superclassName << " {\n";
	headerFile << "  public: \n";
	headerFile << indent << className << "(int localParticipants" << paramSeparator;
	headerFile << "SegmentGroup *segmentGroup)" << stmtSeparator;
	headerFile << indent << "void resetPartialResult(reduction::Result *resultVar)" << stmtSeparator;
	headerFile << "  protected: \n";
	headerFile << indent << "void updateIntermediateResult(reduction::Result *localPartialResult)";
	headerFile << stmtSeparator;
	headerFile << indent << "void performCrossSegmentReduction()" << stmtSeparator;
	headerFile << "}" << stmtSeparator << '\n'; 

	// generate the definition of the constructor in the program file
	Type *exprType = rdMetadata->getExprType();
	ReductionOperator op = rdMetadata->getOpCode();
	const char *opStr = getReductionOpString(op);
	programFile << initials << "::" << className << "::" << className << "(";
	programFile << "int localParticipants" << paramSeparator;
	programFile << paramIndent << "SegmentGroup *segmentGroup)";
	programFile << paramIndent << ": " << superclassName << "(";
	programFile << "sizeof(" << exprType->getCType() << ")" << paramSeparator;
	programFile << opStr << paramSeparator;
	programFile << paramIndent << doubleIndent;
	programFile << "localParticipants" << paramSeparator << "segmentGroup)";
	programFile << " {}\n"; 

	// generate the definition of the result reset function in the program file
	generateResultResetFn(programFile, initials, className, exprType, op);

	// generate the definition of intermediate result update function in the program file
	programFile << std::endl;
	programFile << "void " << initials << "::" << className << "::updateIntermediateResult(";
	programFile << paramIndent << "reduction::Result *localPartialResult) {\n";
	generateIntermediateResultUpdateFnBody(programFile, exprType, op);
	programFile << "}\n";

	// generate the definition of terminal MPI reduction function in the program file 
	programFile << std::endl;
	programFile << "void " << initials << "::" << className << "::performCrossSegmentReduction() {\n";
	generateCodeForDataReduction(programFile, op, exprType);
	programFile << "}\n";
}

void generateReductionPrimitiveClasses(const char *headerFileName,
                const char *programFileName,
                const char *initials,
                MappingNode *mappingRoot,
                List<ReductionMetadata*> *reductionInfos) {
	
	if (reductionInfos->NumElements() == 0) return;

	std::cout << "\tGenerating reduction primitives and their management functions" << std::endl;
        std::ofstream programFile, headerFile;
        headerFile.open (headerFileName, std::ofstream::out | std::ofstream::app);
        programFile.open (programFileName, std::ofstream::out | std::ofstream::app);
        if (!programFile.is_open() || !headerFile.is_open()) {
                std::cout << "Unable to open program or header file";
                std::exit(EXIT_FAILURE);
        }

        const char *message = "Reduction Primitives Management";
        decorator::writeSectionHeader(headerFile, message);
        decorator::writeSectionHeader(programFile, message);

	Space *rootLps = mappingRoot->mappingConfig->LPS;
	int segmentedPpsId = rootLps->getSegmentedPPS();
	for (int i = 0; i < reductionInfos->NumElements(); i++) {
		ReductionMetadata *reduction = reductionInfos->Nth(i);
		
		std::ostringstream subheader;
		subheader << "Primitive for variable '" << reduction->getResultVar() << "'";
		const char *submessage = strdup(subheader.str().c_str());  
		decorator::writeSubsectionHeader(headerFile, submessage);
		headerFile << std::endl;
		decorator::writeSubsectionHeader(programFile, submessage);
		programFile << std::endl;

		Space *reductionRootLps = reduction->getReductionRootLps();
		int ppsId = reductionRootLps->getPpsId();
		
		// if the LPS for root of reduction range is mapped above the PPS where memory segmentation takes
		// place then create a cross-segment reduction primitive class that supports communication	
		if (ppsId > segmentedPpsId) {
			generateCrossSegmentReductionPrimitive(headerFile,
                			programFile, initials, reduction, rootLps);

		// otherwise, create an intra-segment reduction primitive class that uses synchronization only
		} else {
			generateIntraSegmentReductionPrimitive(headerFile,
                			programFile, initials, reduction, rootLps);
		}	
	}
       
	headerFile.close();
	programFile.close();
}

void generateReductionPrimitiveDecls(const char *headerFileName, List<ReductionMetadata*> *reductionInfos) {
	
	if (reductionInfos->NumElements() == 0) return;

	std::cout << "\tGenerating static pointers for reduction primitives" << std::endl;
        std::ofstream headerFile;
        headerFile.open (headerFileName, std::ofstream::out | std::ofstream::app);
        if (!headerFile.is_open()) {
                std::cout << "header file";
                std::exit(EXIT_FAILURE);
        }

	decorator::writeSubsectionHeader(headerFile, "Reduction Primitive Instances");
	headerFile << std::endl;

	for (int i = 0; i < reductionInfos->NumElements(); i++) {
		
		ReductionMetadata *reduction = reductionInfos->Nth(i);
		std::string classNamePrefix = (reduction->isSingleton()) ? "TaskGlobal" : "NonTaskGlobal";
		const char *varName = reduction->getResultVar();
		Space *reductionRootLps = reduction->getReductionRootLps();
		headerFile << "static " << classNamePrefix << "ReductionPrimitive *" << varName << "Reducer[";
		headerFile << "Space_" << reductionRootLps->getName() << "_Threads_Per_Segment]";
		headerFile << stmtSeparator;
	}

	headerFile.close();
}

void generateReductionPrimitiveInitFn(const char *headerFileName, 
                const char *programFileName, 
                const char *initials, 
                List<ReductionMetadata*> *reductionInfos) {
	
	if (reductionInfos->NumElements() == 0) return;

	std::cout << "\tGenerating function for reduction primitive instance initialization" << std::endl;
        std::ofstream programFile, headerFile;
        headerFile.open (headerFileName, std::ofstream::out | std::ofstream::app);
        programFile.open (programFileName, std::ofstream::out | std::ofstream::app);
        if (!programFile.is_open() || !headerFile.is_open()) {
                std::cout << "Unable to open program or header file";
                std::exit(EXIT_FAILURE);
        }
	
	const char *subHeader = "Reduction Primitive Initializer";
	decorator::writeSubsectionHeader(headerFile, subHeader);
	decorator::writeSubsectionHeader(programFile, subHeader);
	headerFile << std::endl;
	programFile << std::endl;

	headerFile << "void setupReductionPrimitives(std::ofstream &logFile)" << stmtSeparator;
	programFile << "void " << initials << "::setupReductionPrimitives(std::ofstream &logFile) {\n";

	programFile << std::endl;
	programFile << indent << "int segmentId " << paramSeparator << "segmentCount" << stmtSeparator;
	programFile << indent << "MPI_Comm_rank(MPI_COMM_WORLD" << paramSeparator;
	programFile << "&segmentId)" << stmtSeparator;
	programFile << indent << "MPI_Comm_size(MPI_COMM_WORLD" << paramSeparator;
	programFile << "&segmentCount)" << stmtSeparator;

	for (int i = 0; i < reductionInfos->NumElements(); i++) {
		
		programFile << std::endl;
		ReductionMetadata *reduction = reductionInfos->Nth(i);
		Space *reductionRootLps = reduction->getReductionRootLps();
		const char *varName = reduction->getResultVar();
		int segmentedPpsId = reductionRootLps->getSegmentedPPS();
		int reductionPpsId = reductionRootLps->getPpsId();
		const char *rdRootLpsName = reductionRootLps->getName();
		Space *reductionExecLps = reduction->getReductionExecutorLps();
		const char *rdExecLpsName = reductionExecLps->getName();

		std::ostringstream commentStream;
		commentStream << "Primitives for '" << varName << "'";
		decorator::writeCommentHeader(1, &programFile, commentStream.str().c_str());
		programFile << std::endl;

		programFile << indent << "{ // scope starts\n";

		// if the LPS for root of reduction range is mapped above the PPS where memory segmentation takes
		// place then we need a cross-segment reduction primitive for the result
		if (reductionPpsId > segmentedPpsId) {

			// determine how many segments should share a single reduction primitive
			programFile << indent << "int " << varName << "SegmentsPerPrim = ";
			programFile << "Max_Segments_Count / ";
			programFile << "Max_Space_" << rdRootLpsName << "_Threads" << stmtSeparator;

			// determine reduction primitives count
			programFile << indent << "int " << varName << "PrimitiveCount = ";
			programFile << "ceil(segmentCount * 1.0 / " << varName << "SegmentsPerPrim)";
			programFile << stmtSeparator;

			// iterate over all the primitives so that MPI groups and communicators can be created
			programFile << indent << "SegmentGroup *segmentGroup = NULL" << stmtSeparator;
			programFile << indent << "// a segment group communicator is only needed if there are ";
			programFile << "more than one segments\n";
			programFile << indent << "if (" << varName << "SegmentsPerPrim > 1) {\n"; 
			programFile << doubleIndent << "for (int i = 0; i < ";
			programFile << varName << "PrimitiveCount; i++) {\n";
			
			// participate in segment group creation for proper primitive; for others, signal non par-
			// ticipation status
			programFile << tripleIndent << "if(segmentId / " << varName;
			programFile << "SegmentsPerPrim == i) {\n";
			programFile << quadIndent << "segmentGroup = new SegmentGroup()";
			programFile << stmtSeparator << quadIndent;
			programFile << "segmentGroup->discoverGroupAndSetupCommunicator(logFile)";
			programFile << stmtSeparator;
			programFile << tripleIndent << "} else {\n";
			programFile << quadIndent << "SegmentGroup::excludeSegmentFromGroupSetup(";
			programFile << "segmentId" << paramSeparator << "logFile)" << stmtSeparator;
			programFile << tripleIndent << "}\n";
			programFile << doubleIndent << "}\n";
			programFile << indent << "}\n";

			// instantiate the static pointer for reduction primitive; there can be just one per segment
			programFile << indent << varName << "Reducer[0] = new ";
			programFile << "ReductionPrimitive_" << varName << "(";
			programFile << paramIndent << indent;
			programFile << "Space_" << rdExecLpsName << "_Threads_Per_Segment";
			programFile << paramSeparator << "segmentGroup)" << stmtSeparator;

			// setup log file reference in the reduction primitive 
			programFile << indent << varName << "Reducer[0]->setLogFile(&logFile)";
			programFile << stmtSeparator;

		// otherwise, one or more intra-segment reduction primtives will be needed for the groups of PPUs
		// rooted at the PPS the LPS has been mapped to.
		} else {
			// there might be multiple reduction primitives for the intra-segment case
			programFile << indent << "for(int i = 0; i < Space_";
			programFile << rdRootLpsName << "_Threads_Per_Segment; i++) {\n";
			programFile << doubleIndent << varName << "Reducer[i] = new ";
			programFile << "ReductionPrimitive_" << varName << "(";
			programFile << paramIndent << doubleIndent;
			programFile << "Space_" << rdExecLpsName << "_Threads_Per_Segment / ";
			programFile << "Space_" << rdRootLpsName  << "_Threads_Per_Segment";
			programFile << ")" << stmtSeparator; 
			
			// setup log file reference in the reduction primitive 
			programFile << doubleIndent << varName << "Reducer[i]->setLogFile(&logFile)";
			programFile << stmtSeparator;
			
			programFile << indent << "}\n";

		}
		
		programFile << indent << "} // scope ends\n";
	}
	
	programFile << "}\n";

	headerFile.close();
	programFile.close();	
}

void generateReductionPrimitiveMapCreateFnForThread(const char *headerFileName,
                const char *programFileName,
                const char *initials,
                List<ReductionMetadata*> *reductionInfos) {

	std::cout << "\tGenerating function for retrieving reduction primitives of a PPU controller" << std::endl;
        std::ofstream programFile, headerFile;
        headerFile.open (headerFileName, std::ofstream::out | std::ofstream::app);
        programFile.open (programFileName, std::ofstream::out | std::ofstream::app);
        if (!programFile.is_open() || !headerFile.is_open()) {
                std::cout << "Unable to open program or header file";
                std::exit(EXIT_FAILURE);
        }
	
	const char *subHeader = "Reduction Primitives Retriever";
	decorator::writeSubsectionHeader(headerFile, subHeader);
	decorator::writeSubsectionHeader(programFile, subHeader);
	headerFile << std::endl;
	programFile << std::endl;

	// generate the function signature in both header and the program files; the return type is a map of void
	// pointers as there are two types of reduction primitives without a common base class
	headerFile << "Hashtable<void*> *";
	programFile << "Hashtable<void*> *";
	programFile << initials << "::";
	headerFile << "getReductionPrimitiveMap(ThreadIds *threadIds)" << stmtSeparator;
	programFile << "getReductionPrimitiveMap(ThreadIds *threadIds) {\n\n";

	// instantiate a reduction primitive map
	programFile << indent << "Hashtable<void*> *rdPrimitiveMap = ";
	programFile << "new Hashtable<void*>" << stmtSeparator;

	// insert into the map one primitive for each reduction operation the PPU controller thread participate into
	for (int i = 0; i < reductionInfos->NumElements(); i++) {
		ReductionMetadata *reduction = reductionInfos->Nth(i);
		const char *varName = reduction->getResultVar();
		Space *reductionRootLps = reduction->getReductionRootLps();
		const char *rdRootLpsName = reductionRootLps->getName();
		Space *reductionExecLps = reduction->getReductionExecutorLps();
	 	const char *rdExecLpsName = reductionExecLps->getName();

		// if the PPU controller thread has a valid PPU ID in the reduction executor LPS only then it gets a
		// reduction primitive
		programFile << indent << "if(threadIds->ppuIds[Space_" << rdExecLpsName << "].id != INVALID_ID) ";
		programFile << "{\n";
		
		// the reduction primitive assignment happens based on the group ID at the reduction root LPS level
		programFile << doubleIndent << "int space" << rdRootLpsName << "Group = ";
		programFile << "threadIds->ppuIds[Space_" << rdRootLpsName << "].groupId % ";
		programFile << "Space_" << rdRootLpsName << "_Threads_Per_Segment";
		programFile << stmtSeparator;

		programFile << doubleIndent << "rdPrimitiveMap->Enter(\"" << varName << "\"" << paramSeparator;
		programFile << varName << "Reducer[space" << rdRootLpsName << "Group])" << stmtSeparator;
	
		programFile << indent << "}\n";
	}	
	
	// return the map
	programFile << indent << "return rdPrimitiveMap" << stmtSeparator;
	programFile << "}\n";

	headerFile.close();
	programFile.close();	
}


