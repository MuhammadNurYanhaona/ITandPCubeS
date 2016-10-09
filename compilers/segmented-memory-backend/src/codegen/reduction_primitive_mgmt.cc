#include "reduction_primitive_mgmt.h"
#include "space_mapping.h"
#include "../syntax/ast_type.h"
#include "../semantics/task_space.h"
#include "../static-analysis/data_flow.h"
#include "../utils/list.h"
#include "../utils/decorator_utils.h"
#include "../utils/code_constant.h"
#include "../utils/common_constant.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

const char *getMpiDataTypeStr(Type *type, ReductionOperator op) {
	if (type == Type::charType) return strdup("MPI_CHAR");
	if (type == Type::intType) return strdup("MPI_INT");
	if (type == Type::floatType) return strdup("MPI_FLOAT");
	if (type == Type::doubleType) return strdup("MPI_DOUBLE");
	if (type == Type::boolType) return strdup("MPI_C_BOOL"); 	// not sure if this is the right type to use
	return NULL;
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
	if (op == MAX_ENTRY) return strdup("MAXLOC");
	if (op == MIN_ENTRY) return strdup("MINLOC");
	return NULL;
}

const char *getReductionOpString(ReductionOperator op) {
	if (op == SUM) return strdup("SUM");
	if (op == PRODUCT) return strdup("PRODUCT");
	if (op == MAX) return strdup("MAX");
	if (op == MIN) return strdup("MIN");
	if (op == AVG) return strdup("AVG");
	if (op == MAX_ENTRY) return strdup("MAX_ENTRY");
	if (op == MIN_ENTRY) return strdup("MIN_ENTRY");
	if (op == LAND) return strdup("LAND");
	if (op == LOR) return strdup("LOR");
	if (op == BAND) return strdup("BAND");
	if (op == BOR) return strdup("BOR");
	return NULL;
}

void generateUpdateCodeForMax(std::ofstream &programFile, Type *varType) {
	
	std::ostringstream propertyNameStr;
	propertyNameStr << varType->getCType() << "Value";
	std::string propertyName = propertyNameStr.str();

	programFile << indent << "if (intermediateResult->data." << propertyName << " < ";
	programFile << "localPartialResult->data." << propertyName << ") {\n";
	programFile << doubleIndent << "intermediateResult->data." << propertyName;
	programFile << " = localPartialResult->data." << propertyName;
	programFile << stmtSeparator;	
	programFile << indent << "}\n";
}

void generateUpdateCodeForSum(std::ofstream &programFile, Type *varType) {
	
	std::ostringstream propertyNameStr;
	propertyNameStr << varType->getCType() << "Value";
	std::string propertyName = propertyNameStr.str();

	programFile << indent << "intermediateResult->data." << propertyName;
	programFile << " += localPartialResult->data." << propertyName;
	programFile << stmtSeparator;	
}

void generateReductionCodeForMax(std::ofstream &programFile, Type *varType) {

	programFile << indent << "MPI_Comm mpiComm = segmentGroup->getCommunicator()" << stmtSeparator;
	programFile << indent << "int status = MPI_Allreduce(sendBuffer" << paramSeparator;
	programFile << paramIndent << indent;
	programFile << "receiveBuffer" << paramSeparator;
	programFile << paramIndent << indent;
	programFile << 1 << paramSeparator;
	programFile << paramIndent << indent;
	programFile << "DataType" << paramSeparator << "MPI_MAX" << paramSeparator;
	programFile << paramIndent << indent;
	programFile << "mpiComm)";
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

	// generate a subclass of the intra-segment reduction primitive for the variable in the header file
	headerFile << "class " << className << " : public ReductionPrimitive {\n";
	headerFile << "  public: \n";
	headerFile << indent << className << "(int localParticipants)" << stmtSeparator;
	headerFile << "  protected: \n";
	headerFile << indent << "void updateIntermediateResult(reduction::Result *localPartialResult)";
	headerFile << stmtSeparator;
	headerFile << "}" << stmtSeparator << '\n'; 

	// generate the definition of the constructor in the program file
	DataStructure *var = rootLps->getStructure(resultVar);
	Type *varType = var->getType();
	ReductionOperator op = rdMetadata->getOpCode();
	const char *opStr = getReductionOpString(op);
	programFile << initials << "::" << className << "::" << className << "(";
	programFile << "int localParticipants)";
	programFile << paramIndent << ": ReductionPrimitive(";
	programFile << "sizeof(" << varType->getCType() << ")" << paramSeparator;
	programFile << opStr << paramSeparator << "localParticipants)";
	programFile << " {}\n"; 
	
	// generate the definition of intermediate result update function in the program file
	programFile << std::endl;
	programFile << "void " << initials << "::" << className << "::updateIntermediateResult(";
	programFile << paramIndent << "reduction::Result *localPartialResult) {\n";
	if (op == MAX) {
		generateUpdateCodeForMax(programFile, varType);
	} else if (op == SUM) {
		generateUpdateCodeForSum(programFile, varType);
	}
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

	// generate a subclass of the MPI reduction primitive for the variable in the header file
	headerFile << "class " << className << " : public MpiReductionPrimitive {\n";
	headerFile << "  public: \n";
	headerFile << indent << className << "(int localParticipants" << paramSeparator;
	headerFile << "SegmentGroup *segmentGroup)" << stmtSeparator;
	headerFile << "  protected: \n";
	headerFile << indent << "void updateIntermediateResult(reduction::Result *localPartialResult)";
	headerFile << stmtSeparator;
	headerFile << indent << "void performCrossSegmentReduction()" << stmtSeparator;
	headerFile << "}" << stmtSeparator << '\n'; 

	// generate the definition of the constructor in the program file
	DataStructure *var = rootLps->getStructure(resultVar);
	Type *varType = var->getType();
	ReductionOperator op = rdMetadata->getOpCode();
	const char *opStr = getReductionOpString(op);
	programFile << initials << "::" << className << "::" << className << "(";
	programFile << "int localParticipants" << paramSeparator;
	programFile << paramIndent << "SegmentGroup *segmentGroup)";
	programFile << paramIndent << ": MpiReductionPrimitive(";
	programFile << "sizeof(" << varType->getCType() << ")" << paramSeparator;
	programFile << opStr << paramSeparator;
	programFile << paramIndent << doubleIndent;
	programFile << "localParticipants" << paramSeparator << "segmentGroup)";
	programFile << " {}\n"; 

	// generate the definition of intermediate result update function in the program file
	programFile << std::endl;
	programFile << "void " << initials << "::" << className << "::updateIntermediateResult(";
	programFile << paramIndent << "reduction::Result *localPartialResult) {\n";
	if (op == MAX) {
		generateUpdateCodeForMax(programFile, varType);
	} else if (op == SUM) {
		generateUpdateCodeForSum(programFile, varType);
	}
	programFile << "}\n";

	// generate the definition of terminal MPI reduction function in the program file 
	programFile << std::endl;
	programFile << "void " << initials << "::" << className << "::performCrossSegmentReduction() {\n";
	if (op == AVG || op == MIN_ENTRY || op == MAX_ENTRY) {
		std::cout << "Avg, Min-entry, and Max-entry cross segment reduction functions ";
		std::cout << "have not been implemented yet\n";
		std::exit(EXIT_FAILURE);
	} else {
		generateCodeForDataReduction(programFile, op, varType);
	}
	programFile << "}\n";
	
}

void generateReductionPrimitiveClasses(const char *headerFileName,
                const char *programFileName,
                const char *initials,
                MappingNode *mappingRoot,
                List<ReductionMetadata*> *reductionInfos) {
	
	if (reductionInfos->NumElements() == 0) return;

	std::cout << "Generating reduction primitives and their management functions" << std::endl;
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

	std::cout << "Generating static pointers for reduction primitives\n" << std::endl;
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
		const char *varName = reduction->getResultVar();
		Space *reductionRootLps = reduction->getReductionRootLps();
		headerFile << "static ReductionPrimitive *" << varName << "Reducer[";
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

	std::cout << "Generating function for reduction primitive instance initialization" << std::endl;
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

		// if the LPS for root of reduction range is mapped above the PPS where memory segmentation takes
		// place then we need a cross-segment reduction primitive for the result
		if (reductionPpsId > segmentedPpsId) {

			// determine how many segments should share a single reduction primitive
			programFile << indent << "int " << varName << "SegmentsPerPrim = ";
			programFile << "Max_Segments_Count / ";
			programFile << "Max_Space_" << rdRootLpsName << "_Threads" << stmtSeparator;

			// determine reduction primitives count
			programFile << indent << "int " << varName << "PrimitiveCount = ";
			programFile << "ceil(segmentCount / " << varName << "SegmentsPerPrim)";
			programFile << stmtSeparator;

			// iterate over all the primitives so that MPI groups and communicators can be created
			programFile << indent << "SegmentGroup *segmentGroup = NULL" << stmtSeparator;
			programFile << indent << "for (int i = 0; i < ";
			programFile << varName << "PrimitiveCount; i++) {\n";
			
			// participate in segment group creation for proper primitive; for others, signal non par-
			// ticipation status
			programFile << doubleIndent << "if(segmentId / " << varName;
			programFile << "SegmentsPerPrim == i) {\n";
			programFile << tripleIndent << "segmentGroup = new SegmentGroup()";
			programFile << stmtSeparator << tripleIndent;
			programFile << "segmentGroup->discoverGroupAndSetupCommunicator(logFile)";
			programFile << stmtSeparator;
			programFile << doubleIndent << "} else {\n";
			programFile << tripleIndent << "SegmentGroup::excludeSegmentFromGroupSetup(";
			programFile << "segmentId" << paramSeparator << "logFile)" << stmtSeparator;
			programFile << doubleIndent << "}\n";
			programFile << indent << "}\n";

			// instantiate the static pointer for reduction primitive; there can be just one per segment
			programFile << indent << varName << "Reducer[0] = new ";
			programFile << "ReductionPrimitive_" << varName << "(";
			programFile << paramIndent << indent;
			programFile << "Space_" << rdExecLpsName << "_Threads_Per_Segment";
			programFile << paramSeparator << "segmentGroup)" << stmtSeparator; 

		// otherwise, one or more intra-segment reduction primtives will be needed for the groups of PPUs
		// rooted at the PPS the LPS has been mapped to.
		} else {
			// there might be multiple reduction primitives for the intra-segment case
			programFile << indent << "for(int i = 0; i < Space_";
			programFile << rdRootLpsName << "_Threads_Per_Segment; i++) {\n";
			programFile << doubleIndent << varName << "Reducer[i] = new ";
			programFile << "ReductionPrimitive_" << varName << "(";
			programFile << "Space_" << rdExecLpsName << "_Threads_Per_Segment";
			programFile << ")" << stmtSeparator; 
			programFile << indent << "}\n";
		}
	}
	
	programFile << "}\n";

	headerFile.close();
	programFile.close();	
}


