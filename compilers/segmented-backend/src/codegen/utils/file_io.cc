#include "file_io.h"
#include "code_constant.h"

#include "../../../../frontend/src/syntax/ast_type.h"
#include "../../../../frontend/src/syntax/ast_task.h"
#include "../../../../frontend/src/semantics/task_space.h"

#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/utils/string_utils.h"
#include "../../../../common-libs/utils/decorator_utils.h"

#include <sstream>
#include <fstream>
#include <iostream>
#include <deque>
#include <cstdlib>

void generatePartReaderForStructure(std::ofstream &headerFile, ArrayDataStructure *array) {

	Space *lps = array->getSpace();
	const char *lpsName = lps->getName();
	ArrayType *arrayType = (ArrayType*) array->getType();
	Type *elementType = arrayType->getTerminalElementType();
	const char *varName = array->getName();

	headerFile << std::endl << "class " << varName << "InSpace" << lpsName << "Reader ";
	headerFile << ": public PartReader {\n";
	
	// a part writer class needs a typed stream to read its data
	headerFile << "  protected:\n";
	headerFile << indent << "TypedInputStream<" << elementType->getCType() << "> *stream" << stmtSeparator;

	// write the constructor for the class
	headerFile << "  public:\n";
	headerFile << indent << varName << "InSpace" << lpsName << "Reader(";
	headerFile << "DataPartitionConfig *partConfig" << paramSeparator << "DataPartsList *partsList)\n" ;
	headerFile << indent << doubleIndent << ": PartReader(";
	headerFile << "partsList" << paramSeparator << "partConfig) {\n";
	headerFile << doubleIndent << "this->partConfig = partConfig" << stmtSeparator;
	headerFile << doubleIndent << "this->stream = NULL" << stmtSeparator;
	headerFile << indent << "}\n"; 

	// write implementations for the three functions needed to do structure specific reading
	headerFile << indent << "void begin() {\n";
	headerFile << doubleIndent << "stream = new TypedInputStream<" << elementType->getCType() << ">";
	headerFile << "(fileName)" << stmtSeparator;
	headerFile << doubleIndent << "Assert(stream != NULL)" << stmtSeparator;  
	headerFile << doubleIndent << "stream->open()" << stmtSeparator;
	headerFile << indent << "}\n";
	headerFile << indent << "void terminate() {\n";
	headerFile << doubleIndent << "stream->close()" << stmtSeparator;
	headerFile << doubleIndent << "delete stream" << stmtSeparator;
	headerFile << indent << "}\n";
	headerFile << indent << "List<int> *getDataIndex(List<int> *partIndex) {";
	if (array->isReordered(lps->getRoot())) {
		headerFile << " return PartHandler::getDataIndex(partIndex)" << stmtTerminator << " }\n";
	} else {
		headerFile << " return partIndex" << stmtTerminator << " }\n";
	}
	headerFile << indent << "void readElement(List<int> *dataIndex, long int storeIndex, void *partStore) {\n";
	headerFile << doubleIndent << elementType->getCType();
	headerFile << " *dataStore = (" << elementType->getCType() << "*) partStore" << stmtSeparator;
	headerFile << doubleIndent << "dataStore[storeIndex] = stream->readElement(dataIndex)" << stmtSeparator;
	headerFile << indent <<  "}\n";

	// if the data item is multi-versioned then we need to implement another function to ensure that all versions
	// of each data part are at sync after a file read (note that the default version count is 0)
	int versionCount = array->getLocalVersionCount();
	if (versionCount > 0) {
		headerFile << indent << "void postProcessPart(DataPart *dataPart) {\n";
		headerFile << doubleIndent << "dataPart->synchronizeAllVersions()" << stmtSeparator;
		headerFile << indent <<  "}\n";
	}

	headerFile << "}" << stmtSeparator;
}

void generatePartWriterForStructure(std::ofstream &headerFile, ArrayDataStructure *array) {
	
	Space *lps = array->getSpace();
	const char *lpsName = lps->getName();
	ArrayType *arrayType = (ArrayType*) array->getType();
	Type *elementType = arrayType->getTerminalElementType();
	const char *varName = array->getName();

	headerFile << std::endl << "class " << varName << "InSpace" << lpsName << "Writer ";
	headerFile << ": public PartWriter {\n";
	
	// a part writer class needs a typed stream to write its data
	headerFile << "  protected:\n";
	headerFile << indent << "TypedOutputStream<" << elementType->getCType() << "> *stream" << stmtSeparator;

	// write the constructor for the class
	headerFile << "  public:\n";
	headerFile << indent << varName << "InSpace" << lpsName << "Writer(";
	headerFile << "DataPartitionConfig *partConfig" << paramSeparator << "int writerId" << paramSeparator;
	headerFile << "DataPartsList *partsList)\n" ;
	headerFile << indent << doubleIndent << ": PartWriter(writerId" << paramSeparator;
	headerFile << "partsList" << paramSeparator << "partConfig) {\n";
	headerFile << doubleIndent << "this->partConfig = partConfig" << stmtSeparator;
	headerFile << doubleIndent << "this->stream = NULL" << stmtSeparator;
	if (array->doesGenerateOverlappingParts()) {
		headerFile << doubleIndent << "setNeedToExcludePadding(true)" << stmtSeparator;
	}
	headerFile << indent << "}\n"; 

	// write implementations for the four functions needed to do structure specific writing
	headerFile << indent << "void begin() {\n";
	headerFile << doubleIndent << "stream = new TypedOutputStream<" << elementType->getCType() << ">";
	headerFile << "(fileName" << paramSeparator << "getDimensionList()" << paramSeparator;
	headerFile << "writerId == 0)" << stmtSeparator;
	headerFile << doubleIndent << "Assert(stream != NULL)" << stmtSeparator;  
	headerFile << doubleIndent << "stream->open()" << stmtSeparator;
	headerFile << indent << "}\n";
	headerFile << indent << "void terminate() {\n";
	headerFile << doubleIndent << "stream->close()" << stmtSeparator;
	headerFile << doubleIndent << "delete stream" << stmtSeparator;
	headerFile << indent << "}\n";
	headerFile << indent << "List<int> *getDataIndex(List<int> *partIndex) {";
	if (array->isReordered(lps->getRoot())) {
		headerFile << " return PartHandler::getDataIndex(partIndex)" << stmtTerminator << " }\n";
	} else {
		headerFile << " return partIndex" << stmtTerminator << " }\n";
	}
	headerFile << indent << "void writeElement(List<int> *dataIndex, long int storeIndex, void *partStore) {\n";
	headerFile << doubleIndent << elementType->getCType();
	headerFile << " *dataStore = (" << elementType->getCType() << "*) partStore" << stmtSeparator;
	headerFile << doubleIndent << "stream->writeElement(dataStore[storeIndex]" <<  paramSeparator;
	headerFile << "dataIndex)" << stmtSeparator;
	headerFile << indent <<  "}\n";

	headerFile << "}" << stmtSeparator;
}

void generateReaderWriterForLpsStructures(std::ofstream &headerFile, 
		const char *initials, Space *lps, List<const char*> *envVariables) {
	
	std::ostringstream header;
	const char *lpsName = lps->getName();
	header << "Space " << lpsName;
	decorator::writeSubsectionHeader(headerFile, header.str().c_str());

	List<const char*> *localStructures = lps->getLocalDataStructureNames();
	for (int i = 0; i < localStructures->NumElements(); i++) {

		// reader-writer classes are needed only for environmental variables
		const char *varName = localStructures->Nth(i);
		if (!string_utils::contains(envVariables, varName)) continue;
		
		// reader and writers are generated for a data structure only if it has been allocated in the current LPS
		DataStructure *structure = lps->getLocalStructure(varName);
		if (!structure->getUsageStat()->isAllocated()) continue;

		// currently we only read/write arrays from/to files
		ArrayDataStructure *array = dynamic_cast<ArrayDataStructure*>(structure);
		if (array == NULL) continue;

		generatePartReaderForStructure(headerFile, array);
		generatePartWriterForStructure(headerFile, array);
	}
}

void generateReaderWriters(const char *headerFileName, const char *initials, TaskDef *taskDef) {
       
	std::cout << "Generating structures and functions for file IO\n";
	 
	Space *rootLps = taskDef->getPartitionHierarchy()->getRootSpace();
	List<const char*> *envVariables = new List<const char*>;
	List<EnvironmentLink*> *envLinkList = taskDef->getEnvironmentLinks();
        for (int i = 0; i < envLinkList->NumElements(); i++) {
                EnvironmentLink *link = envLinkList->Nth(i);
                const char *varName = link->getVariable()->getName();
		envVariables->Append(varName);
	}

	std::ofstream headerFile;
        headerFile.open (headerFileName, std::ofstream::out | std::ofstream::app);
        if (!headerFile.is_open()) {
                std::cout << "Unable to open header file";
                std::exit(EXIT_FAILURE);
        }

        const char *message = "data structure spacific part reader and writer subclasses";
        decorator::writeSectionHeader(headerFile, message);

        std::deque<Space*> lpsQueue;
	List<Space*> *childrenLpses = rootLps->getChildrenSpaces();
        for (int i = 0; i < childrenLpses->NumElements(); i++) {
                lpsQueue.push_back(childrenLpses->Nth(i));
        }
	 while (!lpsQueue.empty()) {
                Space *lps = lpsQueue.front();
                lpsQueue.pop_front();
		childrenLpses = lps->getChildrenSpaces();
        	for (int i = 0; i < childrenLpses->NumElements(); i++) {
                	lpsQueue.push_back(childrenLpses->Nth(i));
		}
		if (lps->getSubpartition() != NULL) lpsQueue.push_back(lps->getSubpartition());
		if (lps->allocateStructures()) {
			generateReaderWriterForLpsStructures(headerFile, initials, lps, envVariables);
		}
	}

	headerFile.close();
}

void generateWritersMap(const char *headerFileName,  const char *programFileName, 
                const char *initials, TaskDef *taskDef) {
	
	std::ofstream programFile;
	std::ofstream headerFile;
        programFile.open (programFileName, std::ofstream::out | std::ofstream::app);
        headerFile.open (headerFileName, std::ofstream::out | std::ofstream::app);
        if (!programFile.is_open() || !headerFile.is_open()) {
                std::cout << "Unable to open header/program file";
                std::exit(EXIT_FAILURE);
        }

	const char *message = "Parts-writer map generator";
        decorator::writeSubsectionHeader(headerFile, message);
        decorator::writeSubsectionHeader(programFile, message);

	// generate function header
	std::ostringstream fnHeader;
	fnHeader << "generateWritersMap(TaskData *taskData" << paramSeparator;
	fnHeader << std::endl << indent << doubleIndent;
	fnHeader << "SegmentState *segment" << paramSeparator;
	fnHeader << std::endl << indent << doubleIndent;
	fnHeader  << "Hashtable<DataPartitionConfig*> *partConfigMap)";
	headerFile << "Hashtable<PartWriter*> *" << fnHeader.str() << stmtSeparator;
	programFile << "Hashtable<PartWriter*> *" << initials << "::" << fnHeader.str();

	// retrieve the names of all environmental variables to generate file writers for them
	Space *rootLps = taskDef->getPartitionHierarchy()->getRootSpace();
	List<const char*> *envVariables = new List<const char*>;
	List<EnvironmentLink*> *envLinkList = taskDef->getEnvironmentLinks();
        for (int i = 0; i < envLinkList->NumElements(); i++) {
                EnvironmentLink *link = envLinkList->Nth(i);
                const char *varName = link->getVariable()->getName();
		envVariables->Append(varName);
	}

	// start function body
	programFile << " {\n\n";

	// instantiate the writer map
	programFile << indent << "Hashtable<PartWriter*> *writersMap = new Hashtable<PartWriter*>" << stmtSeparator;
	programFile << '\n';

	// determine current segment's writer ID and the total number of active writers 
	programFile << indent << "int writerId" << stmtSeparator;
	programFile << indent << "MPI_Comm_rank(MPI_COMM_WORLD, &writerId)" << stmtSeparator;
        programFile << indent << "int mpiProcessCount" << stmtSeparator;
        programFile << indent << "MPI_Comm_size(MPI_COMM_WORLD" << paramSeparator;
        programFile << "&mpiProcessCount)" << stmtSeparator;
        programFile << indent << "int writersCount = min(mpiProcessCount" << paramSeparator;
        programFile << "Max_Segments_Count)" << stmtSeparator;
	
        std::deque<Space*> lpsQueue;
	lpsQueue.push_back(rootLps);
	while (!lpsQueue.empty()) {
                Space *lps = lpsQueue.front();
                lpsQueue.pop_front();
		List<Space*> *childrenLpses = lps->getChildrenSpaces();
        	for (int i = 0; i < childrenLpses->NumElements(); i++) {
                	lpsQueue.push_back(childrenLpses->Nth(i));
		}
		if (lps->getSubpartition() != NULL) lpsQueue.push_back(lps->getSubpartition());
		
		const char *lpsName = lps->getName();
		bool writersFound = false;
		List<const char*> *localStructures = lps->getLocalDataStructureNames();
		for (int i = 0; i < localStructures->NumElements(); i++) {

			// writer classes are needed only for environmental variables
			const char *varName = localStructures->Nth(i);
			if (!string_utils::contains(envVariables, varName)) continue;
			
			// writers are generated for a data structure only if it has been allocated in the current LPS
			DataStructure *structure = lps->getLocalStructure(varName);
			if (!structure->getUsageStat()->isAllocated()) continue;

			// currently we only write arrays to files
			ArrayDataStructure *array = dynamic_cast<ArrayDataStructure*>(structure);
			if (array == NULL) continue;

			if (!writersFound) {
				programFile << "\n" << indent << "// writers for Space " << lpsName << "\n";
				writersFound = true;
			}

			// retrieve the data item and add a checking that says if a writer should be created
			std::ostringstream dataItemName;
			dataItemName << varName << "InSpace" << lpsName;
			programFile << indent << "DataItems *" << dataItemName.str(); 
			programFile << " = taskData->getDataItemsOfLps(\"" << lpsName << "\"" << paramSeparator;
			programFile << "\"" << varName << "\")" << stmtSeparator;
			programFile << indent << "if (" << dataItemName.str() << " != NULL";
			programFile << " && !" << dataItemName.str() << "->isEmpty()";
			programFile << " && segment->computeStagesInLps(Space_" << lpsName << ")";
			programFile << ") {\n";

			std::ostringstream configName;
			configName << varName << "Space" << lpsName << "Config";
			programFile << doubleIndent << "DataPartitionConfig *config" << " = partConfigMap->Lookup";
			programFile << "(\"" << configName.str() << "\")" << stmtSeparator;
			
			// create  the writer and insert it in the writers map
			std::ostringstream writerClassName;
			writerClassName << varName << "InSpace" << lpsName << "Writer";
			programFile << doubleIndent << writerClassName.str() << " *writer = new ";
			programFile << writerClassName.str();
			programFile << "(" << "config" << paramSeparator; 
			programFile << std::endl << doubleIndent << doubleIndent;
			programFile << "writerId" << paramSeparator;
			programFile << dataItemName.str() << "->getPartsList())" << stmtSeparator;
			programFile << doubleIndent << "Assert(writer != NULL)" << stmtSeparator;
			programFile << doubleIndent << "writer->setWritersCount(writersCount)" << stmtSeparator;
			programFile << doubleIndent << "writersMap->Enter(\"" << writerClassName.str() << "\"";
			programFile << paramSeparator << "writer)" << stmtSeparator;

			// in case a writer should not be created, enter a NULL entry in the map
			programFile << indent << "} else {\n";
			programFile << doubleIndent << "writersMap->Enter(\"" << writerClassName.str() << "\"";
			programFile << paramSeparator << "NULL)" << stmtSeparator;
			programFile << indent << "}\n";
		}
	}

	// end function body
	programFile << '\n' << indent << "return writersMap" << stmtSeparator;
	programFile << "}\n";

	headerFile.close();
	programFile.close();
}

// the implementation of this function is similar to the previous function; therefore, comments are not provided here
void generateReadersMap(const char *headerFileName, 
                const char *programFileName, 
                const char *initials, TaskDef *taskDef) {
	
	std::ofstream programFile;
	std::ofstream headerFile;
        programFile.open (programFileName, std::ofstream::out | std::ofstream::app);
        headerFile.open (headerFileName, std::ofstream::out | std::ofstream::app);
        if (!programFile.is_open() || !headerFile.is_open()) {
                std::cout << "Unable to open header/program file";
                std::exit(EXIT_FAILURE);
        }

        const char *message1 = "file I/O for environmental data structures";
        decorator::writeSectionHeader(headerFile, message1);
        decorator::writeSectionHeader(programFile, message1);
	
	const char *message2 = "Parts-reader map generator";
        decorator::writeSubsectionHeader(headerFile, message2);
        decorator::writeSubsectionHeader(programFile, message2);

	std::ostringstream fnHeader;
	fnHeader << "generateReadersMap(TaskData *taskData" << paramSeparator;
	fnHeader << std::endl << indent << doubleIndent;
	fnHeader << "SegmentState *segment" << paramSeparator;
	fnHeader << std::endl << indent << doubleIndent;
	fnHeader  << "Hashtable<DataPartitionConfig*> *partConfigMap)";
	headerFile << "Hashtable<PartReader*> *" << fnHeader.str() << stmtSeparator;
	programFile << "Hashtable<PartReader*> *" << initials << "::" << fnHeader.str();
	
	Space *rootLps = taskDef->getPartitionHierarchy()->getRootSpace();
	List<const char*> *envVariables = new List<const char*>;
	List<EnvironmentLink*> *envLinkList = taskDef->getEnvironmentLinks();
        for (int i = 0; i < envLinkList->NumElements(); i++) {
                EnvironmentLink *link = envLinkList->Nth(i);
                const char *varName = link->getVariable()->getName();
		envVariables->Append(varName);
	}

	programFile << " {\n\n";
	programFile << indent << "Hashtable<PartReader*> *readersMap = new Hashtable<PartReader*>" << stmtSeparator;
        
	std::deque<Space*> lpsQueue;
	lpsQueue.push_back(rootLps);
	while (!lpsQueue.empty()) {
                Space *lps = lpsQueue.front();
                lpsQueue.pop_front();
		List<Space*> *childrenLpses = lps->getChildrenSpaces();
        	for (int i = 0; i < childrenLpses->NumElements(); i++) {
                	lpsQueue.push_back(childrenLpses->Nth(i));
		}
		if (lps->getSubpartition() != NULL) lpsQueue.push_back(lps->getSubpartition());
		
		const char *lpsName = lps->getName();
		bool readersFound = false;
		List<const char*> *localStructures = lps->getLocalDataStructureNames();
		for (int i = 0; i < localStructures->NumElements(); i++) {

			const char *varName = localStructures->Nth(i);
			if (!string_utils::contains(envVariables, varName)) continue;
			DataStructure *structure = lps->getLocalStructure(varName);
			if (!structure->getUsageStat()->isAllocated()) continue;
			ArrayDataStructure *array = dynamic_cast<ArrayDataStructure*>(structure);
			if (array == NULL) continue;

			if (!readersFound) {
				programFile << "\n" << indent << "// readers for Space " << lpsName << "\n";
				readersFound = true;
			}
			
			std::ostringstream dataItemName;
			dataItemName << varName << "InSpace" << lpsName;
			programFile << indent << "DataItems *" << dataItemName.str(); 
			programFile << " = taskData->getDataItemsOfLps(\"" << lpsName << "\"" << paramSeparator;
			programFile << "\"" << varName << "\")" << stmtSeparator;
			programFile << indent << "if (" << dataItemName.str() << " != NULL";
			programFile << " && !" << dataItemName.str() << "->isEmpty()) {\n";

			std::ostringstream configName;
			configName << varName << "Space" << lpsName << "Config";
			programFile << doubleIndent << "DataPartitionConfig *config" << " = partConfigMap->Lookup";
			programFile << "(\"" << configName.str() << "\")" << stmtSeparator;

                        std::ostringstream readerClassName;
                        readerClassName << varName << "InSpace" << lpsName << "Reader";
                        programFile << doubleIndent << readerClassName.str() << " *reader = new ";
                        programFile << readerClassName.str();
                        programFile << "(" << "config" << paramSeparator; 
                        programFile << dataItemName.str() << "->getPartsList())" << stmtSeparator;
                        programFile << doubleIndent << "Assert(reader != NULL)" << stmtSeparator;
			programFile << doubleIndent << "readersMap->Enter(\"" << readerClassName.str() << "\"";
			programFile << paramSeparator << "reader)" << stmtSeparator;

			programFile << indent << "} else {\n";
			programFile << doubleIndent << "readersMap->Enter(\"" << readerClassName.str() << "\"";
			programFile << paramSeparator << "NULL)" << stmtSeparator;
			programFile << indent << "}\n";

		}
	}
	
	programFile << '\n' << indent << "return readersMap" << stmtSeparator;
	programFile << "}\n";
	headerFile.close();
	programFile.close();
}
