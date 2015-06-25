#include "file_io.h"
#include "../syntax/ast_type.h"
#include "../syntax/ast_task.h"
#include "../semantics/task_space.h"
#include "../utils/list.h"
#include "../utils/string_utils.h"
#include "../utils/decorator_utils.h"
#include "../utils/code_constant.h"

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
	headerFile << indent << varName << "InSpace" << lpsName << "Reader (";
	headerFile << "DataPartitionConfig *partConfig" << paramSeparator;
	headerFile << std::endl << indent << doubleIndent;
	headerFile << "DataPartsList *partsList" << paramSeparator;
	headerFile << "const char *fileName)\n" ;
	headerFile << indent << doubleIndent << ": PartReader(";
	headerFile << "partsList" << paramSeparator << "fileName" << paramSeparator << "partConfig) {\n";
	headerFile << doubleIndent << "this->partConfig = partConfig" << stmtSeparator;
	headerFile << doubleIndent << "this->stream = NULL" << stmtSeparator;
	headerFile << indent << "}\n"; 

	// write implementations for the three functions needed to do structure specific reading
	headerFile << indent << "void begin() {\n";
	headerFile << doubleIndent << "stream = new TypedInputStream<" << elementType->getCType() << ">";
	headerFile << "(fileName)" << stmtSeparator;  
	headerFile << doubleIndent << "stream->open()" << stmtSeparator;
	headerFile << indent << "}\n";
	headerFile << indent << "void terminate() { stream->close()" << stmtTerminator << " }\n";
	headerFile << indent << "List<int> *getDataIndex(List<int> *partIndex) {";
	if (array->isReordered(lps->getRoot())) {
		headerFile << " return DataHandler::getDataIndex(partIndex)" << stmtTerminator << " }\n";
	} else {
		headerFile << " return partIndex" << stmtTerminator << " }\n";
	}
	headerFile << indent << "void readElement(List<int> *dataIndex, int storeIndex, void *partStore) {\n";
	headerFile << doubleIndent << elementType->getCType();
	headerFile << " *dataStore = (" << elementType->getCType() << "*) partStore" << stmtSeparator;
	headerFile << doubleIndent << "dataStore[storeIndex] = stream->readElement(dataIndex)" << stmtSeparator;
	headerFile << indent <<  "}\n";

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
	headerFile << indent << varName << "InSpace" << lpsName << "Writer (";
	headerFile << "DataPartitionConfig *partConfig" << paramSeparator;
	headerFile << std::endl << indent << doubleIndent;
	headerFile << "int writerId" << paramSeparator;
	headerFile << "DataPartsList *partsList" << paramSeparator;
	headerFile << "const char *fileName)\n" ;
	headerFile << indent << doubleIndent << ": PartWriter(writerId" << paramSeparator;
	headerFile << "partsList" << paramSeparator << "fileName" << paramSeparator << "partConfig) {\n";
	headerFile << doubleIndent << "this->partConfig = partConfig" << stmtSeparator;
	headerFile << doubleIndent << "this->stream = NULL" << stmtSeparator;
	headerFile << indent << "}\n"; 

	// write implementations for the four functions needed to do structure specific writing
	headerFile << indent << "void begin() {\n";
	headerFile << doubleIndent << "stream = new TypedOutputStream<" << elementType->getCType() << ">";
	headerFile << "(fileName" << paramSeparator << "getDimensionList()" << paramSeparator;
	headerFile << "writerId == 0)" << stmtSeparator;  
	headerFile << doubleIndent << "stream->open()" << stmtSeparator;
	headerFile << indent << "}\n";
	headerFile << indent << "void terminate() { stream->close()" << stmtTerminator << " }\n";
	headerFile << indent << "List<int> *getDataIndex(List<int> *partIndex) {";
	if (array->isReordered(lps->getRoot())) {
		headerFile << " return DataHandler::getDataIndex(partIndex)" << stmtTerminator << " }\n";
	} else {
		headerFile << " return partIndex" << stmtTerminator << " }\n";
	}
	headerFile << indent << "void writeElement(List<int> *dataIndex, int storeIndex, void *partStore) {\n";
	headerFile << doubleIndent << elementType->getCType();
	headerFile << " *dataStore = (" << elementType->getCType() << "*) partStore" << stmtSeparator;
	headerFile << doubleIndent << "stream->writeElement(dataStore[storeIndex]" <<  paramSeparator;
	headerFile << "dataIndex)" << stmtSeparator;
	headerFile << indent <<  "}\n";

	headerFile << "}" << stmtSeparator;
}

void generateReaderWriterForLpsStructures(std::ofstream &headerFile, const char *initials, Space *lps) {
	
	std::ostringstream header;
	const char *lpsName = lps->getName();
	header << "Space " << lpsName;
	decorator::writeSubsectionHeader(headerFile, header.str().c_str());

	List<const char*> *localStructures = lps->getLocalDataStructureNames();
	for (int i = 0; i < localStructures->NumElements(); i++) {
		const char *varName = localStructures->Nth(i);
		
		// reader and writers are generated for a data structure only if it has been allocated in the current LPS
		DataStructure *structure = lps->getLocalStructure(varName);
		if (!structure->getUsageStat()->isAllocated()) continue;

		// currently we only read/write arrays from file
		ArrayDataStructure *array = dynamic_cast<ArrayDataStructure*>(structure);
		if (array == NULL) continue;

		generatePartReaderForStructure(headerFile, array);
		generatePartWriterForStructure(headerFile, array);
	}
}

void generateReaderWriters(const char *headerFileName, const char *initials, Space *rootLps) {
        
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
			generateReaderWriterForLpsStructures(headerFile, initials, lps);
		}
	}

	headerFile.close();
}

void generateRoutineForDataInitialization(const char *headerFileName, const char *programFileName, TaskDef *taskDef) {
	
	std::ofstream programFile;
	std::ofstream headerFile;
        programFile.open (programFileName, std::ofstream::out | std::ofstream::app);
        headerFile.open (headerFileName, std::ofstream::out | std::ofstream::app);
        if (!programFile.is_open() || !headerFile.is_open()) {
                std::cout << "Unable to open header/program file";
                std::exit(EXIT_FAILURE);
        }

        const char *message = "environment initializer function";
        decorator::writeSectionHeader(headerFile, message);
	headerFile << std::endl;
        decorator::writeSectionHeader(programFile, message);
	programFile << std::endl;

	const char *initials = string_utils::getInitials(taskDef->getName());
	Space *rootLps = taskDef->getPartitionHierarchy()->getRootSpace();
	List<EnvironmentLink*> *envLinks = taskDef->getEnvironmentLinks();

	// The initializer function needs the environment parameter to determine what data structures are instructed to 
	// be input from file. It populates the LPS contents of the taskData section based on those intructions and the 
	// part-configuration map is needed for file index to data part index transformation during reading.
	std::ostringstream fnHeader;
	fnHeader << "initializeEnvironment(" << initials << "Environment *env" << paramSeparator;
	fnHeader << std::endl << indent << doubleIndent << "TaskData *taskData" << paramSeparator;
	fnHeader << std::endl << indent << doubleIndent << "Hashtable<DataPartitionConfig*> *partConfigMap)";

	headerFile << "void " << fnHeader.str() << stmtSeparator;
	programFile << "void " << string_utils::toLower(initials) << "::" << fnHeader.str();
	programFile << " {\n\n";

	// iterate over the environment links and consider only those that are or may be external	
	for (int i = 0; i < envLinks->NumElements(); i++) {
		EnvironmentLink *link = envLinks->Nth(i);
		if (!link->isExternal()) continue;

		// add a condition block checking if there is an file input instruction for the link under concern
		const char *varName = link->getVariable()->getName();
		programFile << indent << "if (env->hasInputBinding(\"" << varName << "\")) {\n";
		programFile << doubleIndent << "const char *fileName = env->getInputFileForStructure";
		programFile << "(\"" << varName << "\")" << stmtSeparator;

		// iterate over the list of LPSes and consider only those that allocates the current data structure
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
			if (!lps->allocateStructure(varName)) continue;

			// add a checking to see if the current segment has data parts for the structure for the LPS
			const char *lpsName = lps->getName();
			std::ostringstream dataItemName;
			dataItemName << varName << "InSpace" << lpsName;
			programFile << doubleIndent << "DataItems *" << dataItemName.str(); 
			programFile << " = taskData->getDataItemsOfLps(\"" << lpsName << "\"" << paramSeparator;
			programFile << "\"" << varName << "\")" << stmtSeparator;
			programFile << doubleIndent << "if (" << dataItemName.str() << " != NULL) {\n";

			// retrieve the partition configuration object for the structure in the LPS
			std::ostringstream configName;
			configName << varName << "Space" << lpsName << "Config";
			programFile << tripleIndent << "DataPartitionConfig *config" << " = partConfigMap->Lookup";
			programFile << "(\"" << configName.str() << "\")" << stmtSeparator;

			// instantiate a part reader subclass of appropriate type
			std::ostringstream readerClassName;
			readerClassName << varName << "InSpace" << lpsName << "Reader";
			programFile << tripleIndent << readerClassName.str() << " *reader = new ";
			programFile << readerClassName.str();
			programFile << "(" << "config" << paramSeparator << std::endl << doubleIndent << tripleIndent; 
			programFile << dataItemName.str() << "->getPartsList()" << paramSeparator;
			programFile << "fileName" << ")" << stmtSeparator;

			// do the reading
			programFile << tripleIndent << "reader->processParts()" << stmtSeparator;

			programFile << doubleIndent << "}\n";		
		}
		programFile << indent << "}\n";
	}
	programFile << "}\n";

	headerFile.close();
	programFile.close();		
}

// this function is very similar to the generateRoutineForDataInitialization() function presented above; so did not write
// too many comments in it
void generateRoutineForDataStorage(const char *headerFileName, const char *programFileName, TaskDef *taskDef) {
	
	std::ofstream programFile;
	std::ofstream headerFile;
        programFile.open (programFileName, std::ofstream::out | std::ofstream::app);
        headerFile.open (headerFileName, std::ofstream::out | std::ofstream::app);
        if (!programFile.is_open() || !headerFile.is_open()) {
                std::cout << "Unable to open header/program file";
                std::exit(EXIT_FAILURE);
        }

        const char *message = "function for storing environmental structures in files";
        decorator::writeSectionHeader(headerFile, message);
	headerFile << std::endl;
        decorator::writeSectionHeader(programFile, message);
	programFile << std::endl;

	const char *initials = string_utils::getInitials(taskDef->getName());
	Space *rootLps = taskDef->getPartitionHierarchy()->getRootSpace();
	List<EnvironmentLink*> *envLinks = taskDef->getEnvironmentLinks();
	
	std::ostringstream fnHeader;
	fnHeader << "storeEnvironment(" << initials << "Environment *env" << paramSeparator;
	fnHeader << std::endl << indent << doubleIndent << "TaskData *taskData" << paramSeparator;
	fnHeader << std::endl << indent << doubleIndent << "SegmentState *segment" << paramSeparator;
	fnHeader << std::endl << indent << doubleIndent << "Hashtable<DataPartitionConfig*> *partConfigMap)";

	headerFile << "void " << fnHeader.str() << stmtSeparator;
	programFile << "void " << string_utils::toLower(initials) << "::" << fnHeader.str();
	programFile << " {\n\n";
	
	// unlike in the case of the initializer we consider all linked and created environment objects (though the present
	// implementation is limited to arrays only) while writing
	for (int i = 0; i < envLinks->NumElements(); i++) {
		EnvironmentLink *link = envLinks->Nth(i);
		const char *varName = link->getVariable()->getName();
		programFile << indent << "if (env->hasOutputBinding(\"" << varName << "\")) {\n";
		programFile << doubleIndent << "const char *fileName = env->getOutputFileForStructure";
		programFile << "(\"" << varName << "\")" << stmtSeparator;

		// unlike in the case of reader, we need to write the output from only a single LPS that allocate the data
		// structure;
		Space *allocatingLps = NULL;
		std::deque<Space*> lpsQueue;
		List<Space*> *childrenLpses = rootLps->getChildrenSpaces();
		for (int i = 0; i < childrenLpses->NumElements(); i++) {
			lpsQueue.push_back(childrenLpses->Nth(i));
		}
		 while (!lpsQueue.empty()) {
			Space *lps = lpsQueue.front();
			lpsQueue.pop_front();
			if (lps->allocateStructure(varName)) {
				allocatingLps = lps;
				break;
			}
			childrenLpses = lps->getChildrenSpaces();
			for (int i = 0; i < childrenLpses->NumElements(); i++) {
				lpsQueue.push_back(childrenLpses->Nth(i));
			}
			if (lps->getSubpartition() != NULL) lpsQueue.push_back(lps->getSubpartition());
		}	
		if (allocatingLps != NULL) {
			const char *lpsName = allocatingLps->getName();
			std::ostringstream dataItemName;
			dataItemName << varName << "InSpace" << lpsName;
			programFile << doubleIndent << "DataItems *" << dataItemName.str(); 
			programFile << " = taskData->getDataItemsOfLps(\"" << lpsName << "\"" << paramSeparator;
			programFile << "\"" << varName << "\")" << stmtSeparator;
			programFile << doubleIndent << "if (" << dataItemName.str() << " != NULL) {\n";

			std::ostringstream configName;
			configName << varName << "Space" << lpsName << "Config";
			programFile << tripleIndent << "DataPartitionConfig *config" << " = partConfigMap->Lookup";
			programFile << "(\"" << configName.str() << "\")" << stmtSeparator;
			
			std::ostringstream writerClassName;
			writerClassName << varName << "InSpace" << lpsName << "Writer";
			programFile << tripleIndent << writerClassName.str() << " *writer = new ";
			programFile << writerClassName.str();
			programFile << "(" << "config" << paramSeparator; 
			programFile << std::endl << doubleIndent << tripleIndent;
			programFile << "segment->getSegmentId()" << paramSeparator; 
			programFile << std::endl << doubleIndent << tripleIndent;
			programFile << dataItemName.str() << "->getPartsList()" << paramSeparator;
			programFile << "fileName" << ")" << stmtSeparator;
			programFile << tripleIndent << "writer->processParts()" << stmtSeparator;
				
			programFile << doubleIndent << "}\n";		
		}
		programFile << indent << "}\n";
	}
	programFile << "}\n";

	headerFile.close();
	programFile.close();		
}

