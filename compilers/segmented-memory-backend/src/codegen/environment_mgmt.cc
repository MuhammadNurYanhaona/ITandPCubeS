#include "environment_mgmt.h"
#include "../syntax/ast_task.h"
#include "../syntax/ast_type.h"
#include "../semantics/task_space.h"
#include "../static-analysis/task_env_stat.h"
#include "../utils/list.h"
#include "../utils/hashtable.h"
#include "../utils/decorator_utils.h"
#include "../utils/string_utils.h"
#include "../utils/code_constant.h"

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <deque>	

void generateTaskEnvironmentClass(TaskDef *taskDef, const char *initials, 
		const char *headerFileName, 
		const char *programFileName) {
	
	std::cout << "Generating structures and routines for environment management\n";

        std::ofstream programFile, headerFile;
        programFile.open (programFileName, std::ofstream::out | std::ofstream::app);
        headerFile.open (headerFileName, std::ofstream::out | std::ofstream::app);
        if (!programFile.is_open() || !headerFile.is_open()) {
                std::cout << "Unable to open header/program file";
                std::exit(EXIT_FAILURE);
        }
        const char *header = "Task Environment Management Structures and Functions";
        decorator::writeSectionHeader(headerFile, header);
        decorator::writeSectionHeader(programFile, header);
	const char *message = "Task Environment Implementation Class";
	decorator::writeSubsectionHeader(headerFile, message);

	// declare the extension class for task environment in the header file
	headerFile << '\n' << "class TaskEnvironmentImpl : public TaskEnvironment {\n";

	// declare all non-array environmental variables as properties of the task environment
	bool scalarFound = false;
	List<EnvironmentLink*> *envLinkList = taskDef->getEnvironmentLinks();
	Space *rootLps = taskDef->getPartitionHierarchy()->getRootSpace();
	// if the task environment contains non-array variables, they should be zero initialized in the constructor 
	std::ostringstream constructorContents;
	for (int i = 0; i < envLinkList->NumElements(); i++) {
		EnvironmentLink *link = envLinkList->Nth(i);
		const char *varName = link->getVariable()->getName();
		DataStructure *structure = rootLps->getStructure(varName);
		ArrayDataStructure *array = dynamic_cast<ArrayDataStructure*>(structure);
		StaticArrayType *staticArray = dynamic_cast<StaticArrayType*>(structure->getType());
		if (array != NULL && staticArray == NULL) continue;
	
		if (!scalarFound) {
			headerFile << "  public:\n";
			scalarFound = true;
		}

		// declare a property of the appropriate type for the non-array variable 
		Type *type = structure->getType();
		headerFile << indent << type->getCppDeclaration(varName) << stmtSeparator;

		// if the object is of a user defined type then call its constructor to initialize the property
		NamedType *userDefinedType = dynamic_cast<NamedType*>(type);
		if (userDefinedType != NULL) {
			constructorContents << indent << varName;
			constructorContents << " = " << type->getCType() << "()" << stmtSeparator;
			continue;
		}

		// at this momemt, we are skipping static array initialization as such an array may have other
		// static arrays as elements and we need to capture such repetition is an elegant way
		if (staticArray == NULL) {
			// zero initialize the built in types 
			if (type == Type::boolType) {
				constructorContents << indent << varName  << " = false" << stmtSeparator;
			} else {
				constructorContents << indent << varName  << " = 0" << stmtSeparator;
			}
		}
	}

	// generate the constructor in the program file
	const char *message2 = "Constructor";
	decorator::writeSubsectionHeader(programFile, message2);
	programFile << '\n';
	programFile << initials << "::" << "TaskEnvironmentImpl::TaskEnvironmentImpl() : TaskEnvironment() {\n";
	programFile << constructorContents.str();
	programFile << indent << "prepareItemsMap()" << stmtSeparator;
	programFile << indent << "resetEnvInstructions()" << stmtSeparator;
	programFile << "}\n";

	// definitions for the constructor and two functions each task environment subclass needs to provide 
	// implementations for
	headerFile << "  public:\n";
	headerFile << indent << "TaskEnvironmentImpl()" << stmtSeparator;
	headerFile << indent << "void prepareItemsMap()" << stmtSeparator;
	headerFile << indent << "void setDefaultTaskCompletionInstrs()" << stmtSeparator;
	headerFile << "}" << stmtSeparator;

	// two functions need to be implemented by the task specific environment subclass; call other functions to
	// generate their definitions in the program file
	generateFnForItemsMapPreparation(taskDef, initials, programFile);
	generateFnForTaskCompletionInstrs(taskDef, initials, programFile);

	headerFile.close();
	programFile.close();
}

void generateFnForItemsMapPreparation(TaskDef *taskDef, const char *initials, std::ofstream &programFile) {
	
	Space *rootLps = taskDef->getPartitionHierarchy()->getRootSpace();
	const char *message = "Task Environment Function Implementations";
	decorator::writeSubsectionHeader(programFile, message);

	programFile << '\n';
	programFile << "void " << initials << "::" << "TaskEnvironmentImpl::prepareItemsMap() {\n";

	List<EnvironmentLink*> *envLinkList = taskDef->getEnvironmentLinks();
	for (int i = 0; i < envLinkList->NumElements(); i++) {
		
		EnvironmentLink *link = envLinkList->Nth(i);
		const char *varName = link->getVariable()->getName();
		DataStructure *structure = rootLps->getStructure(varName);
		ArrayDataStructure *array = dynamic_cast<ArrayDataStructure*>(structure);
		StaticArrayType *staticArray = dynamic_cast<StaticArrayType*>(structure->getType());
		
		// item map preparation is applicable for environmental arrays only
		if (array == NULL || staticArray != NULL) continue;

		programFile << '\n';
		ArrayType *type = (ArrayType*) array->getType();
		int dimensionality = type->getDimensions();
		int linkId = i;

		// New items are added to the environment only if they are not already there. In the situation
		// of the same task being invoked multiple times with the same environment parameter, we do not 
		// want to replace the existing task items with new items. Rather, we want to update them. During 
		// the update process, we can do necessary changes in the program environment too. If we throw 
		// away old items, we will loose old associations between the task and program environments. 
		programFile << indent << "if (envItems->Lookup(\"" << varName << "\") == NULL) {\n";
		
		programFile << doubleIndent << "EnvironmentLinkKey *key";
		programFile << linkId << " = new EnvironmentLinkKey(";
		programFile << "\"" << varName << "\"" << paramSeparator;
		programFile << linkId << ")" << stmtSeparator;

		programFile << doubleIndent << "TaskItem *item" << linkId << " = new TaskItem(";
		programFile << "key" << linkId << paramSeparator;

		LinkageType linkageType = link->getMode();
		if (linkageType == TypeLink) {
			programFile << "IN_OUT";
		} else if (linkageType == TypeCreateIfNotLinked) {
			programFile << "OPTIONAL_IN_OUT";
		} else {
			programFile << "OUT";
		}
		programFile << paramSeparator;
		programFile << dimensionality << paramSeparator;
		Type *elementType = type->getTerminalElementType();
		programFile << "sizeof(" << elementType->getCType() << "))" << stmtSeparator;
		programFile << doubleIndent << "item" << linkId << "->setEnvironment(this)" << stmtSeparator;
		
		programFile << doubleIndent << "envItems->Enter(\"" << varName << "\"" << paramSeparator;
		programFile << "item" << linkId << ")" << stmtSeparator;
		
		// closing the if block	
		programFile << indent << "}\n";
	}
	
	programFile << "}\n";
}

void generateFnForTaskCompletionInstrs(TaskDef *taskDef, const char *initials, std::ofstream &programFile) {
	
	Space *rootLps = taskDef->getPartitionHierarchy()->getRootSpace();
	TaskEnvStat *taskEnvStat = taskDef->getAfterExecutionEnvStat();

	programFile << '\n';
	programFile << "void " << initials << "::" << "TaskEnvironmentImpl::setDefaultTaskCompletionInstrs() {\n";

	List<EnvironmentLink*> *envLinkList = taskDef->getEnvironmentLinks();
	for (int i = 0; i < envLinkList->NumElements(); i++) {
		
		EnvironmentLink *link = envLinkList->Nth(i);
		const char *varName = link->getVariable()->getName();
		DataStructure *structure = rootLps->getStructure(varName);
		ArrayDataStructure *array = dynamic_cast<ArrayDataStructure*>(structure);
		StaticArrayType *staticArray = dynamic_cast<StaticArrayType*>(structure->getType());
	
		// environment management instructions are applicable for dynamic arrays only
		if (array == NULL || staticArray != NULL) continue;

		// if the variable's content has not been accessed at all then there is no environmental update to do for it
		EnvVarStat *varEnvStat = taskEnvStat->getVariableStat(varName);
		if (varEnvStat == NULL) continue;

		// if the variable has been modified then other versions of the data it refers to should be flagged as stale
		if (varEnvStat->isUpdated()) {
			programFile << '\n';
			programFile << indent << "TaskItem *" << varName << "Item = envItems->Lookup(\"";
			programFile << varName << "\")" << stmtSeparator;
			programFile << indent << "ChangeNotifyInstruction *instr" << i << " = new ";
			programFile << "ChangeNotifyInstruction(" << varName << "Item)" << stmtSeparator;
			programFile << indent << "addEndEnvInstruction(instr" << i << ")" << stmtSeparator;
		}
	}
	
	programFile << "}\n";
}

void generateFnToInitEnvLinksFromEnvironment(TaskDef *taskDef, const char *initials,
                const char *headerFileName,
                const char *programFileName) {
	
        std::ofstream programFile, headerFile;
        programFile.open (programFileName, std::ofstream::out | std::ofstream::app);
        headerFile.open (headerFileName, std::ofstream::out | std::ofstream::app);
        if (!programFile.is_open() || !headerFile.is_open()) {
                std::cout << "Unable to open header/program file";
                std::exit(EXIT_FAILURE);
        }
	Space *rootLps = taskDef->getPartitionHierarchy()->getRootSpace();
	const char *message = "Environmental Links Object Generator";
	decorator::writeSubsectionHeader(headerFile, message);
	decorator::writeSubsectionHeader(programFile, message);
	headerFile << '\n';
	programFile << '\n';

	// generate function header
	std::ostringstream fnHeader;
	programFile << "EnvironmentLinks " << initials << "::";
        headerFile << "EnvironmentLinks ";
        fnHeader << "initiateEnvLinks(TaskEnvironment *environment)";
	programFile << fnHeader.str();
        headerFile << fnHeader.str() << stmtSeparator;

        // open function definition
        programFile << " {\n\n";

	// declare a local environment link instance that will be returned by the generated function at the end
	programFile << indent << "EnvironmentLinks links" << stmtSeparator;
	
	// convert the generic task environment object into an instance of the task specific subclass so that its
	// properties can be accessed
	programFile << indent << initials << "::TaskEnvironmentImpl *taskEnv = ";
	programFile << "(" << initials << "::TaskEnvironmentImpl *) environment" << stmtSeparator;
	
	List<EnvironmentLink*> *envLinkList = taskDef->getEnvironmentLinks();
	for (int i = 0; i < envLinkList->NumElements(); i++) {
		
		EnvironmentLink *link = envLinkList->Nth(i);
		
		// if the link is for an out variable then it is not a part of the environment links
		if (!link->isExternal()) continue;

		const char *varName = link->getVariable()->getName();
		DataStructure *structure = rootLps->getStructure(varName);
		ArrayDataStructure *array = dynamic_cast<ArrayDataStructure*>(structure);
		StaticArrayType *staticArray = dynamic_cast<StaticArrayType*>(structure->getType());
		
		// for non-array variables or static arrays copy in properties from the environment to the links 
		if (array == NULL || staticArray != NULL) {
			if (staticArray == NULL) {
				programFile << indent << "links." << varName << " = ";
				programFile << "taskEnv->" << varName << stmtSeparator;
			} else {
				int dimensions = array->getDimensionality();
				for (int d = 0; d < dimensions; d++) {
					programFile << indent << "links." << varName << "[" << d << "] = ";
					programFile << "taskEnv->" << varName;
					programFile << "[" << d << "]" << stmtSeparator;
				}
			}
			continue;
		}
	
		// retrieve the item corresponding to the environmental variable from the task environment object
		programFile << '\n';
		programFile << indent << "TaskItem *" << varName << "Item = environment->getItem(\"";
		programFile << varName << "\")" << stmtSeparator;
		
		// copy dimension information into the environment link property 
		int dimensions = array->getDimensionality();
		for (int d = 0; d < dimensions; d++) {
			programFile << indent;
			programFile << "links." << varName;
			programFile << "Dims[" << d << "] = ";
			programFile << varName << "Item->getDimension(" << d << ")" << stmtSeparator;
		}
	}

	programFile << '\n' << indent << "return links" << stmtSeparator;
	programFile << "}\n";

	headerFile.close();
	programFile.close();
}

void generateFnToPreconfigureLpsAllocations(TaskDef *taskDef, const char *initials,
                const char *headerFileName,
                const char *programFileName) {
        
	std::ofstream programFile, headerFile;
        programFile.open (programFileName, std::ofstream::out | std::ofstream::app);
        headerFile.open (headerFileName, std::ofstream::out | std::ofstream::app);
        if (!programFile.is_open() || !headerFile.is_open()) {
                std::cout << "Unable to open header/program file";
                std::exit(EXIT_FAILURE);
        }
	Space *rootLps = taskDef->getPartitionHierarchy()->getRootSpace();
	const char *message = "LPS Allocation Preconfigurers";
	decorator::writeSubsectionHeader(headerFile, message);
	decorator::writeSubsectionHeader(programFile, message);
	headerFile << '\n';
	programFile << '\n';

	// generate function header
	std::ostringstream fnHeader;
	programFile << "void " << initials << "::";
        headerFile << "void ";
        fnHeader << "preconfigureLpsAllocationsInEnv(TaskEnvironment *environment";
	fnHeader << paramSeparator << '\n' << doubleIndent << "ArrayMetadata *metadata";
	fnHeader << paramSeparator << '\n' << doubleIndent;
	fnHeader << "Hashtable<DataPartitionConfig*> *partConfigMap)";
	programFile << fnHeader.str();
        headerFile << fnHeader.str() << stmtSeparator;

        // open function definition
        programFile << " {\n\n";

	List<EnvironmentLink*> *envLinkList = taskDef->getEnvironmentLinks();
	for (int i = 0; i < envLinkList->NumElements(); i++) {
		
		EnvironmentLink *link = envLinkList->Nth(i);
		const char *varName = link->getVariable()->getName();
		DataStructure *structure = rootLps->getStructure(varName);
		ArrayDataStructure *array = dynamic_cast<ArrayDataStructure*>(structure);
		StaticArrayType *staticArray = dynamic_cast<StaticArrayType*>(structure->getType());
	
		// LPS allocations configuration is applicable for dynamic arrays only
		if (array == NULL || staticArray != NULL) continue;

		// retrieve the task item
		programFile << '\n';
		programFile << indent << "TaskItem *" << varName << "Item = environment->getItem(\"";
		programFile << varName << "\")" << stmtSeparator;

		// copy dimension ranges and lengths from array metadata to the task item
		int dimensions = array->getDimensionality();
		for (int d = 0; d < dimensions; d++) {
			programFile << indent;
			programFile << varName << "Item->setDimension(" << d << paramSeparator;
			programFile << "metadata->" << varName << "Dims[" << d << "])" << stmtSeparator;
		}
		
		// go over the partition hierarchy and check the LPSes where the array has been allocated
		std::deque<Space*> lpsQueue;
		lpsQueue.push_back(rootLps);
		while (!lpsQueue.empty()) {
			Space *lps = lpsQueue.front();
			lpsQueue.pop_front();
			List<Space*> *children = lps->getChildrenSpaces();
			for (int i = 0; i < children->NumElements(); i++) {
                        	lpsQueue.push_back(children->Nth(i));
                	}
                	if (lps->getSubpartition() != NULL) lpsQueue.push_back(lps->getSubpartition());
			if (!lps->allocateStructure(varName)) continue;

			// for each LPS that allocates the structure, configure an LPS allocation in the task item
			const char *lpsName = lps->getName();
			std::ostringstream allocationName; 
			allocationName << varName << "InSpace" << lpsName;
			std::ostringstream configName;
			configName << varName << "Space" << lpsName << "Config";
			programFile << indent << "DataPartitionConfig *" << configName.str();
			programFile << " = partConfigMap->Lookup(\"" << configName.str() << "\")" << stmtSeparator;
			programFile << indent << "LpsAllocation *" << allocationName.str() << " = ";
			programFile << varName << "Item->getLpsAllocation(\"" << lpsName << "\")" << stmtSeparator;
			programFile << indent << "if (" << allocationName.str() << " == NULL) {\n";
			programFile << doubleIndent << varName << "Item->preConfigureLpsAllocation(";
			programFile << '"' << lpsName << '"' << paramSeparator;
			programFile << configName.str() << ")" << stmtSeparator;	
			programFile << indent << "} else {\n";
			programFile << doubleIndent << allocationName.str() << "->setPartitionConfig(";
			programFile << configName.str() << ")" << stmtSeparator;
			programFile << indent << "}\n";
		}	
	}	

	// close function definition	
	programFile << "}\n";
	
	headerFile.close();
	programFile.close();
}

void generateFnToCopyBackNonArrayVars(TaskDef *taskDef,
                const char *initials,
                const char *headerFileName,
                const char *programFileName) {

	std::ofstream programFile, headerFile;
        programFile.open (programFileName, std::ofstream::out | std::ofstream::app);
        headerFile.open (headerFileName, std::ofstream::out | std::ofstream::app);
        if (!programFile.is_open() || !headerFile.is_open()) {
                std::cout << "Unable to open header/program file";
                std::exit(EXIT_FAILURE);
        }
	Space *rootLps = taskDef->getPartitionHierarchy()->getRootSpace();
	const char *message = "Non-array variables copier";
	decorator::writeSubsectionHeader(headerFile, message);
	decorator::writeSubsectionHeader(programFile, message);
	headerFile << '\n';
	programFile << '\n';

	// generate function header
	std::ostringstream fnHeader;
	programFile << "void " << initials << "::";
        headerFile << "void ";
        fnHeader << "copyBackNonArrayEnvVariables(TaskEnvironment *environment";
	fnHeader << paramSeparator << "TaskGlobals *taskGlobals)";
	programFile << fnHeader.str();
        headerFile << fnHeader.str() << stmtSeparator;

        // open function definition
        programFile << " {\n";

	// convert the generic task environment object into an instance of the task specific subclass so that its
	// properties can be accessed
	programFile << indent << initials << "::TaskEnvironmentImpl *taskEnv = ";
	programFile << "(" << initials << "::TaskEnvironmentImpl *) environment" << stmtSeparator;
	
	List<EnvironmentLink*> *envLinkList = taskDef->getEnvironmentLinks();
	for (int i = 0; i < envLinkList->NumElements(); i++) {
		
		EnvironmentLink *link = envLinkList->Nth(i);
		const char *varName = link->getVariable()->getName();
		DataStructure *structure = rootLps->getStructure(varName);
		ArrayDataStructure *array = dynamic_cast<ArrayDataStructure*>(structure);
		StaticArrayType *staticArray = dynamic_cast<StaticArrayType*>(structure->getType());
		if (array != NULL && staticArray == NULL) continue;

		if (staticArray == NULL) {
			programFile << indent << "taskEnv->" << varName << " = ";
			programFile << "taskGlobals->" << varName << stmtSeparator;
		} else {
			int dimensions = array->getDimensionality();
			for (int d = 0; d < dimensions; d++) {
				programFile << indent << "taskEnv->" << varName << "[" << d << "] = ";
				programFile << "taskGlobals->" << varName;
				programFile << "[" << d << "]" << stmtSeparator;
			}
		}
	}

	// close function definition	
	programFile << "}\n";
	
	headerFile.close();
	programFile.close();
}

