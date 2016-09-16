#include "ast.h"
#include "ast_type.h"
#include "ast_def.h"
#include "ast_stmt.h"

#include "../utils/list.h"
#include "../utils/string_utils.h"
#include "../semantics/scope.h"
#include "../semantics/symbol.h"
#include "../static-analysis/array_assignment.h"
#include "errors.h"

#include <fstream>
#include <sstream>
#include <cstdlib>

//----------------------------------------- Variable Definition ------------------------------------------/

VariableDef::VariableDef(Identifier *i, Type *t) : Definition(*i->GetLocation()) {
	Assert(i != NULL && t != NULL);
	id = i; type = t;
	id->SetParent(this);
	type->SetParent(this);
}

VariableDef::VariableDef(Identifier *i) : Definition(*i->GetLocation()) {
	Assert(i != NULL);
	id = i; type = NULL;
	id->SetParent(this);
}

void VariableDef::PrintChildren(int indentLevel) {
	id->Print(indentLevel + 1);
        if (type != NULL) type->Print(indentLevel + 1);
}

List<VariableDef*> *VariableDef::DecomposeDefs(List<Identifier*> *idList, Type *type) {
	List<VariableDef*> *varList = new List<VariableDef*>;
	for (int i = 0; i < idList->NumElements(); i++) {
           varList->Append(new VariableDef(idList->Nth(i), type));
       	}
	return varList;
}

void VariableDef::validateScope(Scope *parentScope) {
	
	ArrayType *arrayType = dynamic_cast<ArrayType*>(type);
	ListType *listType = dynamic_cast<ListType*>(type);
	if (arrayType != NULL) {
		Type *elementType = arrayType->getTerminalElementType();
		if (parentScope->lookup(elementType->getName()) == NULL) {
			ReportError::UndeclaredTypeError(id, elementType, "array element ", false);
		}
	} else if (listType != NULL) {
		Type *elementType = listType->getTerminalElementType();
		if (parentScope->lookup(elementType->getName()) == NULL) {
			ReportError::UndeclaredTypeError(id, elementType, "list element ", false);
		}
	} else if (parentScope->lookup(type->getName()) == NULL) {
		ReportError::UndeclaredTypeError(id, type, NULL, false);
	}
}

//----------------------------------------- Program Definition ------------------------------------------/

ProgramDef *ProgramDef::program = NULL;

ProgramDef::ProgramDef(List<Node*> *c) : Definition() {
	Assert(c != NULL);
	components = c;
	for (int i = 0; i < components->NumElements(); i++) {
		Node *node = components->Nth(i);
		node->SetParent(this);
	}
} 

void ProgramDef::PrintChildren(int indentLevel) {
	components->PrintAll(indentLevel+1);
    	printf("\n");
}

void ProgramDef::attachScope(Scope *parentScope) {
	
	Scope *scope = new Scope(ProgramScope);
	Type::storeBuiltInTypesInScope(scope);	

	for (int i = 0; i < components->NumElements(); i++) {
		Node *node = components->Nth(i);
		Definition *def = (Definition *) node;
		def->attachScope(scope);
	}

	for (int i = 0; i < components->NumElements(); i++) {
		Node *node = components->Nth(i);
		TaskDef *taskDef = dynamic_cast<TaskDef*>(node);
		if (taskDef != NULL) {
			taskDef->typeCheckInitSection(scope);
		}
	}
	
	symbol = new Symbol("Program", this);
	symbol->setNestedScope(scope);
}
	
void ProgramDef::validateScope(Scope *parentScope) {
	Scope *programScope = symbol->getNestedScope();
	for (int i = 0; i < components->NumElements(); i++) {
		Node *node = components->Nth(i);
		Definition *def = (Definition *) node;
		def->validateScope(programScope);
	}	
}

void ProgramDef::performStaticAnalysis() {
	for (int i = 0; i < components->NumElements(); i++) {
                Node *node = components->Nth(i);
                TaskDef *taskDef = dynamic_cast<TaskDef*>(node);
                if (taskDef != NULL) {
                        taskDef->analyseCode();
                }
        }
}

void ProgramDef::printTasks() {
	for (int i = 0; i < components->NumElements(); i++) {
                Node *node = components->Nth(i);
                TaskDef *taskDef = dynamic_cast<TaskDef*>(node);
                if (taskDef != NULL) {
                        taskDef->print();
                }
        }
}

Node *ProgramDef::getTaskDefinition(const char *taskName) {
	for (int i = 0; i < components->NumElements(); i++) {
                Node *node = components->Nth(i);
                TaskDef *taskDef = dynamic_cast<TaskDef*>(node);
                if (taskDef == NULL) continue;
		if (strcmp(taskName, taskDef->getName()) == 0) return taskDef;
	}
	return NULL;
}

List<TaskDef*> *ProgramDef::getTasks() {
	List<TaskDef*> *taskList = new List<TaskDef*>;
	for (int i = 0; i < components->NumElements(); i++) {
                Node *node = components->Nth(i);
                TaskDef *taskDef = dynamic_cast<TaskDef*>(node);
                if (taskDef == NULL) continue;
		taskList->Append(taskDef);
	}
	return taskList;
}

List<TupleDef*> *ProgramDef::getTuples() {
	
	List<TupleDef*> *tupleList = new List<TupleDef*>;
	
	for (int i = 0; i < components->NumElements(); i++) {
                Node *node = components->Nth(i);
                TaskDef *taskDef = dynamic_cast<TaskDef*>(node);
                if (taskDef != NULL) {
			tupleList->Append(taskDef->getEnvTuple());
			tupleList->Append(taskDef->getPartitionTuple());	
			continue;	
		}
		TupleDef *tupleDef = dynamic_cast<TupleDef*>(node);
		if (tupleDef != NULL) {
			tupleList->Append(tupleDef);
			continue;
		}
		CoordinatorDef *coordDef = dynamic_cast<CoordinatorDef*>(node);
		if (coordDef != NULL) {
			tupleList->Append(coordDef->getArgumentTuple());
		}
	}
	return tupleList;	
}

CoordinatorDef *ProgramDef::getProgramController() {
	for (int i = 0; i < components->NumElements(); i++) {
                Node *node = components->Nth(i);
		CoordinatorDef *controller = dynamic_cast<CoordinatorDef*>(node);
		if (controller != NULL) return controller;
	}
	return NULL;
}

bool ProgramDef::isIsolatedTaskProgram() {
	List<TaskDef*> *taskList = getTasks();
	if (taskList->NumElements() != 1) return false;
	return getProgramController() == NULL;
}

Scope *ProgramDef::getScope() { return symbol->getNestedScope(); }

void ProgramDef::generateExternLibraryLinkInfo(const char *linkDescriptionFile) {

        List<TaskDef*> *taskList = getTasks();
        List<const char*> *languageList = new List<const char*>;
        Hashtable<List<const char*>*> *languageLibraryMap = new Hashtable<List<const char*>*>;

        // group the library linkage annotations from different extern code block by their language
        for (int i = 0; i < taskList->NumElements(); i++) {
                TaskDef *task = taskList->Nth(i);
                IncludesAndLinksMap *externConfig = task->getExternBlocksHeadersAndLibraries();
                List<const char*> *languages = externConfig->getLanguagesUsed();
                for (int j = 0; j < languages->NumElements(); j++) {
                        const char *language = languages->Nth(j);
                        List<const char*> *libraries = NULL;
                        if (string_utils::contains(languageList, language)) {
                                libraries = languageLibraryMap->Lookup(language);
                        } else {
                                libraries = new List<const char*>;
                                languageLibraryMap->Enter(language, libraries);
                                languageList->Append(language);
                        }
                        LanguageIncludesAndLinks *languageConfig
                                        = externConfig->getIncludesAndLinksForLanguage(language);
                        string_utils::combineLists(libraries, languageConfig->getLibraryLinks());
                }
        }

        // write the libraries to be included by their language type on the library description file
        std::ofstream descriptionFile;
        descriptionFile.open (linkDescriptionFile, std::ofstream::out);
        if (!descriptionFile.is_open()) {
                std::cout << "Unable to open extern block linkage description file";
                std::exit(EXIT_FAILURE);
        }
        for (int i = 0; i < languageList->NumElements(); i++) {
                const char *language = languageList->Nth(i);
                descriptionFile << language << " =";
                List<const char*> *libraryLinks = languageLibraryMap->Lookup(language);
                for (int j = 0; j < libraryLinks->NumElements(); j++) {
                        descriptionFile << ' '  << '-' << libraryLinks->Nth(j);
                }
                descriptionFile << '\n';
        }
        descriptionFile.close();
}

//----------------------------------------- Tuple Definition ------------------------------------------/

TupleDef::TupleDef(Identifier *i, List<VariableDef*> *c) : Definition(*i->GetLocation()) {
	Assert(i != NULL && c != NULL);
	id = i;
	components = c;
	id->SetParent(this);
	for (int j = 0; j < components->NumElements(); j++) {
		VariableDef *var = components->Nth(j);
		var->SetParent(this);	
	}
	environment = false;
}

void TupleDef::PrintChildren(int indentLevel) {
	id->Print(indentLevel);
	components->PrintAll(indentLevel+1);
    	printf("\n");
}

void TupleDef::attachScope(Scope *parentScope) {
	
	Scope *scope = new Scope(TupleScope);
	List<Type*> *elementTypes = new List<Type*>;
	
	for (int i = 0; i < components->NumElements(); i++) {
		VariableDef *element = components->Nth(i);
		VariableSymbol *varSym = new VariableSymbol(element);
		if (scope->lookup(element->getId()->getName()) != NULL) {
			ReportError::ConflictingDefinition(element->getId(), false);
		}
		scope->insert_symbol(varSym);
		elementTypes->Append(varSym->getType());		
	}

	symbol = new TupleSymbol(id, this, elementTypes);
	symbol->setNestedScope(scope);
	parentScope->insert_symbol(symbol);
}

void TupleDef::validateScope(Scope *parentScope) {
	for (int i = 0; i < components->NumElements(); i++) {
		VariableDef *element = components->Nth(i);
		element->validateScope(parentScope);
	}	
}

VariableDef *TupleDef::getComponent(const char *name) {
	for (int i = 0; i < components->NumElements(); i++) {
		VariableDef *element = components->Nth(i);
		if (strcmp(element->getId()->getName(), name) == 0) return element;
	}
	return NULL;	
}

//---------------------------------------- Coordinator/Main Definition --------------------------------------/

CoordinatorDef::CoordinatorDef(Identifier *a, List<Stmt*> *c, yyltype loc) : Definition(loc) {
	Assert(a != NULL && c != NULL);
	argument = a;
	code = c;
	argument->SetParent(this);
	for (int i = 0; i < code->NumElements(); i++) {
		Stmt *stmt = code->Nth(i);
		stmt->SetParent(this);
	}
	executionScope = NULL;
}

void CoordinatorDef::PrintChildren(int indentLevel) {
	if (argument != NULL) argument->Print(indentLevel + 1, "(Argument) ");
	PrintLabel(indentLevel + 1, "Code");
	code->PrintAll(indentLevel + 2);
}

void CoordinatorDef::validateScope(Scope *parentScope) {
	
	// Create a scope for holding the command line arguments
	Scope *scope = parentScope->enter_scope(new Scope(ProgramScope));
	VariableDef *commandLineArg = new VariableDef(argument, new MapType(*argument->GetLocation()));
	scope->insert_symbol(new VariableSymbol(commandLineArg));
	
	// Create the execution scope attached to the command line arguments' scope
	executionScope = scope->enter_scope(new Scope(CoordinatorScope));

	// do semantic analysis of the coordinator program
	for (int j = 0; j < code->NumElements(); j++) {
		Stmt *stmt = code->Nth(j);
		stmt->checkSemantics(executionScope, true);
	}
	
	// Type inference procedure needs to be more elaborate and probably a recursive type inference
	// policy need to be in place. Then we do type inference as long as new expressions are getting
	// their proper type and stop the recursion as we reach a fixed point. Current solution of
	// doing the type inference only twice is a patch up solution.
	for (int j = 0; j < code->NumElements(); j++) {
		Stmt *stmt = code->Nth(j);
		stmt->performTypeInference(executionScope);
	}
	for (int j = 0; j < code->NumElements(); j++) {
		Stmt *stmt = code->Nth(j);
		stmt->performTypeInference(executionScope);
	}

	for (int j = 0; j < code->NumElements(); j++) {
		Stmt *stmt = code->Nth(j);
		stmt->checkSemantics(executionScope, false);
	}

	// generate the definition for a structure to hold the arguments for the coordinator function
	MapType *argumentMap = (MapType*) commandLineArg->getType();
	List<VariableDef*> *argumentList = argumentMap->getElementList();
	Identifier *tupleId = new Identifier(*argument->GetLocation(), "ProgramArgs");
	argumentTuple = new TupleDef(tupleId, argumentList);
}

void CoordinatorDef::declareVariablesInScope(std::ostringstream &stream, int indent) {
	executionScope->declareVariables(stream, indent);
}

void CoordinatorDef::generateCode(std::ostringstream &stream, Scope *scope) {

	executionScope = scope->enter_scope(executionScope);

	// First perform an variable access analysis to set up the flags on fields properly 
	// concerning whether they represent accessing metadata or data of a variable. TODO 
	// this is a patch solution to use variable access analysis here in this manner. We 
	// should refactor the logic in the future.
	TaskGlobalReferences *references = new TaskGlobalReferences(executionScope);
	for (int j = 0; j < code->NumElements(); j++) {
                Stmt *stmt = code->Nth(j);
        	stmt->getAccessedGlobalVariables(references);
        }

	// set the context for code translation to coordinator function
	codecntx::enterCoordinatorContext();

	// Then generate code.
	stream << "\t//------------------------------------------ Coordinator Program\n\n";		
	for (int j = 0; j < code->NumElements(); j++) {
		Stmt *stmt = code->Nth(j);
		stmt->generateCode(stream, 1);
	}
	stream << "\n\t//--------------------------------------------------------------\n";
}

//----------------------------------------- Function Definition ------------------------------------------/

FunctionHeader::FunctionHeader(List<VariableDef*> *a, List<VariableDef*> *r) : Node() {
	Assert(a != NULL && r != NULL);
	arguments = a;
	results = r;	
	for (int i = 0; i < arguments->NumElements(); i++) {
		VariableDef *var = arguments->Nth(i);
		var->SetParent(this);
	}
	for (int i = 0; i < results->NumElements(); i++) {
		VariableDef *var = arguments->Nth(i);
		var->SetParent(this);
	}
}

void FunctionHeader::PrintChildren(int indentLevel) {
	PrintLabel(indentLevel + 1, "Input Arguments");
	arguments->PrintAll(indentLevel + 2);
	PrintLabel(indentLevel + 1, "Result Elements");	
	results->PrintAll(indentLevel + 2);
}

FunctionDef::FunctionDef(Identifier *i, FunctionHeader *h, List<Stmt*> *c) : Definition(*i->GetLocation()) {
	Assert(i != NULL && h != NULL && c != NULL);
	id = i;
	id->SetParent(this);
	header = h;
	header->SetParent(this);
	code = c;
	for (int j = 0; j < code->NumElements(); j++) {
		Stmt *stmt = code->Nth(j);
		stmt->SetParent(this);
	}
}

void FunctionDef::PrintChildren(int indentLevel) {
	id->Print(indentLevel + 1, "(Name) ");
	header->Print(indentLevel + 1);
	PrintLabel(indentLevel + 1, "(Code) ");
	code->PrintAll(indentLevel + 2);
}

void FunctionDef::attachScope(Scope *parentScope) {

	Scope *scope = new Scope(FunctionScope);
	
	List<Type*> *argumentTypes = new List<Type*>;
	List<VariableDef*> *arguments = header->getArguments();
	for (int i = 0; i < arguments->NumElements(); i++) {
		VariableDef *var = arguments->Nth(i);
		argumentTypes->Append(var->getType());
		VariableSymbol *varSym = new VariableSymbol(var);
		scope->insert_symbol(varSym);	
	}
	
	char *resultTypeName = (char *) malloc(strlen(id->getName()) + 7);
	strcpy(resultTypeName, id->getName());
	strcat(resultTypeName, "Result");
	Identifier *returnTypeId = new Identifier(*id->GetLocation(), resultTypeName);
	List<VariableDef*> *returns = header->getResults();
	TupleDef *returnType = new TupleDef(returnTypeId, returns);
	returnType->attachScope(parentScope);
	parentScope->insert_symbol(returnType->getSymbol());
	
	for (int i = 0; i < returns->NumElements(); i++) {
		VariableDef *var = returns->Nth(i);
		VariableSymbol *varSym = new VariableSymbol(var);
		scope->insert_symbol(varSym);	
	}

	symbol = new FunctionSymbol(this, returnType, argumentTypes);
	symbol->setNestedScope(scope);
	parentScope->insert_symbol(symbol);
}

void FunctionDef::validateScope(Scope *parentScope) {

	List<VariableDef*> *arguments = header->getArguments();
	for (int i = 0; i < arguments->NumElements(); i++) {
		VariableDef *var = arguments->Nth(i);
		var->validateScope(parentScope);
	}
	
	List<VariableDef*> *returns = header->getResults();
	for (int i = 0; i < returns->NumElements(); i++) {
		VariableDef *var = returns->Nth(i);
		var->validateScope(parentScope);
	}

	Scope *parameterScope = parentScope->enter_scope(symbol->getNestedScope());
	Scope *executionScope = parameterScope->enter_scope(new Scope(FunctionBodyScope));

	// do semantic analysis of the function body
	for (int j = 0; j < code->NumElements(); j++) {
		Stmt *stmt = code->Nth(j);
		stmt->checkSemantics(executionScope, true);
	}
	
	for (int j = 0; j < code->NumElements(); j++) {
		Stmt *stmt = code->Nth(j);
		stmt->performTypeInference(executionScope);
	}
	for (int j = 0; j < code->NumElements(); j++) {
		Stmt *stmt = code->Nth(j);
		stmt->performTypeInference(executionScope);
	}

	for (int j = 0; j < code->NumElements(); j++) {
		Stmt *stmt = code->Nth(j);
		stmt->checkSemantics(executionScope, false);
	}

	executionScope->detach_from_parent();
	scope = executionScope;
}
