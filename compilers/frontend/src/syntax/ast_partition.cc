#include "ast.h"
#include "ast_task.h"
#include "ast_partition.h"
#include "../common/errors.h"
#include "../common/constant.h"
#include "../semantics/symbol.h"
#include "../semantics/task_space.h"
#include "../../../common-libs/utils/list.h"

//----------------------------------------- Linking Spaces Together ---------------------------------------------/

SpaceLinkage::SpaceLinkage(PartitionLinkType l, char p, yyltype loc) : Node(loc) {
        linkType = l;
        parentSpace = p;
}

void SpaceLinkage::PrintChildren(int indentLevel) {
        PrintLabel(indentLevel + 1, "Type");
        printf((linkType == LinkTypePartition) ? "Partition" : "Sup-partition");
        PrintLabel(indentLevel + 1, "Parent");
        printf("%c", parentSpace);
}

Space *SpaceLinkage::getParentSpace(PartitionHierarchy *partitionHierarchy) {
        if (linkType == LinkTypeSubpartition) {
                return partitionHierarchy->getSubspace(parentSpace);
        } else {
                return partitionHierarchy->getSpace(parentSpace);
        }
}
        
//-------------------------------------------- Partition Argument -----------------------------------------------/

PartitionArg::PartitionArg(Identifier *i) : Node(*i->GetLocation()) {
        Assert(i != NULL);
        constant = false;
        id = i;
        id->SetParent(this);
        value = NULL;
}

PartitionArg::PartitionArg(IntConstant *v) : Node(*v->GetLocation()) {
        Assert(v != NULL);
        constant = true;
        id = NULL;
        value = v;
        value->SetParent(this);
}

void PartitionArg::PrintChildren(int indentLevel) {
        if (id != NULL) id->Print(indentLevel + 1);
        if (value != NULL) value->Print(indentLevel + 1);
}

Node *PartitionArg::getContent() {
        if (id != NULL) return id;
        else return value;
}

void PartitionArg::validateScope(Scope *partitionScope) {
        if (id != NULL) {
                Symbol *symbol = partitionScope->lookup(id->getName());
                if (symbol == NULL) {
                        ReportError::InvalidPartitionArgument(GetLocation());
                }
        }
}

//----------------------------------------- Partition Instruction -----------------------------------------------/

PartitionInstr::PartitionInstr(yyltype loc) : Node(loc) {
        replicated = true;
        partitionFn = NULL;
        dividingArgs = NULL;
        padded = false;
        paddingArgs = NULL;
        order = RandomOrder;
}

PartitionInstr::PartitionInstr(Identifier *pF, List<PartitionArg*> *dA,
                bool p, List<PartitionArg*> *pA, yyltype loc) : Node(loc) {
        Assert(pF != NULL && dA != NULL);
        replicated = false;
        partitionFn = pF;
        partitionFn->SetParent(this);
        dividingArgs = dA;
        for (int i = 0; i < dividingArgs->NumElements(); i++) {
                dividingArgs->Nth(i)->SetParent(this);
        }
        padded = p;
        paddingArgs = pA;
        if (paddingArgs != NULL) {
                for (int i = 0; i < paddingArgs->NumElements(); i++) {
                        paddingArgs->Nth(i)->SetParent(this);
                }
        }
        order = RandomOrder;
}

void PartitionInstr::PrintChildren(int indentLevel) {
        if (replicated) printf("replicated");
        else {
                partitionFn->Print(indentLevel + 1, "(Function) ");
                PrintLabel(indentLevel + 1, "PartitionArgs");
                dividingArgs->PrintAll(indentLevel + 2);
                PrintLabel(indentLevel + 1, "Order");
                if (order == AscendingOrder) printf("Ascending");
                else if (order == DescendingOrder) printf("Descending");
                else printf("Random");
                PrintLabel(indentLevel + 1, "Padding");
                printf((padded) ? "True" : "False");
                if (paddingArgs != NULL) {
                        PrintLabel(indentLevel + 1, "PaddingArgs");
                        paddingArgs->PrintAll(indentLevel + 2);
                }
        }
}

PartitionFunctionConfig *PartitionInstr::generateConfiguration(List<int> *dataDimensions,
                        int dimensionAccessStartIndex, Scope *partitionScope) {

        for (int i = 0; i < dividingArgs->NumElements(); i++) {
                PartitionArg *arg = dividingArgs->Nth(i);
                arg->validateScope(partitionScope);
        }
        if (paddingArgs != NULL) {
                for (int i = 0; i < paddingArgs->NumElements(); i++) {
                        PartitionArg *arg = paddingArgs->Nth(i);
                        arg->validateScope(partitionScope);
                }
        }

        PartitionFunctionConfig *config = PartitionFunctionConfig::generateConfig(
                        partitionFn->GetLocation(),
                        partitionFn->getName(), dividingArgs, paddingArgs);
        int dimensionCount = config->getDimensionality();
        List<int> *dimensionIds = new List<int>;
        for (int i = 0; i < dimensionCount; i++) {
                int index = dimensionAccessStartIndex + i;
                if (index >= dataDimensions->NumElements()) {
                        ReportError::DimensionMissingOrInvalid(GetLocation());
                } else {
                        dimensionIds->Append(dataDimensions->Nth(index));
                }
        }

        // replace the default incremental dimension ID list that starts from 1 with the correct ID list 
        config->setDimensionIds(dimensionIds);
        config->setPartitionOrder(order);
        return config;
}

//--------------------------- A Single Data Structure's Partition Specification ---------------------------------/

DataConfigurationSpec::DataConfigurationSpec(Identifier *v, List<IntConstant*> *d,
                List<PartitionInstr*> *i, SpaceLinkage *p) : Node(*v->GetLocation()) {
        Assert(v != NULL);
        variable = v;
        variable->SetParent(this);
        dimensions = d;
        if (dimensions != NULL) {
                for (int j = 0; j < dimensions->NumElements(); j++) {
                        dimensions->Nth(j)->SetParent(this);
                }
        }
        instructions = i;
        if (instructions != NULL) {
                for (int j = 0; j < instructions->NumElements(); j++) {
                        instructions->Nth(j)->SetParent(this);
                }
        }
        parentLink = p;
        if (parentLink != NULL) {
                parentLink->SetParent(this);
        }
}

List<DataConfigurationSpec*> *DataConfigurationSpec::decomposeDataConfig(List<VarDimensions*> *varList,
                List<PartitionInstr*> *instrList, SpaceLinkage *parentLink) {
        List<DataConfigurationSpec*> *specList = new List<DataConfigurationSpec*>;
        for (int i = 0; i < varList->NumElements(); i++) {
                VarDimensions *vD = varList->Nth(i);
                DataConfigurationSpec *spec
                        = new DataConfigurationSpec(vD->GetVar(), vD->GetDimensions(), instrList, parentLink);
                specList->Append(spec);
        }
        return specList;
}

void DataConfigurationSpec::PrintChildren(int indentLevel) {
        variable->Print(indentLevel + 1, "(Array) ");
        if (parentLink != NULL) parentLink->Print(indentLevel + 1);
        if (dimensions != NULL) {
                PrintLabel(indentLevel + 1, "Dimensions");
                dimensions->PrintAll(indentLevel + 2);
        }
        if (instructions != NULL) {
                PrintLabel(indentLevel + 1, "Instructions");
                instructions->PrintAll(indentLevel + 2);
        }
}

ArrayDataStructure* DataConfigurationSpec::addPartitionConfiguration(Space *space, 
		Scope *partitionScope,
                PartitionHierarchy *partitionHierarchy) {

        // first determine the source of the data structure under concern
        Space *spaceParent = space->getParent();
        const char* variableName = variable->getName();
        DataStructure *structure = spaceParent->getStructure(variableName);

        // if there is a data structure specific hierarchy here then validate it
        if (parentLink != NULL) {
                Space *dataSourceParent = parentLink->getParentSpace(partitionHierarchy);
                if (dataSourceParent == NULL) {
                        ReportError::ParentForDataStructureNotFound(GetLocation(), 
					parentLink->getParentId(), variableName);
                } else {
                        if (spaceParent->getLocalStructure(variableName) != NULL
                                        && spaceParent != dataSourceParent
                                        && spaceParent != partitionHierarchy->getRootSpace()) {
                                ReportError::SpaceAndDataHierarchyConflict(variable,
                                                spaceParent->getName(), dataSourceParent->getName());
                        } else if (dataSourceParent->getLocalStructure(variableName) == NULL) {
                                ReportError::ParentDataStructureNotFound(GetLocation(),
                                                parentLink->getParentId(), variableName);
                        }
                        structure = dataSourceParent->getStructure(variableName);
                }
        }
	
        if (structure == NULL) {
                ReportError::NonTaskGlobalArrayInPartitionSection(variable);
                return NULL;
	}

	ArrayDataStructure *arrayStruct = dynamic_cast<ArrayDataStructure*>(structure);
	if (arrayStruct == NULL) {
		ReportError::NonTaskGlobalArrayInPartitionSection(variable);
		return NULL;
	}

	// determine the dimensions of the data structure that will be aligned with the coordinates 
	// of the space and partitioned along them 
	List<int> *sourceDimensions = arrayStruct->getRemainingDimensions();
	List<int> *partitionDimensions = NULL;

	// There is a possible reordering of dimensions for the sake of partitioning when dimension 
	// IDs are provided in the source code. Therefore, a validation is needed regarding the 
	// feasibility of the partition along specified dimension. Furthermore, we need to associate 
	// appropriate dimensions with the partition functions in place of the default linearly 
	// increasing order based assignments.               
	if (dimensions != NULL && dimensions->NumElements() > 0) {
		partitionDimensions = new List<int>;
		for (int i = 0; i < dimensions->NumElements(); i++) {
			int value = dimensions->Nth(i)->getValue();
			bool matchingFound = false;
			for (int j = 0; j < sourceDimensions->NumElements(); j++) {
				if (sourceDimensions->Nth(j) == value) {
					matchingFound = true;
					break;
				}
			}
			if (!matchingFound) {
				ReportError::DimensionMissingOrInvalid(variable, value);
			} else {
				partitionDimensions->Append(value);
			}
		}
	// When no reordering is done. So no extra validation or processing is required like in the 
	// previous case.
	} else {
		partitionDimensions = sourceDimensions;
	}

	// initialize variables for definining the partitioning of the data structure for the 
	// current space 
	ArrayDataStructure *newDef = new ArrayDataStructure(arrayStruct);
	Type *type = newDef->getType();
	List<int> *blockedDimensions = new List<int>;
	int coordinateDimensionsInSpace = space->getDimensionCount();
	int currentCoordinate = 1;
	int currentDataDimensionIndex = 0;

	// iterate over the partition instructions and translate them to a form more suitable for 
	// latter analysis
	for (int i = 0; i <instructions->NumElements(); i++) {
		if (currentCoordinate > coordinateDimensionsInSpace) {
			ReportError::TooFineGrainedVariablePartition(variable);
			break;
		}
		PartitionInstr *instr = instructions->Nth(i);
		if (instr->isInstructedForReplication()) {
			Token *token = new Token(newDef, Token::wildcardTokenId);
			space->storeToken(currentCoordinate, token);
			currentCoordinate++;
			currentDataDimensionIndex++;
		} else {
			// a partition function specification does its own validation
			PartitionFunctionConfig *pFnConfig = instr->generateConfiguration(
					partitionDimensions,
					currentDataDimensionIndex, partitionScope);
			List<int> *dimsFurtherPartitioned = pFnConfig->getPartitionedDimensions();
			for (int j = 0; j < dimsFurtherPartitioned->NumElements(); j++) {
				if (currentCoordinate > coordinateDimensionsInSpace) {
					ReportError::TooFineGrainedVariablePartition(variable);
					break;
				}
				Token *token = new Token(newDef, dimsFurtherPartitioned->Nth(j));
				space->storeToken(currentCoordinate, token);
				currentCoordinate++;
			}
			blockedDimensions->AppendAll(pFnConfig->getBlockedDimensions(type));
			currentDataDimensionIndex += dimsFurtherPartitioned->NumElements();
			newDef->addPartitionSpec(pFnConfig);
		}
	}

	// Set the before and after partition dimension information of the data structure. Note that 
	// both of these dimension lists are sorted and kept that way so that the validation logic
	// of the above can be applied to each space on the hierarchy one-after-another without any 
	// additional reordering of these lists. 
	newDef->setSourceDimensions(sourceDimensions);
	List<int> *afterPartitionDimensions = new List<int>;
	for (int i = 0; i < sourceDimensions->NumElements(); i++) {
		int dim = sourceDimensions->Nth(i);
		bool isBlocked = false;
		for (int j = 0; j < blockedDimensions->NumElements(); j++) {
			if (blockedDimensions->Nth(j) == dim) {
				isBlocked = true;
				break;
			}
		}
		if (!isBlocked) afterPartitionDimensions->Append(dim);
	}
	newDef->setAfterPartitionDimensions(afterPartitionDimensions);

	// store the new data structure definition in the space and return it
	newDef->setSpaceReference(space);
	space->addDataStructure(newDef);
	return newDef;
}

//----------------------------------- Subpartitioning Space Configuration ---------------------------------------/

SubpartitionSpec::SubpartitionSpec(int d, bool o, List<DataConfigurationSpec*> *sl, yyltype loc) : Node(loc) {
        Assert(sl != NULL);
        dimensionality = d;
        ordered = o;
        specList = sl;
        for (int i = 0; i < specList->NumElements(); i++) {
                specList->Nth(i)->SetParent(this);
        }
}

void SubpartitionSpec::PrintChildren(int indentLevel) {
        PrintLabel(indentLevel + 1, "Dimensions");
        printf("%d", dimensionality);
        PrintLabel(indentLevel + 1, "Ordered");
        printf(ordered ? "True" : "False");
        specList->PrintAll(indentLevel + 1);
}

// The logic for subpartitioned space configuration validation is similar to a normal space validation. A further
// restriction is that subpartitioned spaces will always have a non-zero dimension. Therefore an extra validation 
// for possibly unpartitioned space configuration is skipped. 
void SubpartitionSpec::addSpaceConfiguration(TaskDef *taskDef, 
		PartitionHierarchy *currentHierarchy, 
		Space *ownerSpace) {

        int suffixLength = strlen(Space::SubSpaceSuffix);
        char *spaceName = (char *) malloc(sizeof(char) * (suffixLength + 2));
        strcpy(spaceName, ownerSpace->getName());
        strcat(spaceName, Space::SubSpaceSuffix);
        Space *space = new Space(spaceName, dimensionality, false, true);
        space->setParent(ownerSpace);
        ownerSpace->setSubpartition(space);

        TaskSymbol *taskSymbol = (TaskSymbol *) taskDef->getSymbol();
        Scope *partitionScope = taskSymbol->getPartitionScope();

        if (dimensionality <= 0) {
                ReportError::SubpartitionDimensionsNotPositive(GetLocation());
        }
        for (int i = 0; i < specList->NumElements(); i++) {
                ArrayDataStructure *structure = specList->Nth(i)->addPartitionConfiguration(space,
                                partitionScope, currentHierarchy);
                if (structure != NULL) {
                        if (!ordered && structure->isOrderDependent()) {
                                ReportError::SubpartitionOrderConflict(specList->Nth(i)->GetLocation());
                        }
                        if (ownerSpace->getLocalStructure(structure->getName()) == NULL) {
                                ReportError::SubpartitionedStructureMissingInParentSpace(
                                                specList->Nth(i)->GetLocation(), structure->getName());
                        } else {
                                ownerSpace->getLocalStructure(structure->getName())->flagAsNonStorable();
                        }
                }
        }
        if (space->isValidCoordinateSystem() == false) {
                ReportError::InvalidSpaceCoordinateSystem(GetLocation(), 
				ownerSpace->getName()[0], dimensionality, true);
        }
        currentHierarchy->addNewSpace(space);
}

//-------------------------------------- A Single Space Configuration ------------------------------------------/

PartitionSpec::PartitionSpec(char si, int d, List<DataConfigurationSpec*> *sl, bool dy,
                SpaceLinkage *pl, SubpartitionSpec *sp, yyltype loc) : Node(loc) {
        Assert(sl != NULL && d > 0);
        spaceId = si;
        dimensionality = d;
        specList = sl;
        for (int i = 0; i < specList->NumElements(); i++) {
                specList->Nth(i)->SetParent(this);
        }
        dynamic = dy;
        parentLink = pl;
        if (parentLink != NULL) {
                parentLink->SetParent(this);
        }
        subpartition = sp;
        if (subpartition != NULL) {
                subpartition->SetParent(this);
        }
        variableList = NULL;
}

PartitionSpec::PartitionSpec(char s, List<Identifier*> *v, yyltype loc) : Node(loc) {
        Assert(v != NULL);
        spaceId = s;
        variableList = v;
        for (int i = 0; i < variableList->NumElements(); i++) {
                variableList->Nth(i)->SetParent(this);
        }
        dimensionality = 0;
        specList = NULL;
        dynamic = false;
        parentLink = NULL;
        subpartition = NULL;
}

void PartitionSpec::PrintChildren(int indentLevel) {
        PrintLabel(indentLevel + 1, "Id");
        printf("%c", spaceId);
        PrintLabel(indentLevel + 1, "Dimensions");
        if (dimensionality > 0) printf("%d", dimensionality);
        else printf("N/A");
        PrintLabel(indentLevel + 1, "Dynamic");
        printf(dynamic ? "True" : "False");
        if (parentLink != NULL) parentLink->Print(indentLevel + 1);
        if (subpartition != NULL) subpartition->Print(indentLevel + 1);
        if (variableList != NULL) variableList->PrintAll(indentLevel + 1);
        if (specList != NULL) specList->PrintAll(indentLevel + 1);
}

void PartitionSpec::addSpaceConfiguration(TaskDef *taskDef, PartitionHierarchy *currentHierarchy) {

        char *spaceName = (char *) malloc(sizeof(char) * 2);
        spaceName[0] = spaceId;
        spaceName[1] = '\0';
        Space *space = new Space(spaceName, dimensionality, dynamic, false);
        if (currentHierarchy->addNewSpace(space) == false) {
                ReportError::DuplicateSpaceDefinition(GetLocation(), spaceId);
        }

        // If there is a parent-child relationship between current and another space then the parent space must
        // be specificied before this one to do proper validation of data structure partitions.
        if (this->parentLink != NULL) {
                Space *parentSpace = parentLink->getParentSpace(currentHierarchy);
                if (parentSpace == NULL) {
                        ReportError::ParentSpaceNotFound(parentLink->GetLocation(), parentLink->getParentId(),
                                        spaceId, parentLink->linkedToSubpartition());
                        space->setParent(currentHierarchy->getRootSpace());
                        currentHierarchy->getRootSpace()->addChildSpace(space);
                } else {
                        space->setParent(parentSpace);
                        parentSpace->addChildSpace(space);
                }
        } else {
                space->setParent(currentHierarchy->getRootSpace());
                currentHierarchy->getRootSpace()->addChildSpace(space);
        }
	
	// Unpartitioned spaces should only specify the list of variables they hold without any partition
        // specification (which will contradict the nature of the space) for any data structure included within.
        if (variableList != NULL) {
                if (dimensionality > 0) {
                        ReportError::UnpartitionedDataInPartitionedSpace(GetLocation(), spaceId, dimensionality);
                }
                Hashtable<DataStructure*> *structureList = new Hashtable<DataStructure*>;
                for (int i = 0; i < variableList->NumElements(); i++) {
                        Identifier *varId = variableList->Nth(i);
                        DataStructure *structure = space->getStructure(varId->getName());
                        if (structure == NULL) {
                                ReportError::NonTaskGlobalArrayInPartitionSection(varId);
                        } else {
                                ArrayDataStructure *arrayStruct = dynamic_cast<ArrayDataStructure*>(structure);
                                if (arrayStruct == NULL) {
                                        ReportError::NonTaskGlobalArrayInPartitionSection(varId);
                                } else {
                                        ArrayDataStructure *newDef = new ArrayDataStructure(arrayStruct);
                                        newDef->setSpaceReference(space);
                                        structureList->Enter(structure->getName(), newDef, false);
                                }
                        }
                }
                space->setStructureList(structureList);
	
	// For a partitioned space, on the other hand, detail validation is needed regarding the dimensionality
        // matching of the space itself with the dimensionality of the data structure partitions. 
        } else {
                space->initEmptyStructureList();
                TaskSymbol *taskSymbol = (TaskSymbol *) taskDef->getSymbol();
                Scope *partitionScope = taskSymbol->getPartitionScope();
                for (int i = 0; i < specList->NumElements(); i++) {
                        // each partition specification validates itself as it has been added to the space
                        specList->Nth(i)->addPartitionConfiguration(space, partitionScope, currentHierarchy);
                }
        }
        if (space->isValidCoordinateSystem() == false) {
                ReportError::InvalidSpaceCoordinateSystem(GetLocation(), spaceId, dimensionality, false);
        }

        if (subpartition != NULL) {
                subpartition->addSpaceConfiguration(taskDef, currentHierarchy, space);
        }
}

//-------------------------------------------- Partition Section -----------------------------------------------/

PartitionSection::PartitionSection(List<Identifier*> *a, List<PartitionSpec*> *s, yyltype loc) : Node(loc) {
        
	Assert(a != NULL && s != NULL);
        arguments = a;
        for (int i = 0; i < arguments->NumElements(); i++) {
                arguments->Nth(i)->SetParent(this);
        }
        spaceSpecs = s;
        for (int i = 0; i < spaceSpecs->NumElements(); i++) {
                spaceSpecs->Nth(i)->SetParent(this);
        }

	partitionHierarchy = NULL;
}

void PartitionSection::PrintChildren(int indentLevel) {
        PrintLabel(indentLevel + 1, "Arguments");
        arguments->PrintAll(indentLevel + 2);
        spaceSpecs->PrintAll(indentLevel + 1);
}

void PartitionSection::constructPartitionHierarchy(TaskDef *taskDef) {

        PartitionHierarchy *hierarchy = new PartitionHierarchy();
        Space *rootSpace = new Space(Space::RootSpaceName, 0, false, false);

        // create an un-partitioned  root space holding all the task global variables
        Hashtable<DataStructure*> *rootStructureList = new Hashtable<DataStructure*>;
        List<VariableDef*> *taskGlobals = taskDef->getDefineSection()->getDefinitions();
        for (int i = 0; i < taskGlobals->NumElements(); i++) {
                DataStructure *structure = NULL;
                VariableDef *var = taskGlobals->Nth(i);
                Type *type = var->getType();
                ArrayType *arrayType = dynamic_cast<ArrayType*>(type);
                if (arrayType == NULL) {
                        structure = new DataStructure(var);
                } else {
                        structure = new ArrayDataStructure(var);
                }
                structure->setSpaceReference(rootSpace);
                rootStructureList->Enter(structure->getName(), structure, false);
        }
        rootSpace->setStructureList(rootStructureList);
        hierarchy->addNewSpace(rootSpace);

        // iterate through the spaces in the partition configuration and populate the hierarchy
        for (int i = 0; i < spaceSpecs->NumElements(); i++) {
                PartitionSpec *spec = spaceSpecs->Nth(i);
                spec->addSpaceConfiguration(taskDef, hierarchy);
        }

        // set the local variable for the partition hierarchy
        partitionHierarchy = hierarchy;
}
