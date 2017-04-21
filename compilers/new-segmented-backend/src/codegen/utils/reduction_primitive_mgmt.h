#ifndef _H_reduction_primitive_mgmt
#define _H_reduction_primitive_mgmt

// This header file contains library functions to generate runtime routines and data structures for implementing any
// parallel reductions found in an IT task. 

#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/domain-obj/constant.h"

#include "../../../../frontend/src/semantics/task_space.h"
#include "../../../../frontend/src/semantics/computation_flow.h"

#include <iostream>
#include <fstream>

class MappingNode;

/**********************************************************************************************************************
						Utility Functions
***********************************************************************************************************************/

const char *getMpiDataTypeStr(Type *type, ReductionOperator op);
const char *getMpiReductionOp(ReductionOperator op);
const char *getReductionOpString(ReductionOperator op);

/**********************************************************************************************************************
					Generator for Result Reset Function
***********************************************************************************************************************/

void generateResultResetFn(std::ofstream &programFile, 
		const char *initials, 
		const char *className, 
		Type *resultType, 
		ReductionOperator op);

/**********************************************************************************************************************
				Generators for Intermediate Result Update Functions' body
***********************************************************************************************************************/

// functions for different specific types of reduction operation
void generateUpdateCodeForMax(std::ofstream &programFile, std::string propertyName);
void generateUpdateCodeForMin(std::ofstream &programFile, std::string propertyName);
void generateUpdateCodeForSum(std::ofstream &programFile, std::string propertyName);
void generateUpdateCodeForProduct(std::ofstream &programFile, std::string propertyName);
void generateUpdateCodeForLand(std::ofstream &programFile, std::string propertyName);
void generateUpdateCodeForLor(std::ofstream &programFile, std::string propertyName);
void generateUpdateCodeForBand(std::ofstream &programFile, std::string propertyName);
void generateUpdateCodeForBor(std::ofstream &programFile, std::string propertyName);

// this function picks anyone of the above function, as deemed appropriate, to generate the body
void generateIntermediateResultUpdateFnBody(std::ofstream &programFile, 
		Type *varType, ReductionOperator op);

/**********************************************************************************************************************
				Generators for Perform Cross Segment Reduction Functions' body
***********************************************************************************************************************/

void generateCodeForDataReduction(std::ofstream &programFile, 
		ReductionOperator op, Type *varType);

/**********************************************************************************************************************
					Reduction Primitive Class Generators
***********************************************************************************************************************/

void generateIntraSegmentReductionPrimitive(std::ofstream &headerFile, 
		std::ofstream &programFile, 
		const char *initials, 
		ReductionMetadata *rdMetadata, 
		Space *rootLps);

void generateCrossSegmentReductionPrimitive(std::ofstream &headerFile, 
		std::ofstream &programFile, 
		const char *initials, 
		ReductionMetadata *rdMetadata, 
		Space *rootLps);

/* this function invokes the above to functions to create appropriate reduction primitive subclass for all 
   reduction operations found in the task */
void generateReductionPrimitiveClasses(const char *headerFile,
                const char *programFile,
                const char *initials,
                MappingNode *mappingRoot, 
		List<ReductionMetadata*> *reductionInfos);

/**********************************************************************************************************************
					Reduction Primitive Instantiation  
***********************************************************************************************************************/

/* this function declares all arrays of static reduction primitives in the header file */
void generateReductionPrimitiveDecls(const char *headerFile, List<ReductionMetadata*> *reductionInfos);

/* this function generates a routine that initialize all static reduction primitives of a segment */
void generateReductionPrimitiveInitFn(const char *headerFile, 
		const char *programFile, 
		const char *initials, 
		List<ReductionMetadata*> *reductionInfos);

/* this function generates a routine that a PPU controller thread can use to receive the reduction primitives 
   relevant to it */
void generateReductionPrimitiveMapCreateFnForThread(const char *headerFile,
                const char *programFile,
                const char *initials,
                List<ReductionMetadata*> *reductionInfos);

#endif
