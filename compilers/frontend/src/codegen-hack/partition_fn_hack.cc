#include "../semantics/partition_function.h"

const char *StridedBlock::getTransformedIndex(int dimensionNo, const char *origIndexName, bool copyMode) { return NULL; }
const char *StridedBlock::getOriginalIndex(int dimensionNo, const char *xformIndexName, bool copyMode) { return NULL; }
const char *StridedBlock::getInclusionTestExpr(int dimensionNo, const char *origIndexName, bool copyMode) { return NULL; }
const char *StridedBlock::getImpreciseBoundOnXformedIndex(int dimension,
		const char *index, bool lowerBound, bool copyMode, int indent) { return NULL; }

const char *Strided::getTransformedIndex(int dimensionNo, const char *origIndexName, bool copyMode) { return NULL; }
const char *Strided::getOriginalIndex(int dimensionNo, const char *xformIndexName, bool copyMode) { return NULL; }
const char *Strided::getInclusionTestExpr(int dimensionNo, const char *origIndexName, bool copyMode) { return NULL; }
const char *Strided::getImpreciseBoundOnXformedIndex(int dimension,
		const char *index, bool lowerBound, bool copyMode, int indent) { return NULL; }
