#include "code_constant.h"

#include <string>

//--------------------------------------------------------- GPU Code Constants ----------------------------------------------------

GpuCodeConstants *GpuCodeConstants::getConstantsForGpuLevel() {
	
	GpuCodeConstants *codeConst = new GpuCodeConstants();
	
	codeConst->storageSuffix = std::string("");
	codeConst->storageIndex = std::string("");
	codeConst->distrIndex = std::string("0");
	codeConst->localDistrIndex = std::string("0");
	codeConst->jumpExpr = std::string("1");
	codeConst->localJumpExpr = std::string("1");
	
	return codeConst;
}

GpuCodeConstants *GpuCodeConstants::getConstantsForSmLevel() {
	
	GpuCodeConstants *codeConst = new GpuCodeConstants();

	codeConst->storageSuffix = std::string("");
	codeConst->storageIndex = std::string("");
	codeConst->distrIndex = std::string("smId");
	codeConst->localDistrIndex = std::string("smId");
	codeConst->jumpExpr = std::string("SM_COUNT");	
	codeConst->localJumpExpr = std::string("SM_COUNT");	
	
	return codeConst;
}
        
GpuCodeConstants *GpuCodeConstants::getConstantsForWarpLevel() {
	
	GpuCodeConstants *codeConst = new GpuCodeConstants();
	
	codeConst->storageSuffix = std::string("[WARP_COUNT]");
	codeConst->storageIndex = std::string("[warpId]");
	codeConst->distrIndex = std::string("globalWarpId");
	codeConst->localDistrIndex = std::string("warpId");
	codeConst->jumpExpr = std::string("SM_COUNT * WARP_COUNT");
	codeConst->localJumpExpr = std::string("WARP_COUNT");
	
	return codeConst;
}
