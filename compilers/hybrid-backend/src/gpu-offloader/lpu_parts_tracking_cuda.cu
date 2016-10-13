#include "lpu_parts_tracking.h"
#include "../utils/list.h"
#include "../gpu-utils/gpu_utils.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdlib>

/************************************************************************************************************************************************* 
                             Note that *_cuda.cu files have the code that needs the NVCC compiler for compilation
*************************************************************************************************************************************************/

//----------------------------------------------------------- Property Buffer Manager -----------------------------------------------------------/

void PropertyBufferManager::prepareGpuBuffers() {
	
	check_error(cudaMalloc((void **) &gpuBuffer, bufferSize), *logFile);
	check_error(cudaMemcpyAsync(gpuBuffer, cpuBuffer, bufferSize, cudaMemcpyHostToDevice, 0), *logFile);
	
	check_error(cudaMalloc((void **) &gpuPartIndexBuffer, bufferReferenceCount * sizeof(int)), *logFile);
	check_error(cudaMemcpyAsync(gpuPartIndexBuffer, cpuPartIndexBuffer, 
			bufferReferenceCount * sizeof(int), cudaMemcpyHostToDevice, 0), *logFile);
	
	check_error(cudaMalloc((void **) &gpuPartRangeBuffer, partRangeBufferSize * sizeof(int)), *logFile);
	check_error(cudaMemcpyAsync(gpuPartRangeBuffer, cpuPartRangeBuffer, 
			partRangeBufferSize * sizeof(int), cudaMemcpyHostToDevice, 0), *logFile);
	
	check_error(cudaMalloc((void **) &gpuPartBeginningBuffer, 
			bufferEntryCount * sizeof(long int)), *logFile);
	check_error(cudaMemcpyAsync(gpuPartBeginningBuffer, cpuPartBeginningBuffer, 
			bufferEntryCount * sizeof(long int), cudaMemcpyHostToDevice, 0), *logFile);
}

void PropertyBufferManager::syncDataPartsFromBuffer(List<LpuDataPart*> *dataPartsList) {
	
	check_error(cudaMemcpy(cpuBuffer, gpuBuffer, bufferSize, cudaMemcpyDeviceToHost), *logFile);

	int currentIndex = 0;
	for (int i = 0; i < dataPartsList->NumElements(); i++) {
		LpuDataPart *dataPart = dataPartsList->Nth(i);
		char *dataStart = cpuBuffer + currentIndex;
		void *data = dataPart->getData();
		int size = dataPart->getSize();
		memcpy(data, dataStart, size);
		currentIndex += size;
	}
}

void PropertyBufferManager::cleanupBuffers() {
	
	bufferSize = 0;
	bufferEntryCount = 0;
	bufferReferenceCount = 0;
	partRangeDepth = 0;
	
	free(cpuBuffer);
	cpuBuffer = NULL;
	free(cpuPartIndexBuffer); 
        cpuPartIndexBuffer = NULL;
	free(cpuPartRangeBuffer); 
        cpuPartRangeBuffer = NULL;
	free(cpuPartBeginningBuffer);
        cpuPartBeginningBuffer = NULL;
       
	cudaFree(gpuBuffer); 
	gpuBuffer = NULL;
	cudaFree(gpuPartIndexBuffer);
        gpuPartIndexBuffer = NULL;
	cudaFree(gpuPartRangeBuffer);
        gpuPartRangeBuffer = NULL;
	cudaFree(gpuPartBeginningBuffer);
        gpuPartBeginningBuffer = NULL;
}

//------------------------------------------------------- Versioned Property Buffer Manager -----------------------------------------------------/

void VersionedPropertyBufferManager::prepareGpuBuffers() {
	PropertyBufferManager::prepareGpuBuffers();
	check_error(cudaMalloc((void **) &gpuDataPartVersions, bufferEntryCount * sizeof(short)), *logFile);
	check_error(cudaMemcpyAsync(gpuDataPartVersions, cpuDataPartVersions, 
			bufferEntryCount * sizeof(short), cudaMemcpyHostToDevice, 0), *logFile);
}

void VersionedPropertyBufferManager::syncDataPartsFromBuffer(List<LpuDataPart*> *dataPartsList) {

	check_error(cudaMemcpyAsync(cpuBuffer, gpuBuffer, bufferSize, cudaMemcpyDeviceToHost, 0), *logFile);
	check_error(cudaMemcpy(cpuDataPartVersions, gpuDataPartVersions, 
			bufferEntryCount * sizeof(short), cudaMemcpyDeviceToHost), *logFile);

	int currentIndex = 0;
	for (int i = 0; i < bufferEntryCount; i++) {
		VersionedLpuDataPart *dataPart = (VersionedLpuDataPart *) dataPartsList->Nth(i);
		short currVersion = cpuDataPartVersions[i];
		int sizePerVersion = dataPart->getSize() / versionCount;
		for (int j = 0; j < versionCount; j++) {
			int versionIndex = (currVersion + j) % versionCount;
			char *dataStart = cpuBuffer + currentIndex + versionIndex * sizePerVersion;
			void *data = dataPart->getDataVersion(versionIndex);
			memcpy(data, dataStart, sizePerVersion);
		}
		currentIndex += dataPart->getSize();
	}
}

void VersionedPropertyBufferManager::cleanupBuffers() {
	
	PropertyBufferManager::cleanupBuffers();

	free(cpuDataPartVersions);
        cpuDataPartVersions = NULL;       
	cudaFree(gpuDataPartVersions); 
	gpuDataPartVersions = NULL;
}
