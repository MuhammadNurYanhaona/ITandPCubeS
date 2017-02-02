#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>

#include "../utils/list.h"
#include "../utils/utility.h"
#include "../utils/hashtable.h"
#include "comm_statistics.h"

CommStatistics::CommStatistics() {
	commDependencyNames = new List<const char*>;
        confinementConstrTimeMap = new Hashtable<double*>;
        bufferSetupTimeMap = new Hashtable<double*>;
        commResourcesSetupTimeMap = new Hashtable<double*>;
        bufferReadTimeMap = new Hashtable<double*>;
        communicationTimeMap = new Hashtable<double*>;
        bufferWriteTimeMap = new Hashtable<double*>;
	pthread_mutex_init(&mutex, NULL);
}

CommStatistics::~CommStatistics() {
	delete commDependencyNames;
        delete confinementConstrTimeMap;
        delete bufferSetupTimeMap;
        delete commResourcesSetupTimeMap;
	delete bufferReadTimeMap;
        delete communicationTimeMap;
	delete bufferWriteTimeMap;
	pthread_mutex_destroy(&mutex);
}

void CommStatistics::enlistDependency(const char *dependency) {
	
	pthread_mutex_lock(&mutex);
	
	commDependencyNames->Append(dependency);
	double *cCTime = new double;
	*cCTime = 0.0;
        confinementConstrTimeMap->Enter(dependency, cCTime);
	double *bSTime = new double;
	*bSTime = 0.0;
        bufferSetupTimeMap->Enter(dependency, bSTime);
	double *cRSTime = new double;
	*cRSTime = 0.0;
        commResourcesSetupTimeMap->Enter(dependency, cRSTime);
	double *bRTime = new double;
	*bRTime = 0.0;
        bufferReadTimeMap->Enter(dependency, bRTime);
	double *cTime = new double;
	*cTime = 0.0;
        communicationTimeMap->Enter(dependency, cTime);
	double *bWTime = new double;
	*bWTime = 0.0;
        bufferWriteTimeMap->Enter(dependency, bWTime);

	pthread_mutex_unlock(&mutex);
}

void CommStatistics::addConfinementConstrTime(const char *dependency, struct timeval &start, struct timeval &end) {
	recordTiming(confinementConstrTimeMap, dependency, start, end);
}
        
void CommStatistics:: addBufferSetupTime(const char *dependency, struct timeval &start, struct timeval &end) {
	recordTiming(bufferSetupTimeMap, dependency, start, end);
}
        
void CommStatistics::addCommResourcesSetupTime(const char *dependency, struct timeval &start, struct timeval &end) {
	recordTiming(commResourcesSetupTimeMap, dependency, start, end);
}
        
void CommStatistics::addBufferReadTime(const char *dependency, struct timeval &start, struct timeval &end) {
	recordTiming(bufferReadTimeMap, dependency, start, end);
}

void CommStatistics::addCommunicationTime(const char *dependency, struct timeval &start, struct timeval &end) {
	recordTiming(communicationTimeMap, dependency, start, end);
}

void CommStatistics::addBufferWriteTime(const char *dependency, struct timeval &start, struct timeval &end) {
	recordTiming(bufferWriteTimeMap, dependency, start, end);
}

void CommStatistics::logStatistics(int indentation, std::ofstream &logFile) {
	std::ostringstream indent;
	for (int i = 0; i < indentation; i++) indent << '\t';
	for (int i = 0; i < commDependencyNames->NumElements(); i++) {
		const char *dependency = commDependencyNames->Nth(i);
		
		// setup times
		logFile << indent.str() << "Dependency: " << dependency << ":\n";
		logFile << indent.str() << '\t' << "Confinements processing time: ";
		logFile << *(confinementConstrTimeMap->Lookup(dependency)) << "\n";
		logFile << indent.str() << '\t' << "Buffer setup time: ";
		logFile << *(bufferSetupTimeMap->Lookup(dependency)) << "\n";
		logFile << indent.str() << '\t' << "Communication resources setup time: ";
		logFile << *(commResourcesSetupTimeMap->Lookup(dependency)) << "\n";

		// different parts of communication
		logFile << indent.str() << '\t' << "Communication time: \n";
		logFile << indent.str() << "\t\t" << "Buffer reading: ";
		double reading = *(bufferReadTimeMap->Lookup(dependency));
		logFile << reading << "\n";
		logFile << indent.str() << "\t\t" << "MPI transfer: ";
		double communication = *(communicationTimeMap->Lookup(dependency));
		logFile << communication << "\n";
		logFile << indent.str() << "\t\t" << "Buffer writing: ";
		double writing = *(bufferWriteTimeMap->Lookup(dependency));
		logFile << writing << "\n";
		logFile << indent.str() << "\t\t" << "Total: ";
		logFile << (reading + communication + writing) << "\n";
	}	
}

double CommStatistics::getTotalCommunicationTime() {
	
	double bufferReadTime = 0.0;
	double communicationTime = 0.0;
	double bufferWriteTime = 0.0;
	
	for (int i = 0; i < commDependencyNames->NumElements(); i++) {
		const char *dependency = commDependencyNames->Nth(i);
		bufferReadTime += *(bufferReadTimeMap->Lookup(dependency));
		communicationTime += *(communicationTimeMap->Lookup(dependency));
		bufferWriteTime += *(bufferWriteTimeMap->Lookup(dependency));
	}
	
	return bufferReadTime + communicationTime + bufferWriteTime;
}

void CommStatistics::recordTiming(Hashtable<double*> *map, const char *dependency, 
                struct timeval &start, 
		struct timeval &end) {

	pthread_mutex_lock(&mutex);
	double timeSpent = ((end.tv_sec + end.tv_usec / 1000000.0)
                        - (start.tv_sec + start.tv_usec / 1000000.0));
	double *timer = map->Lookup(dependency);
	Assert(time != NULL);
	*timer = *timer + timeSpent;
	pthread_mutex_unlock(&mutex);
}
