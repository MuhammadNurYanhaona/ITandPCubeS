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
        communicationTimeMap = new Hashtable<double*>;
	pthread_mutex_init(&mutex, NULL);
}

CommStatistics::~CommStatistics() {
	delete commDependencyNames;
        delete confinementConstrTimeMap;
        delete bufferSetupTimeMap;
        delete commResourcesSetupTimeMap;
        delete communicationTimeMap;
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
	double *cTime = new double;
	*cTime = 0.0;
        communicationTimeMap->Enter(dependency, cTime);

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
        
void CommStatistics::addCommunicationTime(const char *dependency, struct timeval &start, struct timeval &end) {
	recordTiming(communicationTimeMap, dependency, start, end);
}

void CommStatistics::logStatistics(int indentation, std::ofstream &logFile) {
	std::ostringstream indent;
	for (int i = 0; i < indentation; i++) indent << '\t';
	for (int i = 0; i < commDependencyNames->NumElements(); i++) {
		const char *dependency = commDependencyNames->Nth(i);
		logFile << indent.str() << "Dependency: " << dependency << ":\n";
		logFile << indent.str() << '\t' << "Confinements processing time: ";
		logFile << *(confinementConstrTimeMap->Lookup(dependency)) << "\n";
		logFile << indent.str() << '\t' << "Buffer setup time: ";
		logFile << *(bufferSetupTimeMap->Lookup(dependency)) << "\n";
		logFile << indent.str() << '\t' << "Communication resources setup time: ";
		logFile << *(commResourcesSetupTimeMap->Lookup(dependency)) << "\n";
		logFile << indent.str() << '\t' << "Communication time: ";
		logFile << *(communicationTimeMap->Lookup(dependency)) << "\n";
	}	
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
