#ifndef _H_comm_stat
#define _H_comm_stat

#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>

#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/utils/hashtable.h"

// This class has been provided to track how much time different aspects of communications take at runtime. IT 
// communicator setup includes a lot of data processing and computations. Furthermore, the actual transfer of 
// data can be more or less costly depending on the data volume and the communication mechanism. All these costs
// needs to be analyzed to determine what aspects can be optimized or overhauled for better performance. This
// class gathers all the statistics for such analyses.    
class CommStatistics {
  protected:
	// maps for gathering different types of timing statistic
	List<const char*> *commDependencyNames;
	Hashtable<double*> *confinementConstrTimeMap;
	Hashtable<double*> *bufferSetupTimeMap;
	Hashtable<double*> *commResourcesSetupTimeMap;
	Hashtable<double*> *bufferReadTimeMap;
	Hashtable<double*> *communicationTimeMap;
	Hashtable<double*> *bufferWriteTimeMap;

	// a mutex to protect the stat object from being corrupted if multiple threads try to enter timing data 
	//into it at the same time
	pthread_mutex_t mutex;
  public:
	CommStatistics();
	~CommStatistics();
	
	// function to initiate entries in different maps for a particular communication dependency
	void enlistDependency(const char *dependency);
	
	// functions for recording time spent on different aspects of a specific communication dependency
	void addConfinementConstrTime(const char *dependency, struct timeval &start, struct timeval &end);	
	void addBufferSetupTime(const char *dependency, struct timeval &start, struct timeval &end);	
	void addCommResourcesSetupTime(const char *dependency, struct timeval &start, struct timeval &end);	
	void addBufferReadTime(const char *dependency, struct timeval &start, struct timeval &end);
	void addCommunicationTime(const char *dependency, struct timeval &start, struct timeval &end);
	void addBufferWriteTime(const char *dependency, struct timeval &start, struct timeval &end);
	
	// function to be used at program's end to log the total time spent on different communication dependencies
	void logStatistics(int indentation, std::ofstream &logFile);

	// function to find the overall time the task spent on communication 
	double getTotalCommunicationTime();
  private:
	void recordTiming(Hashtable<double*> *map, const char *dependency, 
			struct timeval &start, struct timeval &end);	
};

#endif
