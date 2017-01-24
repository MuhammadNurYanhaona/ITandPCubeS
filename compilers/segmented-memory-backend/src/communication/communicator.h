/* In a segmented-memory system, it is typical to have multiple PPU controllers (we have implemented them with Pthreads) per 
 * segment. A typical example of this architecture is a set of multicore CPUs in a cluster. The data structure parts are managed
 * per segment basis. Consequently, communication also happens at the segment level. The execution of the task over the data
 * structures, however, takes place independently in each PPU controller within a segment. These controllers ask for data send
 * and receive independently likewise. Thus, there is a need for coordinating the sends and receives coming from individual
 * controllers before the segment can engage in the data transfer. The current implementation does a bulk-synchronous transfer
 * of data relevent to all PPUs. Once all PPUs that may participate in a send report to the communication handler, it fires the
 * send or skip sending based on what deemed appropriate. The same policy has been applied for data reception.
 *
 * This header file contains classes that are needed to implement pause/resume of PPU controllers for the sake of communication
 * control & coordination.  
 */

#ifndef _H_communicator
#define _H_communicator

#include "../utils/list.h"
#include "../runtime/parallel_comm_barrier.h"
#include "comm_buffer.h"
#include "comm_statistics.h"
#include "mpi_group.h"

#include <iostream>
#include <fstream>
#include <time.h>
#include <sys/time.h>

class Communicator;

/* Barrier class that is used to implement bulk synchronization during sending data
*/
class SendBarrier : public ParallelCommBarrier {
  protected:
	Communicator *communicator;
  public:
	SendBarrier(int participantCount, Communicator *communicator);

	// barrier interface functions
	bool shouldWait(SignalType signal, int callerIterationNo);
	bool shouldPerformTransfer(int activeSignalsCount, int callerIterationNo);
	void beforeTransfer(int order, int participants);
        void transferFunction();
        void afterTransfer(int order, int participants);
	void recordTimingLog(TimingLogType logType, struct timeval &start, struct timeval &end);

	// @depricated old send barrier function when communication activities were not divided into sequential and parallel parts	
	void executeSend();
};

/* Barrier class that is used to implement bulk synchronization during receiving data
*/
class ReceiveBarrier : public ParallelCommBarrier {
  protected:
	Communicator *communicator;
  public:
	ReceiveBarrier(int participantCount, Communicator *communicator);

	// barrier interface functions
        bool shouldWait(SignalType signal, int callerIterationNo);
	bool shouldPerformTransfer(int activeSignalsCount, int callerIterationNo);
        void beforeTransfer(int order, int participants);
        void transferFunction();
        void afterTransfer(int order, int participants);	
	void recordTimingLog(TimingLogType logType, struct timeval &start, struct timeval &end);

	// @depricated old recv barrier function when communication activities were not divided into sequential and parallel parts	
	void executeReceive();
};

/* This is the common super-class to implement bulk-synchronous communication for any synchronization type. The subclasses should
 * override relevent functions to modify the bulk-synchronous behavior, if intended. To support a lot of variations in subclass's 
 * behavior this class provides several plugin functions to be applied in the send and reception execution pipeline.
 */
class Communicator : public CommBufferManager {
  protected:
	// identifier for the current segment to be used for local and remote disambiguation
	int localSegmentTag;
	// keep track of the number of times this communicator has been used
	int iterationNo;
	// two barriers to pause/resume PPU controllers participating in communication
	SendBarrier *sendBarrier;
	ReceiveBarrier *receiveBarrier;
	// for a communicator within a group for MPI communications
	SegmentGroup *segmentGroup;
	// the list of segments interacting for this communicator; this is needed to set up the segment group
	std::vector<int> *participantSegments;
	// a stream for logging events on the communicator
	std::ofstream *logFile;
	// a communicator Id to be used as tag if needed
	int communicatorId;
	// a reference to the communication-statistics gatherer object to log time spent on this communicator
	CommStatistics *commStat;
  public:
	Communicator(int localSegmentTag, const char *dependencyName, int localSenderPpus, int localReceiverPpus);
	void setLogFile(std::ofstream *logFile) { this->logFile = logFile; }
	void setParticipants(std::vector<int> *participants) { this->participantSegments = participants; }
	void setCommStat(CommStatistics *commStat) { this->commStat = commStat; }
	CommStatistics *getCommStat() { return commStat; }
	virtual void describe(int indentation);

	// two functions to pre and post process communication buffers before a send and after a receive respectively these basically 
	// read and write the communication buffers
        void prepareBuffersForSend();
        void processBuffersAfterReceive();

	// alternative functions for buffer processing that can be used for parallel subclass implementations
	void prepareBuffersForSend(int currentPpuOrder, int participantsCount);
        void processBuffersAfterReceive(int currentPpuOrder, int participantsCount);

	// sets up the tags in each communication buffer that will be used during MPI communications in some communicator types; it 
	// also assign the communicator an ID which can also be used as a tag in some contexts
	void setupBufferTags(int communicatorId, int totalSegmentsInMachine);

	// this function should be overriden to setup any platform specific communication resource, if needed 
	virtual void setupCommunicator(bool includeNonInteractingSegments);

	// These are the implementation of send and receive functions from the communication-buffer manager class. The logic is
	// to just wait on the relevent barrier whose shouldWait() and releaseFunction() invoke other functions in this class to
	// apply synchronization type specific logic.
	void send(SignalType signal, int iteration) { sendBarrier->wait(signal, iteration); }
        void receive(SignalType signal, int iteration) { receiveBarrier->wait(signal, iteration); }

	// By default, any PPU requesting a send or reporting that it has nothing to send will wait on the barrier for all other
	// prospective senders.
	virtual bool shouldWaitOnSend(SignalType sendSignal, int iteration) { return true; }

	// By default, a PPU requesting a data reception only waits if the data has not been received yet for the iteration the
	// PPU is currently in.
	virtual bool shouldWaitOnReceive(SignalType receiveSignal, int iteration) { return true; }

	// By default, there must be at least one PPU that has reported that it has someting to send for the send barrier to execute
	// send in its release function
	virtual bool shouldSend(int sendRequestsCount) { return sendRequestsCount > 0; }
	
	// By default, any PPU waiting for data reception flags the need for issuing date receive on the communicator
	virtual bool shouldReceive(int receiveRequestsCount, int iteration) { 
		return (iteration == iterationNo && receiveRequestsCount > 0); 
	}

	//-------------------------------------------------------------------------- Parallel Preprocessing Functions for Transfer
	virtual void performSendPreprocessing(int currentPpuOrder, int participantsCount) {
		prepareBuffersForSend(currentPpuOrder, participantsCount);
	}
	virtual void perfromRecvPreprocessing(int currentPpuOrder, int participantsCount) {}	

	//----------------------------------------------------------------------------------------------------- Transfer Functions
	// The two transfer functions to be implemented by subclasses to do the actual platform specific data send and receive 
	// operations
	virtual void sendData() = 0;
	virtual void receiveData() = 0;

	//------------------------------------------------------------------------- Parallel Postprocessing Functions for Transfer 
	virtual void performSendPostprocessing(int currentPpuOrder, int participantsCount) {}
	virtual void perfromRecvPostprocessing(int currentPpuOrder, int participantsCount) {
		processBuffersAfterReceive(currentPpuOrder, participantsCount);
	}	
	
	//--------------------------------------------------------------------- After Transfer Communicator State Update Functions
	// There is nothing to do other than releasing the waiting PPUs after a send by default
	virtual void afterSend() {}	
	// Receive, on the other hand, increases the iteration number of the communicator to avoid PPUs reporting later on to get
	// halted on the receive-barrier.
	virtual void afterReceive() { iterationNo++; }

	// Some communication mechanisms, for example MPI, requires that even the non-participating segments should explicitly say
	// that they will not be communicating for proper communication resources (e.g., groups, channel) setup. Therefore, this
	// function is provided to exclude the current segment when other segments that will interact calls the setupCommunicator
	// function.
	static void excludeOwnselfFromCommunication(const char *dependencyName, 
		int localSegmentTag, std::ofstream &logFile);
};


#endif
