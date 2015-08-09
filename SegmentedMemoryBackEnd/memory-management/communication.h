#ifndef _H_communication
#define _H_communication

/* This header file provides the library specification related to managing communication buffers. Actual
   implementation of communication will be dealt elsewhere and kept separate to support different 
   communication technologies. Management of communication buffers involves their allocation, update 
   tracking, and copying data to (and from) operating memory (the memory accessed and modified as a PPU
   executes its LPUs) out of (and into) those buffers.  
*/

#include "../utils/list.h"
#include "../utils/interval_utils.h"
#include "../utils/hashtable.h"
#include "../codegen/structure.h"
#include "allocation.h"

/* A class for grouping the set of PPUs that will send data to or receive the same data parts from the 
   current PPU as part of some synchonization of a data structure. Note that the group may contain the
   current PPU if the synchronization happens across LPSes.
*/
class CommGroup {
  protected:
	// id of the LPS that will be considered when finding the interacting PPUs (remember that we may
	// have a single processing unit -- a thread or a process -- working on multiple PPSes and hence
        // possessing multiple PPU ids)
	int lpsId;
	// PPU Ids of the processing units that will interact with the current one
	List<int> *ppuIds;
  public:
	CommGroup(int lpsId, List<int> *ppuIds) {
		this->lpsId = lpsId;
		this->ppuIds = ppuIds;
	}
	inline int getLpsId() { return lpsId; }
	inline List<int> *getPpuIds() { return ppuIds; }
	// combines the set of ppus found in current group with the argument group and returns the union
	List<int> *combineGroups(List<int> *others);
};

/* A class for holding the configuration and data for a piece of communication buffer that a particular 
   group of PPUs will exchange with the current PPU.   
*/
class CommBuffer {
  protected:
	// the group of PPUs that will participate in the communication to update the buffer or to receive 
	// any update done on the buffer by current PPU
	CommGroup *group;
	// the interval description for the data been exchanged
	IntervalSet *bufferConfig;
	// memory to hold the buffer
	void *data;
	// length of the buffer to know how much data needs to be exchanged
	int length;
	// a variable to track the portion of the buffer updated by any ongoing communication; this will
	// be useful when we will support parts of the buffer to come from different PPUs in the group
	int uptodateElements;
	// This indicates the number of distinct intervals that are parts of this buffer. For partial com-
	// munications probably the most reasonable thing to do is to allow each sequence to be updated by
	// one PPU.
	int intervalSeqCount;
	// meaning of this variable will be clear from the 'setBuffer' method
	bool needDataCopy;
  public:
	CommBuffer(CommGroup *group, IntervalSet *bufferConfig);
	// makes the buffer ready for reuse
	inline void resetState() { uptodateElements = 0; }
	// check if the buffer is fully updated with new data
	inline bool isUptodate() { return intervalSeqCount == uptodateElements; }
	template <class Type> void allocate() { data = new Type[length]; }
	inline void *getData() { return data; }
	inline int getLength() { return length; }
	inline int getPartCount() { return intervalSeqCount; }
	inline CommGroup *getGroup() { return group; }
	inline IntervalSet *getBufferConfig() { return bufferConfig; }
	// a simplified interface for updating the whole buffer all at once; note that in such cases the
	// data buffer can be used directly to receive/send uptodate content
	void update() { uptodateElements = intervalSeqCount; }
	// interface for updating only a part (i.e., one interval sequence) of the buffer; the second
	// parameter indicates the index ranges in the buffer the updated part resides
	template <class Type> void updatePart(Type *part, Range updateRange) {
		Type *buffer = (Type *) data;
		for (int i = updateRange.min; i <= updateRange.max; i++) {
			buffer[i] = part[i];
		}
		uptodateElements++;
	}
	// Sometimes the operating memory of a data part can be used directly for communication. When 
	// doing that is feasible, we can avoid copying data to and from an extra buffer. This method is
	// added to support such cases. Here the caller will directly pass the buffer for data exchange
	inline void setBuffer(void *buffer) { data = buffer; needDataCopy = false; }
	inline bool doesNeedDataCopy() { return needDataCopy; }
};

/* A class for holding all buffers related to a single data synchronization operation in an IT task. Once
   all of its buffers are updated, communication for the synchronization is done and computation -- if there 
   are no more synchronization to be resolved -- can resume again. Copying data from communication buffers
   to the operating memory can be done when individual buffers get updated or all at once after the entire
   list is updated. The design is flexible. Whatever the policy for copying may be in use, after syncing is 
   done the list need to be reset to make it usable for the next iteration.	   
*/
class CommBufferList {
  protected:
	int bufferCount;
	List<CommBuffer*> *buffers;
	int version;
	int updatedBuffers;
  public:
	CommBufferList(List<CommBuffer*> *buffers);
	inline bool isUptodate() { return updatedBuffers == bufferCount; }
	void reset();
	inline List<CommBuffer*> *getBuffers() { return buffers; }
	// the argument buffer must be a part of current list make a sense of this function; this function 
	// has been provided so that we can determine when the synchronization is over and we can reset 
	// the list
	inline void noteDataReception(CommBuffer *buffer) { 
		if (buffer->isUptodate()) updatedBuffers++; 
	}
	// This is a utility function to generate the communication configuration for a particular data
	// synchronization based on overlapping of data parts among different PPUs. The first argument 
	// lists all potential PPUs that may interact with the current PPU for this synchronization. The
	// second argument provides the interval description for data of each of those PPUs. There is an
	// one-to-one correspondance between i'th entry of the first and second argument. The the third
	// argument is for the interval description for current PPU to compute intersection. The final
	// argument refers to the LPS of the interacting PPUs. If there is no need of communication, in 
	// the rare occasion of no overlappings, then the function should return NULL.
	static CommBufferList *generateList(List<int> *ppuIds, 
			List<IntervalSet*> *intervalConfigs, 
			IntervalSet *currentConfig, int lpsId);
  private:	 
	// a recursive utility function to aid in the computation of 'generateList' function of the above
	// it uses a dynamic computing algorithm to achieve its feat. The updatedBuffers keeps track of 
	// the communication buffers that been generated as each step of the recursiion. The second argument
	// holds a list of original groups formed by calculating pairwise intersection of PPUs with current
	// PPU. This list never changes but entries from it are checked against entries of updatedBuffers
	// and get broken down and added into the latter. The probeIndex determines the next location of
	// recursion. 
	static List<CommBuffer*> *generateCommBuffers(List<CommBuffer*> *updatedBuffers, 
			List<CommBuffer*> *overlappingPairs, int probeIndex);
};

#endif
