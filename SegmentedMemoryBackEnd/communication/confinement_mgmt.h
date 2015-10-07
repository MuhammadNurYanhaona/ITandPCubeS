#ifndef CONFINEMENT_MGMT_H_
#define CONFINEMENT_MGMT_H_

/* The idea of a confinement of communication is that it represents a location in the parts distribution
 * hierarchy of the underlying data structure such that the parts underneath the confinement should be
 * synchronized with one another depending on their data sharing characteristics. Individual parts may be
 * residing in different segments or within a single segments. Their physical location will effect how
 * data should be moved from one part to another but not the content that should be moved.
 *
 * Note that for a single data synchronization need there might be one or more confinements per segment.
 * All of these confinements will be in the same LPS level that is determined by the nature of the data
 * dependency arc originating the synchronization need. Therefore, the broad picture is there will be a
 * confinement level per communication with one or more independent confinement roots.
 *
 * Note the the notion of confinement become applicable only after we validated that there is a need of
 * data movement due to some shared/overlapped data structure update. This is important to remember as
 * when multiple PPUs update regions of a single part then barrier or other forms of local sync primitives
 * are enough to synchronize all PPUs before they proceed further.
 * */

#include "../utils/list.h"
#include "../utils/interval.h"
#include "../partition-lib/partition.h"
#include "part_distribution.h"
#include "part_folding.h"
#include "part_config.h"
#include "../memory-management/part_tracking.h"
#include <iostream>
#include <vector>

class ExchangeIterator;

// constants to determine what role a particular branch in a confinement should play in the communication
enum CommRole {SEND, RECEIVE, SEND_OR_RECEIVE};

// To create the confinement locations for a communication we need a lot of information regarding the to
// be synchronized data structure; this class holds all useful information and supporting functions for
// confinement construction.
class ConfinementConstructionConfig {
  private:
	// sender and receiver side data partition configuration are needed to traverse part-container and part
	// distribution trees to identify and fold branches to generate the participants in a confined sync
	int senderLps;
	DataItemConfig *senderConfig;
	int receiverLps;
	DataItemConfig *receiverConfig;

	// an identifier for the local segment is needed to distinguish between local and remote data exchanges
	int localSegmentTag;

	// as the name suggests, this represents the confinement level of communication
	int confinementLps;

	// the part tracking trees of the data structure for both sides are needed to determine the confinement
	// roots; any of the two trees can be NULL; if both are NULL then the current segment has no part in the
	// communication
	PartIdContainer *senderPartTree;
	PartIdContainer *receiverPartTree;

	// the part distribution tree that coalesce all independent partitions of a data structure and contains
	// information regarding what part residing in what segment is needed to find all participant branches
	// within a confinement root after the root has been identified from the previous trees
	BranchingContainer *partDistributionTree;
  public:
	ConfinementConstructionConfig(int lST,
			int sLps, DataItemConfig *sCon,
			int rLps, DataItemConfig *rCon,
			int cLps,
			PartIdContainer *sPTree, PartIdContainer *rPTree,
			BranchingContainer *pDTree);

	int getLocalSegmentTag() { return localSegmentTag; }
	int getSenderLps() { return senderLps; }
	DataItemConfig *getSenderConfig() { return senderConfig; }
	PartIdContainer *getSenderPartTree() { return senderPartTree; }
	int getReceiverLps() { return receiverLps; }
	DataItemConfig *getReceiverConfig() { return receiverConfig; }
	PartIdContainer *getReceiverPartTree() { return receiverPartTree; }
	int getDataDimensions() { return senderConfig->getDimensionality(); }
	bool isRootConfinement(int rootLps) { return confinementLps == rootLps; }
	BranchingContainer *getDistributionTree() { return partDistributionTree; }

	// path configuration for traversal of the part distribution tree from a confinement root
	std::vector<LpsDimConfig> *getSendPathInDistributionTree();
	std::vector<LpsDimConfig> *getReceivePathInDistributionTree();

	// tells if updater and receiver data parts are from the same data distribution; this should be true for
	// ghost-boundary-region synchronization; do not confuse it with intra-segment synchronization
	bool isIntraContrainerSync() { return senderLps == receiverLps; }

	// determines the part container levels of the confinement root for the part tracking trees; note that
	// the confinement level should be the same in both trees
	int getConfinementLevel(std::vector<LpsDimConfig> *configVector = NULL);

	// generates all confinement root IDs by searching either the sender or the receiver tree
	List<List<int*>*> *getConfinementIds(int confinementLevel);

	// Paddings in the partition configuration of participating data parts need to be enabled/disabled based
	// the sender and receiver LPS levels and the nature of the synchronization before the data configuration
	// objects can be used to generate interval sequence descriptions for data parts. This function does the
	// configuration update.
	// Note that this function assume that the sender and receiver configuration are distinct objects even if
	// the senderLPS and receiverLPS are the same. So the assumption is that new instances of configurations
	// will be created for each communication scenarios. If that assumption does not hold then in the future
	// then reconfigure sender and receiver paddings before generating interval descriptions for branches of a
	// confinement based on demand of the situation.
	void configurePaddingInPartitionConfigs();

	// two auxiliary functions to be used to construct sender and receiver LPS dimension order paths
	std::vector<LpsDimConfig> *forwardTruncatedDimVector(int truncateLevel,
			std::vector<LpsDimConfig> *originalVector);
	std::vector<LpsDimConfig> *backwardTruncatedDimVector(int truncateLevel,
				std::vector<LpsDimConfig> *originalVector);
  private:
	// an auxiliary function to be used for configuring padding in sender and receiver data configurations
	void setPaddingThresholdInDataConfig(DataItemConfig *config, int thresholdLps);
};

// A participants is an entry within a confinement that will either send/receive or both data points from the
// part (or parts if it is accumulating data from lower LPS data parts) it represents.
class Participant {
  private:
	CommRole role;
	std::vector<int*> *containerId;
	List<MultidimensionalIntervalSeq*> *dataDescription;

	// because of data replication the same confinement participant may be shared in multiple segments; so we
	// maintain a list of segment tags to allow optimization through group communications in environment that
	// support that
	std::vector<int> segmentTags;

	// this ID is supposed to be the list index of this participant in sender/receiver list of a confinement
	// it has been added for fast retrieval by list index access -- the true identifier of a participant is its
	// container ID
	int id;
  public:
	Participant(CommRole r, std::vector<int*> *c, List<MultidimensionalIntervalSeq*> *d);
	void addSegmentTag(int segmentTag);
	std::vector<int> getSegmentTags() { return segmentTags; }
	bool hasSegmentTag(int segmentTag);
	List<MultidimensionalIntervalSeq*> *getDataDescription() { return dataDescription; }
	void setId(int id) { this->id = id; }
	int getId() { return id; }
	void setRole(CommRole role) { this->role = role; }
	std::vector<int*> *getContainerId() { return containerId; }

	// two participants are equal if their data descriptions are equal
	bool isEqual(Participant *other);
};

/* this class represents the data that should be communicated between a sender-receiver pair of a communication
 * */
class DataExchange {
  private:
	Participant *sender;
	Participant *receiver;
	List<MultidimensionalIntervalSeq*> *exchangeDesc;

	// A full overlap flag is used when the sender and receiver have exactly the same data configuration. In
	// that case there is no need to calculate the exchange description as an intersection of their contents and
	// there might be some opportunity of optimizing the implementation of data exchange later.
	bool fullOverlap;
  public:
	// constructor to be used in the full overlap situation
	DataExchange(Participant *sender, Participant *receiver);
	// constructor to be used in other situation
	DataExchange(Participant *sender,
			Participant *receiver,
			List<MultidimensionalIntervalSeq*> *exchangeDesc);

	Participant *getSender() { return sender; }
	Participant *getReceiver() {return receiver; }
	bool isFullOverlap() { return fullOverlap; }
	List<MultidimensionalIntervalSeq*> *getExchangeDesc() { return exchangeDesc; }
	void describe(int indentLevel, std::ostream &stream);

	// tells the total number of data points that need to be exchanged between the sender and receiver as part of
	// this data exchange
	int getTotalElementsCount();

	// This static function is to be used to determine if a candidate pair of sender-receiver participants of a
	// confinement should indeed exchange data. If the outcome of this function is NULL then the two parties do
	// not interact. Otherwise, a data exchange instance should be created to record information about the pair.
	static List<MultidimensionalIntervalSeq*> *getCommonRegion(Participant *sender, Participant *receiver);

	// This function is used to order data-exchanges participating in a single synchronization; the second argu-
	// ment indicates if the sender or receiver side of the data exchanges should be compared for ordering. It
	// returns -1 if the current should appear first, 1 if rather the 'other' should take that position, and 0 if 
	// their ordering does not matter.  
	int compareTo(DataExchange *other, bool forReceive);
  private:
	void drawDataDescription(List<MultidimensionalIntervalSeq*> *seqList);
};

/* For a particular data synchronization scenario there might be a need to move data between different segments,
 * between data parts of different part-container trees within a single segment, or between overlapped data parts
 * within a single part container tree spanning a single or multiple segments. If we keep the case of data
 * movements across segments and across part-container trees separate from within-container data movements, there
 * will be less computation overhead in the calculation of the data exchange configuration in most scenarios. This
 * class handles the first two cases.
 * */
class CrossSegmentInteractionSpec {
  private:
	int localSegmentTag;
	Participant *localSender;
	Participant *localReceiver;
	List<Participant*> *remoteSenders;
	List<Participant*> *remoteReceivers;

	// local interaction is allowed when the local sender and receiver are from different LPS hierarchies; the
	// same hierarchy case is tackled by the next class
	bool localInteractionAllowed;
  public:
	CrossSegmentInteractionSpec(Container *confinementContainer, ConfinementConstructionConfig *config);

	List<DataExchange*> *generateSendExchanges();
	List<DataExchange*> *generateReceiveExchanges();

	Participant *getRemoteSender(int listIndex) { return remoteSenders->Nth(listIndex); }
	Participant *getRemoteReceiver(int listIndex) { return remoteReceivers->Nth(listIndex); }
	void describe(int indentLevel, std::ostream &stream);
  private:
	// an utility function to initialize the participant lists
	void generateParticipantList(Container *confinementContainer,
			int dataDimensions,
			CommRole role,
			std::vector<LpsDimConfig> pathOrder,
			DataItemConfig *dataConfig,
			int pruningLevel);
};

/* This class handles the case of within-container parts synchronization
 * */
class IntraContainerInteractionSpec {
  private:
	int localSegmentTag;
	int dataDimensions;

	// even for within-container data synchronization, we maintain separate sender and receiver lists despite the
	// same set of part-tracking containers being processed to create the two lists because the data description for
	// sending and receiving might be different, e.g., padding may be enabled for the sender but not for the
	// receiver side during interval description generation process
	List<Participant*> *senderList;
	List<Participant*> *receiverList;
  public:
	IntraContainerInteractionSpec(BranchingContainer *confinementContainer, ConfinementConstructionConfig *config);

	// functions to use to get sender-receiver pair of a data exchange
	Participant *getSender(int listIndex) { return senderList->Nth(listIndex); }
	Participant *getReceiver(int listIndex) { return receiverList->Nth(listIndex); }

	// determines if their is indeed any need for synchronization within the container
	bool isSynchronizationNeeded() { return senderList->NumElements() > 1; }

	List<DataExchange*> *generateExchanges();
	void describe(int indentLevel, std::ostream &stream);
  private:
	// a helper function to be used for generating the sender and receiver list of the confinement
	List<Participant*> *generateParticipantList(List<Container*> *participantBranches,
			CommRole role,
			std::vector<LpsDimConfig> pathOrder,
			DataItemConfig *dataConfig, int pruningLevel);
};

/* This class represents a confinement root of a communication.
 * Note that when confinement roots are generated by searching the sender and/or receiver part container tree(s)
 * then the list of confinements a particular segment considers interesting varies from segment to segment. The
 * participants of a confinement, however, are the same regardless of what segment among the list of segments
 * commonly holding the confinement investigates it. This is because the participants are derived from the part
 * distribution tree that contains information about all data parts and all segments.
 * */
class Confinement {
  private:
	int dataDimensions;
	std::vector<int> participantSegments;
	CrossSegmentInteractionSpec *remoteInteractions;
	IntraContainerInteractionSpec *localInteractions;

	// this is the attribute that uniquely identifies the confinement in the part distribution tree
	BranchingContainer *confinementContainer;
  public:
	Confinement(int dd, BranchingContainer *cC, ConfinementConstructionConfig *config);
	BranchingContainer *getContainer() { return confinementContainer; }

	CrossSegmentInteractionSpec *getRemoteInteractions() { return remoteInteractions; }
	IntraContainerInteractionSpec *getLocalInteractions() { return localInteractions; }

	// this static utility method should be used to find and set up all confinement roots for a communication
	static List<Confinement*> *generateAllConfinements(ConfinementConstructionConfig *config, int rootLps);

	// an utility method to be used by cross-segment and within-container interaction specification classes
	static List<DataExchange*> *generateDataExchangeList(List<Participant*> *senderList,
			List<Participant*> *receiverList);
};

/* An iterator to traverse through the data points of a data exchange
 * */
class ExchangeIterator {
  private:
	int totalElementsCount;
	int currentElement;
	List<MultidimensionalIntervalSeq*> *sequences;
	int currentSequence;
	SequenceIterator *iterator;
  public:
	ExchangeIterator(DataExchange *exchange);
	~ExchangeIterator();
	bool hasMoreElements() { return currentElement < totalElementsCount; }

	// returns the next index point in the data exchange; returns NULL if it reaches the end of the exchange
	std::vector<int> *getNextElement();

	void printNextElement(std::ostream &stream);
};

#endif /* CONFINEMENT_MGMT_H_ */
