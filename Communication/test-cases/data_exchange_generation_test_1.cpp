/* The goal of this test is to verify if data-exchanges for a communication are generated correctly by the relevant
 * libraries. The scenario we use here is the exchange between a matrix-matrix-multiply and LU-factorization tasks.
 * The first uses a 2D partitioning of a matrix that does not reorder the indices while the second uses a 1D stride
 * reordering partition. This kind of interactions may be needed in an efficient linear algebra solver program.
 * */

#include "../utils/list.h"
#include "../utils/id_generation.h"
#include "../utils/interval.h"
#include "../utils/partition.h"
#include "../part-management/part_config.h"
#include "../part-management/part_tracking.h"
#include "../communication/part_distribution.h"
#include "../communication/confinement_mgmt.h"
#include "../communication/data_transfer.h"

#include <iostream>
#include <vector>
#include <cstdlib>

using namespace std;

int mainDEGT1() {

	// matrix dimensions
	Dimension rows = Dimension(100);
	Dimension cols = Dimension(100);

	// data partition configuration for the matrix-matrix-multiply task
	DataItemConfig *mmmConfig = new DataItemConfig(2, 2);
	mmmConfig->setDimension(0, rows);
	mmmConfig->setDimension(1, cols);
	mmmConfig->setPartitionInstr(0, 1, new BlockSizeInstr(25)); // partition the columns in the upper level
	mmmConfig->getInstruction(0, 1)->setPriorityOrder(0);
	mmmConfig->setPartitionInstr(0, 0, new VoidInstr()); 		// replicate the rows (2D upper level LPS)
	mmmConfig->setLpsIdOfLevel(0, 1);
	mmmConfig->getInstruction(0, 0)->setPriorityOrder(1);
	mmmConfig->setPartitionInstr(1, 0, new BlockSizeInstr(25)); // partition the rows in the lower level
	mmmConfig->getInstruction(1, 0)->setPriorityOrder(0);
	mmmConfig->setPartitionInstr(1, 1, new VoidInstr()); 		// left alone the columns (1D lower level LPS)
	mmmConfig->getInstruction(1, 1)->setPriorityOrder(1);
	mmmConfig->setLpsIdOfLevel(1, 2);
	mmmConfig->updateParentLinksOnPartitionConfigs();

	// data partition configuration for the LU-factorization task
	DataItemConfig *lufConfig = new DataItemConfig(2, 1);
	lufConfig->setDimension(0, rows);
	lufConfig->setDimension(1, cols);
	lufConfig->setPartitionInstr(0, 0, new StrideInstr(8));		// distribute strides of rows to 8 PPUs
	lufConfig->getInstruction(0, 0)->setPriorityOrder(0);
	lufConfig->setPartitionInstr(0, 1, new VoidInstr()); 		// left alone the columns
	lufConfig->getInstruction(0, 0)->setPriorityOrder(1);
	lufConfig->setLpsIdOfLevel(0, 3);
	lufConfig->updateParentLinksOnPartitionConfigs();

	// Assume there are 4 segments and 2 PPUs per segments at a lower PPS level. Further assume that the upper
	// LPS of MMM has been mapped to the segment level and the sole LPS of LUF has been mapped to the nested
	// PPS level within a segment. With that configuration each segment will get two upper level LPUs of MMM and
	// two LPUs of LUF.

	// Assume the current segment is the Segment #0 and populate its part container trees
	List<int*> *partId = NULL;
	vector<DimConfig> *dimOrderMMM = mmmConfig->generateDimOrderVectorWithoutLps();
	PartIdContainer *mmmHierarchy = new PartListContainer(dimOrderMMM->at(0));
	partId = idutils::generateIdFromArray(new int[4] {0, 0, 0, 0}, 2, 4);
	mmmHierarchy->insertPartId(partId, 2, *dimOrderMMM);
	partId = idutils::generateIdFromArray(new int[4] {0, 0, 1, 0}, 2, 4);
	mmmHierarchy->insertPartId(partId, 2, *dimOrderMMM);
	partId = idutils::generateIdFromArray(new int[4] {0, 0, 2, 0}, 2, 4);
	mmmHierarchy->insertPartId(partId, 2, *dimOrderMMM);
	partId = idutils::generateIdFromArray(new int[4] {0, 0, 3, 0}, 2, 4);
	mmmHierarchy->insertPartId(partId, 2, *dimOrderMMM);
	partId = idutils::generateIdFromArray(new int[4] {0, 1, 0, 0}, 2, 4);
	mmmHierarchy->insertPartId(partId, 2, *dimOrderMMM);
	partId = idutils::generateIdFromArray(new int[4] {0, 1, 1, 0}, 2, 4);
	mmmHierarchy->insertPartId(partId, 2, *dimOrderMMM);
	partId = idutils::generateIdFromArray(new int[4] {0, 1, 2, 0}, 2, 4);
	mmmHierarchy->insertPartId(partId, 2, *dimOrderMMM);
	partId = idutils::generateIdFromArray(new int[4] {0, 1, 3, 0}, 2, 4);
	mmmHierarchy->insertPartId(partId, 2, *dimOrderMMM);

	vector<DimConfig> *dimOrderLUF = lufConfig->generateDimOrderVectorWithoutLps();
	PartIdContainer *lufHierarchy = new PartListContainer(dimOrderLUF->at(0));
	partId = idutils::generateIdFromArray(new int[2] {0, 0}, 2, 2);
	lufHierarchy->insertPartId(partId, 2, *dimOrderLUF);
	partId = idutils::generateIdFromArray(new int[2] {1, 0}, 2, 2);
	lufHierarchy->insertPartId(partId, 2, *dimOrderLUF);

	// Create a part-distribution-tree and put the parts for MMM and LUF of current segment there
	BranchingContainer *distributionTree = new BranchingContainer(0, LpsDimConfig());
	vector<LpsDimConfig> *mmmLpsDimOrder = mmmConfig->generateDimOrderVector();
	PartIterator *mmmIterator = mmmHierarchy->getIterator();
	SuperPart *part = NULL;
	while ((part = mmmIterator->getCurrentPart()) != NULL) {
		distributionTree->insertPart(*mmmLpsDimOrder, 0, part->getPartId());
		mmmIterator->advance();
	}
	vector<LpsDimConfig> *lufLpsDimOrder = lufConfig->generateDimOrderVector();
	PartIterator *lufIterator = lufHierarchy->getIterator();
	while ((part = lufIterator->getCurrentPart()) != NULL) {
		distributionTree->insertPart(*lufLpsDimOrder, 0, part->getPartId());
		lufIterator->advance();
	}

	// Assuming we are concerned about a scenario where data is supposed to move from MMM to LUF task, enter the
	// data parts of LPUs of the second segment
	partId = idutils::generateIdFromArray(new int[2] {2, 0}, 2, 2);
	distributionTree->insertPart(*lufLpsDimOrder, 1, partId);
	partId = idutils::generateIdFromArray(new int[2] {3, 0}, 2, 2);
	distributionTree->insertPart(*lufLpsDimOrder, 1, partId);

//	distributionTree->print(0, cout);

	// Create a confinement construction configuration for the assumed communication scenario
	ConfinementConstructionConfig *ccConfig = new ConfinementConstructionConfig(
			0, 					// local segment tag
			2, 					// sending side's LPS level (the last level of it's part container tree)
			mmmConfig, 			// partition configuration for the sending side
			3, 					// receiving side's LPS level
			lufConfig,			// partition configuration for the receiving side
			0,					// confinement level LPS (in this case the Root LPS)
			mmmHierarchy, 		// sender side's part-container hierarchy
			lufHierarchy,		// receiver side's part-container hierarchy
			distributionTree);	// part-distribution tree

	// Generate confinements
	List<Confinement*> *confinementList = Confinement::generateAllConfinements(ccConfig, 0);

	// Investigate the confinements
	cout << "Number of Confinements: " << confinementList->NumElements() << "\n";
	for (int i = 0; i < confinementList->NumElements(); i++) {
		cout << "Confinement No: " << i << "\n";
		Confinement *confinement = confinementList->Nth(i);
		CrossSegmentInteractionSpec *remoteInteractions = confinement->getRemoteInteractions();
		if (remoteInteractions != NULL) {
			cout << "This confinement involves remote and/or cross-containers interactions\n";
			remoteInteractions->describe(1, cout);
			List<DataExchange*> *sendExchanges = remoteInteractions->generateSendExchanges();
			cout << "Send Exchange Count: " << sendExchanges->NumElements() << "\n";
			for (int j = 0; j < sendExchanges->NumElements(); j++) {
				cout << "\nExchange #" << j << "\n";
				sendExchanges->Nth(j)->describe(0, cout);
			}
			List<DataExchange*> *receiveExchanges = remoteInteractions->generateReceiveExchanges();
			cout << "\nReceive Exchange Count: " << receiveExchanges->NumElements() << "\n";
			for (int j = 0; j < receiveExchanges->NumElements(); j++) {
				cout << "\nExchange #" << j << "\n";
				receiveExchanges->Nth(j)->describe(0, cout);
			}
		}
		IntraContainerInteractionSpec *localInteractions = confinement->getLocalInteractions();
		if (localInteractions == NULL) {
			cout << "\nThis confinement involves no intra-container parts interactions\n";
		}
	}

	return 0;
}


