/* This data exchange generation test consider the scenario of a 5-point stencil computation that has two levels of
 * ghost-region padding
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

int mainDEGT2() {

	// grid dimensions
	Dimension rows = Dimension(100);
	Dimension cols = Dimension(100);

	// data partition configuration
	DataItemConfig *stencilConfigSender = new DataItemConfig(2, 2);
	stencilConfigSender->setDimension(0, rows);
	stencilConfigSender->setDimension(1, cols);
	BlockSizeInstr* rowPartitionInstr = new BlockSizeInstr(20);
	stencilConfigSender->setPartitionInstr(0, 0, rowPartitionInstr); // partition the rows in the upper level LPS
	rowPartitionInstr->setPadding(5, 5);					   // add 5 overlapping ghost rows on top and bottom
	rowPartitionInstr->setPriorityOrder(0);
	stencilConfigSender->setPartitionInstr(0, 1, new VoidInstr());
	stencilConfigSender->getInstruction(0, 1)->setPriorityOrder(1);
	stencilConfigSender->setLpsIdOfLevel(0, 1);
	BlockSizeInstr* colPartitionInstr = new BlockSizeInstr(20);
	stencilConfigSender->setPartitionInstr(1, 1, colPartitionInstr); // partition the columns in the lower level LPS
	colPartitionInstr->setPadding(2, 2);					   // add a 2 overlapping ghost columns on left and right
	colPartitionInstr->setPriorityOrder(0);
	stencilConfigSender->setPartitionInstr(1, 0, new VoidInstr());
	stencilConfigSender->getInstruction(1, 0)->setPriorityOrder(1);
	stencilConfigSender->setLpsIdOfLevel(1, 2);
	stencilConfigSender->updateParentLinksOnPartitionConfigs();

	// duplicate of the above configuration for the receiver side (in the compiler both configuration will be
	// generated automatically from other data structures)
	DataItemConfig *stencilConfigReceiver = new DataItemConfig(2, 2);
	stencilConfigReceiver->setDimension(0, rows);
	stencilConfigReceiver->setDimension(1, cols);
	BlockSizeInstr* rowPartitionInstr1 = new BlockSizeInstr(20);
	stencilConfigReceiver->setPartitionInstr(0, 0, rowPartitionInstr1);
	rowPartitionInstr1->setPadding(5, 5);
	rowPartitionInstr1->setPriorityOrder(0);
	stencilConfigReceiver->setPartitionInstr(0, 1, new VoidInstr());
	stencilConfigReceiver->getInstruction(0, 1)->setPriorityOrder(1);
	stencilConfigReceiver->setLpsIdOfLevel(0, 1);
	BlockSizeInstr* colPartitionInstr1 = new BlockSizeInstr(20);
	stencilConfigReceiver->setPartitionInstr(1, 1, colPartitionInstr1);
	colPartitionInstr1->setPadding(2, 2);
	colPartitionInstr1->setPriorityOrder(0);
	stencilConfigReceiver->setPartitionInstr(1, 0, new VoidInstr());
	stencilConfigReceiver->getInstruction(1, 0)->setPriorityOrder(1);
	stencilConfigReceiver->setLpsIdOfLevel(1, 2);
	stencilConfigReceiver->updateParentLinksOnPartitionConfigs();

	// assume the parts have been distributed to segments as such that the first segment has all the
	// parts with row index 0 (there are 5 such parts) and the first part with row index 1, and the second
	// segment has the remaining 4 parts for row index 1; finally, let the third segment has all five parts
	// with the row index 2

	// insert the parts of Segment #1 in the part-container hierarchy
	List<int*> *partId = NULL;
	vector<DimConfig> *dimOrderStencil = stencilConfigReceiver->generateDimOrderVectorWithoutLps();
	PartIdContainer *stencilHierarchy = new PartListContainer(dimOrderStencil->at(0));
	partId = idutils::generateIdFromArray(new int[4] {0, 0, 0, 0}, 2, 4);
	stencilHierarchy->insertPartId(partId, 2, *dimOrderStencil);
	partId = idutils::generateIdFromArray(new int[4] {0, 0, 0, 1}, 2, 4);
	stencilHierarchy->insertPartId(partId, 2, *dimOrderStencil);
	partId = idutils::generateIdFromArray(new int[4] {0, 0, 0, 2}, 2, 4);
	stencilHierarchy->insertPartId(partId, 2, *dimOrderStencil);
	partId = idutils::generateIdFromArray(new int[4] {0, 0, 0, 3}, 2, 4);
	stencilHierarchy->insertPartId(partId, 2, *dimOrderStencil);
	partId = idutils::generateIdFromArray(new int[4] {0, 0, 0, 4}, 2, 4);
	stencilHierarchy->insertPartId(partId, 2, *dimOrderStencil);
	partId = idutils::generateIdFromArray(new int[4] {1, 0, 0, 0}, 2, 4);
	stencilHierarchy->insertPartId(partId, 2, *dimOrderStencil);

	// Create a part-distribution-tree and put the parts for the current segment there
	BranchingContainer *distributionTree = new BranchingContainer(0, LpsDimConfig());
	vector<LpsDimConfig> *stencilLpsDimOrder = stencilConfigReceiver->generateDimOrderVector();
	PartIterator *stencilIterator = stencilHierarchy->getIterator();
	SuperPart *part = NULL;
	while ((part = stencilIterator->getCurrentPart()) != NULL) {
		distributionTree->insertPart(*stencilLpsDimOrder, 0, part->getPartId());
		stencilIterator->advance();
	}

	// Enter the parts for Segment #2 and #3 in the distribution tree
	partId = idutils::generateIdFromArray(new int[4] {1, 0, 0, 1}, 2, 4);
	distributionTree->insertPart(*stencilLpsDimOrder, 1, partId);
	partId = idutils::generateIdFromArray(new int[4] {1, 0, 0, 2}, 2, 4);
	distributionTree->insertPart(*stencilLpsDimOrder, 1, partId);
	partId = idutils::generateIdFromArray(new int[4] {1, 0, 0, 3}, 2, 4);
	distributionTree->insertPart(*stencilLpsDimOrder, 1, partId);
	partId = idutils::generateIdFromArray(new int[4] {1, 0, 0, 4}, 2, 4);
	distributionTree->insertPart(*stencilLpsDimOrder, 1, partId);
	partId = idutils::generateIdFromArray(new int[4] {2, 0, 0, 0}, 2, 4);
	distributionTree->insertPart(*stencilLpsDimOrder, 2, partId);
	partId = idutils::generateIdFromArray(new int[4] {2, 0, 0, 1}, 2, 4);
	distributionTree->insertPart(*stencilLpsDimOrder, 2, partId);
	partId = idutils::generateIdFromArray(new int[4] {2, 0, 0, 2}, 2, 4);
	distributionTree->insertPart(*stencilLpsDimOrder, 2, partId);
	partId = idutils::generateIdFromArray(new int[4] {2, 0, 0, 3}, 2, 4);
	distributionTree->insertPart(*stencilLpsDimOrder, 2, partId);
	partId = idutils::generateIdFromArray(new int[4] {2, 0, 0, 4}, 2, 4);
	distributionTree->insertPart(*stencilLpsDimOrder, 2, partId);

//	distributionTree->print(0, cout);

	// Consider the scenario for level 2 padding and create a confinement construction configuration for the
	// communication scenario
	ConfinementConstructionConfig *ccConfig = new ConfinementConstructionConfig(
			0, 						// local segment tag
			2, 						// sending side's LPS level (the last level of it's part container tree)
			stencilConfigSender, 	// partition configuration for the sending side
			2, 						// receiving side's LPS level
			stencilConfigReceiver,	// partition configuration for the receiving side
			1,						// confinement level LPS (the upper level LPS)
			stencilHierarchy, 		// sender side's part-container hierarchy
			stencilHierarchy,		// receiver side's part-container hierarchy
			distributionTree);		// part-distribution tree

	// Generate confinements
	List<Confinement*> *confinementList = Confinement::generateAllConfinements(ccConfig, 0);
	// Investigate the confinements
	cout << "Number of Confinements: " << confinementList->NumElements() << "\n";
	for (int i = 0; i < confinementList->NumElements(); i++) {
		cout << "Confinement No: " << i << "\n";
		Confinement *confinement = confinementList->Nth(i);
		CrossSegmentInteractionSpec *remoteInteractions = confinement->getRemoteInteractions();
		List<DataExchange*> *sendExchanges = remoteInteractions->generateSendExchanges();
		List<DataExchange*> *receiveExchanges = remoteInteractions->generateReceiveExchanges();
		if (sendExchanges->NumElements() > 0 || receiveExchanges->NumElements() > 0) {
			cout << "This confinement involves remote and/or cross-containers interactions\n";
			remoteInteractions->describe(1, cout);
			cout << "Send Exchange Count: " << sendExchanges->NumElements() << "\n";
			for (int j = 0; j < sendExchanges->NumElements(); j++) {
				cout << "\nExchange #" << j << "\n";
				sendExchanges->Nth(j)->describe(0, cout);
			}
			cout << "\nReceive Exchange Count: " << receiveExchanges->NumElements() << "\n";
			for (int j = 0; j < receiveExchanges->NumElements(); j++) {
				cout << "\nExchange #" << j << "\n";
				receiveExchanges->Nth(j)->describe(0, cout);
			}
		} else cout << "This confinement involves no cross segment/container interactions";

		IntraContainerInteractionSpec *localInteractions = confinement->getLocalInteractions();
		if (localInteractions != NULL && localInteractions->isSynchronizationNeeded()) {
			cout << "\nThis confinement involves intra-container parts interactions\n";
			localInteractions->describe(1, cout);
			List<DataExchange*> *dataExchanges = localInteractions->generateExchanges();
			cout << "\nExchange Count: " << dataExchanges->NumElements() << "\n";
			for (int j = 0; j < dataExchanges->NumElements(); j++) {
				cout << "\nLocal Exchange #" << j << "\n";
				dataExchanges->Nth(j)->describe(0, cout);
			}
		} else cout << "\nThis confinement involves no intra-container parts interactions\n";
		cout << "\n\n\n";
	}

	return 0;
}


