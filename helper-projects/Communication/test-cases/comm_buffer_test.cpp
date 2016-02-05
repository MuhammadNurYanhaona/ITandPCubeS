/* The goal of this test is to verify that different kinds of communication buffers that we have created
 * read and write data properly. The part-hierarchy scenario used for this test is the same hierarchy as
 * in the data-read-write-test.
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
#include "../communication/comm_buffer.h"

#include <iostream>
#include <vector>
#include <cstdlib>

using namespace std;

int mainCBT() {

	// sample matrix dimensions
	Dimension rows = Dimension(100);
	Dimension cols = Dimension(100);

	// data partition configuration
	DataItemConfig *dataConfig = new DataItemConfig(2, 4);
	dataConfig->setDimension(0, rows);
	dataConfig->setDimension(1, cols);
	// level 0
	dataConfig->setPartitionInstr(0, 0, new BlockCountInstr(2));
	dataConfig->getInstruction(0, 0)->setPriorityOrder(0);
	dataConfig->setPartitionInstr(0, 1, new BlockCountInstr(2));
	dataConfig->getInstruction(0, 1)->setPriorityOrder(1);
	dataConfig->setLpsIdOfLevel(0, 1);
	// level 1
	dataConfig->setPartitionInstr(1, 0, new BlockStrideInstr(5, 5));
	dataConfig->getInstruction(1, 0)->setPriorityOrder(0);
	dataConfig->setPartitionInstr(1, 1, new BlockStrideInstr(5, 5));
	dataConfig->getInstruction(1, 1)->setPriorityOrder(1);
	dataConfig->setLpsIdOfLevel(1, 2);
	// level 2
	dataConfig->setPartitionInstr(2, 0, new BlockSizeInstr(5));
	dataConfig->getInstruction(2, 0)->setPriorityOrder(0);
	dataConfig->setPartitionInstr(2, 1, new BlockSizeInstr(5));
	dataConfig->getInstruction(2, 1)->setPriorityOrder(1);
	dataConfig->setLpsIdOfLevel(2, 3);
	// level 3
	dataConfig->setPartitionInstr(3, 0, new StrideInstr(2));
	dataConfig->getInstruction(3, 0)->setPriorityOrder(0);
	dataConfig->setPartitionInstr(3, 1, new StrideInstr(2));
	dataConfig->getInstruction(3, 1)->setPriorityOrder(1);
	dataConfig->setLpsIdOfLevel(3, 4);
	dataConfig->updateParentLinksOnPartitionConfigs();

	// generate a part container hierarchy containing two parts
	vector<DimConfig> *dimOrder = dataConfig->generateDimOrderVectorWithoutLps();
	PartIdContainer *dataHierarchy = new PartListContainer(dimOrder->at(0));
	List<int*> *partId = NULL;
	partId = idutils::generateIdFromArray(new int[8] {0, 0, 0, 0, 0, 0, 0, 0}, 2, 8);
	dataHierarchy->insertPartId(partId, 2, *dimOrder);
	partId = idutils::generateIdFromArray(new int[8] {1, 1, 4, 4, 1, 1, 1, 1}, 2, 8);
	dataHierarchy->insertPartId(partId, 2, *dimOrder);

	// fold the part container hierarchy and generate data exchanges as substitutes for communication buffers
	List<PartFolding*> *foldList = new List<PartFolding*>;
	dataHierarchy->foldContainer(foldList);
	List<DataExchange*> *dataExchangeList = new List<DataExchange*>;
	for (int i = 0; i < foldList->NumElements(); i++) {
		PartFolding* fold = foldList->Nth(i);
		Participant *sender = new Participant(SEND, NULL, NULL);
		sender->addSegmentTag(0);
		Participant *receiver = new Participant(RECEIVE, NULL, NULL);
		receiver->addSegmentTag(0);
		dataExchangeList->Append(new DataExchange(sender, receiver, fold->generateIntervalDesc(dataConfig)));
	}

	// create and allocate the two data parts (we will use these two data parts as senders)
	Dimension *part0Dims = new Dimension[2];
	part0Dims[0] = Dimension(3);
	part0Dims[1] = Dimension(3);
	partId = idutils::generateIdFromArray(new int[8] {0, 0, 0, 0, 0, 0, 0, 0}, 2, 8);
	DataPart *part0 = new DataPart(new PartMetadata(2, partId, part0Dims));
	DataPart::allocate<double>(part0);
	Dimension *part1Dims = new Dimension[2];
	part1Dims[0] = Dimension(2);
	part1Dims[1] = Dimension(2);
	partId = idutils::generateIdFromArray(new int[8] {1, 1, 4, 4, 1, 1, 1, 1}, 2, 8);
	DataPart *part1 = new DataPart(new PartMetadata(2, partId, part1Dims));
	DataPart::allocate<double>(part1);
	List<DataPart*> *dataPartList1 = new List<DataPart*>;
	dataPartList1->Append(part0);
	dataPartList1->Append(part1);

	// iterate over the part hierarchy and replace the two placeholder super-parts with two data part locators
	PartIterator *iterator = dataHierarchy->getIterator();
	SuperPart *part = NULL;
	int index = 0;
	while ((part = iterator->getCurrentPart()) != NULL) {
		iterator->replaceCurrentPart(new PartLocator(part->getPartId(), 2, index));
		iterator->advance();
		index++;
	}

	// populate the two data parts with constants
	double *data = reinterpret_cast<double*>(part0->getData());
	for (int i = 0; i < 9; i++) data[i] = 1;
	data = reinterpret_cast<double*>(part1->getData());
	for (int i = 0; i < 4; i++) data[i] = 2;

	// we verify that data parts initialization worked as intended
	cout << "sender part 1: ";
	data = reinterpret_cast<double*>(part0->getData());
	for (int i = 0; i < 9; i++) cout << data[i] << " ";
	cout << "\n";
	cout << "sender part 2: ";
	data = reinterpret_cast<double*>(part1->getData());
	for (int i = 0; i < 4; i++) cout << data[i] << " ";
	cout << "\n";

	// then we create two new data parts that will be used as receivers
	partId = idutils::generateIdFromArray(new int[8] {0, 0, 0, 0, 0, 0, 0, 0}, 2, 8);
	DataPart *part00 = new DataPart(new PartMetadata(2, partId, part0Dims));
	DataPart::allocate<double>(part00);
	partId = idutils::generateIdFromArray(new int[8] {1, 1, 4, 4, 1, 1, 1, 1}, 2, 8);
	DataPart *part01 = new DataPart(new PartMetadata(2, partId, part1Dims));
	DataPart::allocate<double>(part01);
	List<DataPart*> *dataPartList2 = new List<DataPart*>;
	dataPartList2->Append(part00);
	dataPartList2->Append(part01);

	// we verify that the data parts are zero initialized
	cout << "Before transfer\n";
	cout << "receiver part 1: ";
	data = reinterpret_cast<double*>(part00->getData());
	for (int i = 0; i < 9; i++) cout << data[i] << " ";
	cout << "\n";
	cout << "receiver part 2: ";
	data = reinterpret_cast<double*>(part01->getData());
	for (int i = 0; i < 4; i++) cout << data[i] << " ";
	cout << "\n";

	// create a confinement for data transfer as the communication buffers' constructors need one
	ConfinementConstructionConfig *ccConfig = new ConfinementConstructionConfig(
			0, 					// local segment tag
			3, 					// sending side's LPS level (the last level of it's part container tree)
			dataConfig, 		// partition configuration for the sending side
			3, 					// receiving side's LPS level
			dataConfig,			// partition configuration for the receiving side
			0,					// confinement level LPS (in this case the Root LPS)
			dataHierarchy, 		// sender side's part-container hierarchy
			dataHierarchy,		// receiver side's part-container hierarchy
			NULL);				// part-distribution tree

	// create a synchronization configuration for the transfer
	SyncConfig *syncConfig = new SyncConfig(ccConfig, dataPartList1, dataPartList2, sizeof(double));

	// create communication buffers and perform read and write
	PhysicalCommBuffer *commBuffer1 = new PhysicalCommBuffer(dataExchangeList->Nth(0), syncConfig);
	commBuffer1->readData();
	commBuffer1->writeData();
	VirtualCommBuffer *commBuffer2 = new VirtualCommBuffer(dataExchangeList->Nth(1), syncConfig);
	commBuffer2->readData();

	// test the content of the receiver parts after data transfer
	cout << "After transfer\n";
	cout << "receiver part 1: ";
	data = reinterpret_cast<double*>(part00->getData());
	for (int i = 0; i < 9; i++) cout << data[i] << " ";
	cout << "\n";
	cout << "receiver part 2: ";
	data = reinterpret_cast<double*>(part01->getData());
	for (int i = 0; i < 4; i++) cout << data[i] << " ";
	cout << "\n";

	// update the sender parts again with new data
	data = reinterpret_cast<double*>(part0->getData());
	for (int i = 0; i < 9; i++) data[i] = i * i;
	data = reinterpret_cast<double*>(part1->getData());
	for (int i = 0; i < 4; i++) data[i] = i * i * i;

	// we verify that data parts re-initialization worked as intended
	cout << "Rewritten sender parts:\n";
	cout << "sender part 1: ";
	data = reinterpret_cast<double*>(part0->getData());
	for (int i = 0; i < 9; i++) cout << data[i] << " ";
	cout << "\n";
	cout << "sender part 2: ";
	data = reinterpret_cast<double*>(part1->getData());
	for (int i = 0; i < 4; i++) cout << data[i] << " ";
	cout << "\n";

	// now create pre-processed communication buffers
	PreprocessedPhysicalCommBuffer *commBuffer00 =
			new PreprocessedPhysicalCommBuffer(dataExchangeList->Nth(0), syncConfig);
	commBuffer00->readData();
	commBuffer00->writeData();
	PreprocessedVirtualCommBuffer *commBuffer01 =
			new PreprocessedVirtualCommBuffer(dataExchangeList->Nth(1), syncConfig);
	commBuffer01->readData();
	commBuffer01->writeData();

	// test the content of the receiver parts after data transfer
	cout << "After second transfer\n";
	cout << "receiver part 1: ";
	data = reinterpret_cast<double*>(part00->getData());
	for (int i = 0; i < 9; i++) cout << data[i] << " ";
	cout << "\n";
	cout << "receiver part 2: ";
	data = reinterpret_cast<double*>(part01->getData());
	for (int i = 0; i < 4; i++) cout << data[i] << " ";
	cout << "\n";

	return 0;
}



