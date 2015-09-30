/* The goal of this test is to verify the accuracy of data transfers between communication buffer and operating
 * memory part within a segment. We have a complicated hierarchical partition specification. We enter a few sample
 * parts for the segment. Then create an interval description for the data parts and treat the description as the
 * specification of the communication buffer. Then we populate constants to the operating memory parts through the
 * buffer. Finally we print the data contents in the parts to check if they have the expected contents.
 */

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

int mainDRWT() {

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
		dataExchangeList->Append(new DataExchange(NULL, NULL, fold->generateIntervalDesc(dataConfig)));
	}

	// verify that the exchange descriptions are correct by iterating over their elements
	cout << "data items to be updated\n";
	for (int i = 0; i < dataExchangeList->NumElements(); i++) {
		cout << "Exchange #" << i << "\n";
		DataExchange *exchange = dataExchangeList->Nth(i);
		ExchangeIterator *iterator = new ExchangeIterator(exchange);
		while (iterator->hasMoreElements()) {
			vector<int> *index = iterator->getNextElement();
			for (unsigned int d = 0; d < index->size(); d++) {
				cout << index->at(d) << ", ";
			}
			cout << "\n";
		}
	}

	// create and allocate the two data parts
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
	List<DataPart*> *dataPartList = new List<DataPart*>;
	dataPartList->Append(part0);
	dataPartList->Append(part1);

	// print the initial states of the data parts
	cout << "\nThe content of the data parts before data transfer\n";
	for (int i = 0; i < dataPartList->NumElements(); i++) {
		cout << "Data part #" << i << "\n";
		DataPart *part = dataPartList->Nth(i);
		DataPart::print<double>(part, cout);
	}

	// iterate over the part hierarchy and replace the two placeholder super-parts with two data part locators
	PartIterator *iterator = dataHierarchy->getIterator();
	SuperPart *part = NULL;
	int index = 0;
	while ((part = iterator->getCurrentPart()) != NULL) {
		iterator->replaceCurrentPart(new PartLocator(part->getPartId(), 2, index));
		iterator->advance();
		index++;
	}

	// add some more entries in the part hierarchy just to make the search more interesting
	partId = idutils::generateIdFromArray(new int[8] {0, 1, 0, 0, 0, 0, 0, 0}, 2, 8);
	dataHierarchy->insertPartId(partId, 2, *dimOrder);
	partId = idutils::generateIdFromArray(new int[8] {1, 1, 4, 4, 1, 1, 1, 0}, 2, 8);
	dataHierarchy->insertPartId(partId, 2, *dimOrder);
	partId = idutils::generateIdFromArray(new int[8] {1, 1, 4, 4, 1, 1, 0, 0}, 2, 8);
	dataHierarchy->insertPartId(partId, 2, *dimOrder);

	// create a transfer specification, a data part specification, and a transform index vector
	TransferSpec *writeSpec = new TransferSpec(COMM_BUFFER_TO_DATA_PART, sizeof(double));
	DataPartSpec *dataPartSpec = new DataPartSpec(dataPartList, dataConfig);
	vector<XformedIndexInfo*> *transformVector = new vector<XformedIndexInfo*>;
	transformVector->push_back(new XformedIndexInfo());
	transformVector->push_back(new XformedIndexInfo());

	// do the writing
	cout << "\nUpdate the data parts\n";
	for (int i = 0; i < dataExchangeList->NumElements(); i++) {
		DataExchange *exchange = dataExchangeList->Nth(i);
		ExchangeIterator *iterator = new ExchangeIterator(exchange);
		double data = i + 1;
		while (iterator->hasMoreElements()) {
			vector<int> *index = iterator->getNextElement();
			writeSpec->setBufferEntry(reinterpret_cast<char*>(&data), index);
			dataPartSpec->initPartTraversalReference(index, transformVector);
			dataHierarchy->transferData(transformVector, writeSpec, dataPartSpec);
		}
	}

	// print the final states of the data parts by read transfers
	cout << "\nThe content of the data parts after data transfer\n";
	TransferSpec *readSpec = new TransferSpec(DATA_PART_TO_COMM_BUFFER, sizeof(double));
	for (int i = 0; i < dataExchangeList->NumElements(); i++) {
		cout << "Data part #" << i << "\n";
		DataExchange *exchange = dataExchangeList->Nth(i);
		ExchangeIterator *iterator = new ExchangeIterator(exchange);
		while (iterator->hasMoreElements()) {
			double data = 0.0;
			vector<int> *index = iterator->getNextElement();
			readSpec->setBufferEntry(reinterpret_cast<char*>(&data), index);
			dataPartSpec->initPartTraversalReference(index, transformVector);
			dataHierarchy->transferData(transformVector, readSpec, dataPartSpec);
			cout << data << " ";
		}
		cout << "\n";
	}

	return 0;
}




