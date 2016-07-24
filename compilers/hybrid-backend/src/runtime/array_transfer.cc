#include "array_transfer.h"
#include "structure.h"
#include "../utils/list.h"

#include <stack>
#include <cstdlib>
#include <iostream>

//-------------------------------------------------------- Array Transfer Config ---------------------------------------------------/

ArrayTransferConfig::ArrayTransferConfig() {
	inactive = true;
	source = NULL;
	propertyName = NULL;
	transferDimConfig = new List<DimensionTransfer>;
	parent = NULL;
}

void ArrayTransferConfig::setSource(void *source, const char *propertyName) {
	this->source = source;
	this->propertyName = propertyName;
	inactive = false;
}

void ArrayTransferConfig::setParent(ArrayTransferConfig *parent) { 
	if (parent != NULL && !parent->inactive) {
		this->parent = parent;
	} 
}

void ArrayTransferConfig::recordTransferDimConfig(int dimId, int index) {
	transferDimConfig->Append(DimensionTransfer(dimId, Range(index)));
}

void ArrayTransferConfig::recordTransferDimConfig(int dimId, Range transferRange) {
	transferDimConfig->Append(DimensionTransfer(dimId, transferRange));
}

ArrayTransferConfig *ArrayTransferConfig::getCollapsed() {
	
	// later transfer configurations override earlier configurations' dimension boundaries; thus we need a stack to
	// determine the information source to current destination's dimension association 
	std::stack<ArrayTransferConfig*> configStack;
	configStack.push(this);
	ArrayTransferConfig *current = this->parent;
	while (current != NULL) {
		configStack.push(current);

		// avoid loop back reference to the original destination array to create an infinit loop; a loop back
		// refrerence should rather be treated as an effort to curtail the current destination array
		if (current == this) break;
		
		current = current->parent;
	}

	// record the information source reference
	void *primarySource = configStack.top()->source;
	const char *primarySourceProperty = configStack.top()->propertyName;
	List<DimensionTransfer>	*accumulatedTransfers = new List<DimensionTransfer>;
	
	// track the information source to the current destination dimension association 
	while (!configStack.empty()) {
		current = configStack.top();
		List<DimensionTransfer> *currentDimTransfers = current->transferDimConfig;
		for (int i = 0; i < currentDimTransfers->NumElements(); i++) {
			DimensionTransfer dimXfer = currentDimTransfers->Nth(i);
			int matchingIndex = -1;
			for (int j = 0; j < accumulatedTransfers->NumElements(); j++) {
				DimensionTransfer includedXfer = accumulatedTransfers->Nth(j);
				if (includedXfer.getDimensionId() == dimXfer.getDimensionId()) {
					matchingIndex = j;
					break;
				}
			}
			if (matchingIndex == -1) {
				accumulatedTransfers->Append(dimXfer);
			} else {
				accumulatedTransfers->RemoveAt(matchingIndex);
				accumulatedTransfers->InsertAt(dimXfer, matchingIndex);
			}
		}
		configStack.pop();
	}

	// finally, create a new array transfer config for the collapsed configuration and return it
	ArrayTransferConfig *collapsedConfig = new ArrayTransferConfig();
	collapsedConfig->setSource(primarySource, primarySourceProperty);
	collapsedConfig->setTransferDimConfig(accumulatedTransfers);
	return collapsedConfig;
}

void ArrayTransferConfig::reset() {
	inactive = true;
	source = NULL;
	propertyName = NULL;
	parent = NULL;
	transferDimConfig->clear();
}

void ArrayTransferConfig::copyDimensions(PartDimension *partDimensions, int dimCount) {
	for (int i = 0; i < dimCount; i++) {
		Range range = getNearestDimTransferRange(i);
		Dimension dim = Dimension();
		dim.range = range;
		dim.setLength();
		partDimensions[i].partition = dim;
	}
}

Range ArrayTransferConfig::getNearestDimTransferRange(int dimensionNo) {

	for (int i = 0; i < transferDimConfig->NumElements(); i++) {
		DimensionTransfer dimTransfer = transferDimConfig->Nth(i);
		if (dimTransfer.getDimensionId() == dimensionNo) {
			return dimTransfer.getTransferRange();
		}
	}
	if (parent != NULL) return parent->getNearestDimTransferRange(dimensionNo);

	std::cout << "Dimension transfer information has not been found for " << dimensionNo << "\n";	
	std::exit(EXIT_FAILURE);
}


