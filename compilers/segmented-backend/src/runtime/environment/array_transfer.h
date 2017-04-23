#ifndef _H_array_transfer
#define _H_array_transfer

/* In the segmented memory architecture assignment of an array (or part of an array) to some other array needs to be
 * handled with great care. As the memory for array parts within even a single segmented is divided, let alone the
 * distribution of regions or arrays among different segments, a simple 'a = b' kind of assignment may lead to a lot
 * of data reorganization and communication. This is unlike in the case of the multicore compiler where passing along
 * the reference of the right hand side of the expression would do. 
 * The strategy we chose is to encode the transfer characteristics into a transfer configuration object that will be
 * passed to the task environment as an environment update instruction and the runtime environment management library
 * takes care of whatever consequences there may be for the assignment.
 * At the same time the old-style reference passing of arrays taken from the multicore compiler remain intact for 
 * local array variables inside the program coordinator function and compute stages of tasks.    
 */

#include "../../../../common-libs/utils/list.h"
#include "../../../../common-libs/domain-obj/structure.h"

/* This class tells what index ranges of a particular dimension from the source array should be transferred to the
 * destination array
 */
class DimensionTransfer {
  protected:
	int dimensionId;
	Range transferRange;
  public:
	DimensionTransfer(int dimensionId, Range transferRange) {
		this->dimensionId = dimensionId;
		this->transferRange = transferRange;
	}
	int getDimensionId() { return dimensionId; }
	Range getTransferRange() { return transferRange; }
};

/* This class encodes the data transfer requirements for an entire source to destination array assignment. Note that
 * the ultimate source of content array me be transferred through several local intermediate variables before the
 * assignment the current instance is responsible for encoding. Thus, a parent link is maintained to be able track
 * back to the information source.  
 */
class ArrayTransferConfig {
  protected:
	// every array used in the program will have an associated default transfer config object, if no assignment
	// is made to the array then the transfer config remains inactive.
	bool inactive;

	// the source can itself be an array reference, in case it is a local array, or it can be a property of some
	// task environment; only in the second case the propertyName will be non-NULL 
	void *source;
	const char *propertyName;

	// this tells how the dimensions of the source are related to the dimensions of the destination 
	List<DimensionTransfer> *transferDimConfig;

	// previous transfer config that define the right hand side of the current transfer
	ArrayTransferConfig *parent;
  public:
	ArrayTransferConfig();
	void setSource(void *source, const char *propertyName = NULL);
	void *getSource() { return source; }
	const char *getPropertyName() { return propertyName; }
	void recordTransferDimConfig(int dimId, int index);
	void recordTransferDimConfig(int dimId, Range transferRange);
	void setParent(ArrayTransferConfig *parent);
	bool isActive() { return !inactive; }
	void setTransferDimConfig(List<DimensionTransfer> *config) { this->transferDimConfig = config; }

	// this function is used to convert the chain of array transfer config into a single direct transfer for the
	// information source to the current destination array
	ArrayTransferConfig *getCollapsed();

	// function to deactivate a transfer config once the operation dictated by it has been undertaken; this is 
	// needed to avoid repetitive (and false) updates on the same array
	void reset();

	// copy the first dimCount dimensions' information from the transfer configuration to the partition properties
	// of the part-dimension array passed as the first argument; note that this assume that dimension transfer 
	// information for the concerned dimensions are available
	void copyDimensions(PartDimension *PartDimensions, int dimCount);
	
	// recursively find the range of the first dimension transfer from the current configuration to the source 
	// for the dimension number passed as the argument 
	Range getNearestDimTransferRange(int dimensionNo);	
};

#endif
