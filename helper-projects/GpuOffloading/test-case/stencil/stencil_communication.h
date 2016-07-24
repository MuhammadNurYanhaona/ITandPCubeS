#ifndef _H_stencil_communication
#define _H_stencil_communication

/* This header file provides two classes that work as the substitute for the communicators of the segmented memory
 * compiler. It would be too much hassle if you dragged the communicator library from the compiler to this experimental
 * project. For simplication, we just assumed that at the process level, data is partitioned into slab of rows with
 * fixed amount of padding. Each MPI process then have to communicate with its neighbor boundary rows from its first
 * and last plate parts.
 */

#include "stencil_structure.h"
#include "../../runtime/structure.h"
#include "../../utils/list.h"

/* This communicator is intended for the scenario where we do not do the duplicate computation in the CPU to verify
 * correctness of the offloading implementation. */
class StencilComm {
  protected:
	int padding;
	int lpuCount;
	bool active;
	Range localParts;
	List<stencil::PlatePart*> *partList;
  public:
	StencilComm(int padding, int lpuCount, List<stencil::PlatePart*> *partList);
	virtual void synchronizeDataParts();
  protected:
	void performLocalExchange(List<stencil::PlatePart*> *listOfParts);
	void performRemoteExchange(List<stencil::PlatePart*> *listOfParts);
};

/* This communicator should be used when the host CPU duplicates the computation done by the GPU over redundant data */
class StencilCommWithVerifier : public StencilComm {
  private:
	List<stencil::PlatePart*> *duplicatePartList;
  public:
	StencilCommWithVerifier(int padding, int lpuCount, 
		List<stencil::PlatePart*> *partList, 
		List<stencil::PlatePart*> *duplicatePartList);
	void synchronizeDataParts();
};

#endif
