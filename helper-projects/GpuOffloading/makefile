.PHONY: clean

# Set the default target. When you make with no arguments, this will be the target built.
EXECUTABLE = gpu_offload.o
PRODUCTS = $(EXECUTABLE) 
default: $(PRODUCTS)

# The backend C++ MPI compiler
C_COMPILER=mpic++

# The backend CUDA compiler
CUDA_COMPILER=nvcc

# Set up the list of C++ and CUDA source files
C_SRCS = $(shell find . ! -name "hashtable.cc" -name "*.cc" -not -path "./gpu*")
CUDA_SRCS = $(shell find . -name "*.cu" -path "./gpu*")

# C++ and CUDA object files
C_OBJS = $(patsubst %.cc, %.o, $(filter %.cc,$(C_SRCS)))
CUDA_OBJS = $(patsubst %.cu, %.co, $(filter %.cu,$(CUDA_SRCS)))

SRC_JUNK = ./*.o ./*/*.o ./*.co ./*/*.co ./*/*/*.co

# Define the tools we are going to use
CXX= $(C_COMPILER)
LD = $(C_COMPILER)
NVCC = $(CUDA_COMPILER)

#--------------------------------------------------------------------------------------------------
# Set up the necessary flags for the tools
#--------------------------------------------------------------------------------------------------

# settings for unoptimized code generation
CPPFLAGS = -g -Wall -fno-inline -Wno-unused -Wno-sign-compare -O0

# settings for newer GPGPU target hardwares 
GPUARCH = sm_30 

# Link with standard c library, math library, and the CUDA library
LIBS = -lc -lm -lcudart

#--------------------------------------------------------------------------------------------------
# Rules for various parts of the target
#--------------------------------------------------------------------------------------------------

%.o: %.cc
	$(CXX) $(CPPFLAGS) -c $*.cc -o $@

%.co: %.cu
	$(NVCC) -arch=$(GPUARCH) -c $*.cu -o $@

$(EXECUTABLE): $(C_OBJS) $(CUDA_OBJS)
	$(LD) -o $@ $(C_OBJS) $(CUDA_OBJS) $(LIBS)

clean:
	rm -f $(SRC_JUNK) 
	rm -f $(PRODUCTS)

	

