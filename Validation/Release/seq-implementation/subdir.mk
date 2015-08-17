################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../seq-implementation/blockMMMultiply.cpp \
../seq-implementation/colLuFactorization.cpp \
../seq-implementation/rowLuFactorization.cpp \
../seq-implementation/seqMatrixMatrixMult.cpp 

OBJS += \
./seq-implementation/blockMMMultiply.o \
./seq-implementation/colLuFactorization.o \
./seq-implementation/rowLuFactorization.o \
./seq-implementation/seqMatrixMatrixMult.o 

CPP_DEPS += \
./seq-implementation/blockMMMultiply.d \
./seq-implementation/colLuFactorization.d \
./seq-implementation/rowLuFactorization.d \
./seq-implementation/seqMatrixMatrixMult.d 


# Each subdirectory must supply rules for building sources it contributes
seq-implementation/%.o: ../seq-implementation/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


