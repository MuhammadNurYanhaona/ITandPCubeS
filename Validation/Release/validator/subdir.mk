################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../validator/CSRMVMultiplyValidator.cpp \
../validator/ConjugateGradientValidator.cpp \
../validator/DotProductValidator.cpp \
../validator/ProgramLUValidator.cpp \
../validator/blockMMMultValidator.cpp \
../validator/colLuFactorizationValidator.cpp \
../validator/matrixMatrixMultValidator.cpp \
../validator/rowLuFactorizationValidator.cpp 

OBJS += \
./validator/CSRMVMultiplyValidator.o \
./validator/ConjugateGradientValidator.o \
./validator/DotProductValidator.o \
./validator/ProgramLUValidator.o \
./validator/blockMMMultValidator.o \
./validator/colLuFactorizationValidator.o \
./validator/matrixMatrixMultValidator.o \
./validator/rowLuFactorizationValidator.o 

CPP_DEPS += \
./validator/CSRMVMultiplyValidator.d \
./validator/ConjugateGradientValidator.d \
./validator/DotProductValidator.d \
./validator/ProgramLUValidator.d \
./validator/blockMMMultValidator.d \
./validator/colLuFactorizationValidator.d \
./validator/matrixMatrixMultValidator.d \
./validator/rowLuFactorizationValidator.d 


# Each subdirectory must supply rules for building sources it contributes
validator/%.o: ../validator/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


