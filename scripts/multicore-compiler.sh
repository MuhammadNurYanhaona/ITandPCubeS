#!/bin/bash

# keep track of the current directory
current_dir=`pwd`

# get the installer directory and associated subdirectories
installer_dir=.
config_dir=$installer_dir/config

# multicore compiler directory and executable
multicore_compiler_dir=$installer_dir/compilers/multicore-backend/
multicore_compiler=./micc

# assign command line parameters to different properties
src_code=$1
mapping_file=$2
executable=$current_dir/mult.o

# validate that the user specified at least the source and mapping files
if [ "$#" -lt 2 ]; then
	echo "Two arguments are mandatory to generate an executable"
	echo "1. the IT source code file"
	echo "2. the mapping file"
	echo "Optionally, you can specify the intended name of the executable as the third parameter"
	exit 1
fi
if [ "$#" -gt 2 ]; then
	executable=$3
fi

# generate an executable file beforehand and get its absolute location
touch $executable
executable=`readlink -f $executable`

# jump into the installation directory
cd $installer_dir

# determine a temp directory to generate the intermediate C++ code
t=`date +%s`
build_dir=tmp_$t

# retrieve the hardware description files' location
machine_model_dir=`cat ${config_dir}/executable.properties | grep 'multicore.machine.model' | cut -d '=' -f2`
echo "machine model for generating the executable is taken from: $machine_model_dir"

# retrieve the two files needed for code generation from hardware description directory
pcubes_file=`ls ${machine_model_dir}/*.ml`
core_numbering_file=`ls ${machine_model_dir}/*.cn`

# validate that all necessary input files exist
if [ ! -f "$src_code" ]; then
	echo "IT source '$src_code' not found"
	exit 1
fi 
if [ ! -f "$mapping_file" ]; then
	echo "Mapping file '$mapping_file' not found"
	exit 1
fi 
if [ ! -f "$pcubes_file" ]; then
	echo "PCubeS file '$pcubes_file' not found"
	exit 1
fi 
if [ ! -f "$core_numbering_file" ]; then
	echo "Core numbering file '$core_numbering_file' not found"
	exit 1
fi

# generate absolute file paths from relative paths
src=`readlink -f $src_code`
pcubes=`readlink -f $pcubes_file`
cores=`readlink -f $core_numbering_file`
mapping=`readlink -f $mapping_file`

# determine what backend C++ compiler has been used to install the IT multicore compiler
multicore_c=`cat ${config_dir}/compiler.properties | grep 'multicore.backend.c.compiler' | cut -d '=' -f2`

# determine what compiler optimization flags are enabled for the backend C++ compiler
c_opt_flags=`cat ${config_dir}/executable.properties | grep 'c.optimization.flags' | cut -d '=' -f2`

# generate intermediate C++ source codes
echo ""
echo ""
echo "generating intermediate code"
cd $multicore_compiler_dir
${multicore_compiler} $src $pcubes $cores $mapping $build_dir

# generate the binary from the intermediate source code and delete the intermediate source code
echo ""
echo ""
echo "generating executable from the intermediate code"
make -f MakeFile-Executable C_COMPILER=$multicore_c EXECUTABLE=$executable BUILD_SUBDIR=$build_dir C_OPT_FLAGS="$c_opt_flags"
make -f MakeFile-Executable BUILD_SUBDIR=$build_dir clean 

cd $current_dir
echo "executable has been written in the file: $executable"

