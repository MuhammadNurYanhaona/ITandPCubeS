#!/bin/bash

# keep track of the current directory
installer_dir=`pwd`

# list of compiler directories
multicore_compiler_dir=compilers/multicore-backend  
segmented_memory_compiler_dir=compilers/segmented-memory-backend

# check if multicore-backend IT compiler should installed
multicore_enabled=`cat config/compiler.properties | grep 'multicore.backend.enabled' | cut -d '=' -f2`

# install the multicore-backend compiler if the user wants it
if [ "$multicore_enabled" == "true" ]; then
	# select the back-end c++ compiler from the configuration file
	multicore_c=`cat config/compiler.properties | grep 'multicore.backend.c.compiler' | cut -d '=' -f2`
	echo "underlying C++ compiler for the multicore back-end: $multicore_c"
	# go inside the compiler directory
	cd $multicore_compiler_dir
	# make the compiler executable
	echo "generating the IT compiler for multicore back-end"
	make -f MakeFile-Compiler clean 
	make -f MakeFile-Compiler C_COMPILER=$multicore_c
fi

# come back to the installer directory
cd $installer_dir

# if multicore compiler has been installed then create a compiler script for it in the current directory
if [ "$multicore_enabled" == "true" ]; then
	# also set up the installer directory in the copied script as an absolute path so that the script can be run from anywhere
	replacement_expr="s#installer_dir=.#installer_dir=$installer_dir#g"
	cat scripts/multicore-compiler.sh  | sed -e $replacement_expr > micc
	chmod a+x micc
	echo "The generated IT multicore compiler is: micc"
fi

# check if the segmented-memory-backend IT compiler should installed
segmented_enabled=`cat config/compiler.properties | grep 'segmented.memory.backend.enabled' | cut -d '=' -f2`

# install the segmented-memory-backend compiler if the user wants it
if [ "$segmented_enabled" == "true" ]; then
	# select the back-end c++ compiler from the configuration file
	segmented_memory_c=`cat config/compiler.properties | grep 'segmented.memory.backend.c.compiler' | cut -d '=' -f2`
	echo "underlying C++ compiler for the segmented-memory back-end: $segmented_memory_c"
	# go inside the compiler directory
	cd $segmented_memory_compiler_dir
	# make the compiler executable
	echo "generating the IT compiler for segmented-memory back-end"
	make -f MakeFile-Compiler clean 
	make -f MakeFile-Compiler C_COMPILER=$segmented_memory_c
fi

# come back to the installer directory
cd $installer_dir

# if segmented-memory compiler has been installed then create a compiler script for it in the current directory
if [ "$segmented_enabled" == "true" ]; then
	# also set up the installer directory in the copied script as an absolute path so that the script can be run from anywhere
	replacement_expr="s#installer_dir=.#installer_dir=$installer_dir#g"
	cat scripts/segmented-memory-compiler.sh  | sed -e $replacement_expr > smicc
	chmod a+x smicc
	echo "The generated IT segmented memory compiler is: smicc"
fi

