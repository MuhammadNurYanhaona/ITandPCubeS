#!/bin/bash

# keep track of the current directory
installer_dir=`pwd`

# list of compiler directories
multicore_compiler_dir=compilers/multicore-backend  
segmented_memory_compiler_dir=compilers/segmented-memory-backend

# enter the multicore compiler directory and clean it
echo "cleaning up the IT multicore backend compiler"
cd $multicore_compiler_dir
make -f MakeFile-Compiler clean
rm -f micc

# come back to the installer directory and delete the compiler script
cd $installer_dir
rm -f micc

# enter the multicore compiler directory and clean it
echo "cleaning up the IT segmented memory backend compiler"
cd $segmented_memory_compiler_dir
make -f MakeFile-Compiler clean
rm -f sicc

# come back to the installer directory and delete the compiler script
cd $installer_dir
rm -f sicc
