This directory contains mock implementations of all platform specific
code generation related functions. Back-end compilers should skip
the content of this directory and provide their own and appropriate
implementations replacing these mock implementations.

Note that any future file added in this directory should end with the
suffix "_hack.cc" to be excluded during the make process of a back-end
compiler.

