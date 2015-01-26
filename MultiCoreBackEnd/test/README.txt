Codes is in this directory are for comparing performance of compiler generated
code with a hand-written code for any particular task/program. Note that the
codes are written in a way that they are dependent on data structures that the
compiler should generate. Therefore, we cannot just compile and run these codes
directly in isolation. Rather they need to be placed in their corresponding 
generated code to work. Once we do that, we can invoke the reference hand-written
code from the main function and compare its performance with generated code.

The files here have the extension .cpp instead of .cc. So they are ignored by
the make routine.
