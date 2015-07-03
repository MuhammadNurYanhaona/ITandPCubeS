#!/bin/bash

# first convert two input text files into binary files
../../../tools/text-to-binary data/a-20 a
../../../tools/text-to-binary data/b-20 b

# then create an input output configuration file for the program
echo a >  input
echo b >> input
echo 4 >> input
echo 4 >> input
echo c >> input
echo 4 >> input

# then run the mm-multiply program for all segments
for segment in {0..3}
do
	./mmm-on-laptop $segment < input
done

# convert the output file from binary to text form to output validation
../../../tools/binary-to-text c c-20

# finally check if the generated output differs from the output got from the multicore back-end compiler
diff c-20 data/c-20

