#!/bin/bash

# the program need only one input, the output file name
echo output >> input

# then run the reordering program for all segments
for segment in {0..3}
do
	./reorder-on-laptop $segment < input
done

# convert the output file from binary to text form to output validation
../../../tools/binary-to-text output output.txt

