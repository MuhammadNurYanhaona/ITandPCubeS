#!/bin/bash

matrix_size=$1
echo $matrix_size
block_size=$2
echo $block_size

# first convert two input text files into binary files
../../../tools/text-to-binary data/a-$matrix_size a
../../../tools/text-to-binary data/b-$matrix_size b

# then create an input output configuration file for the program
echo a >  input
echo b >> input
echo $block_size >> input
echo $block_size >> input
echo c >> input
echo a-out >> input
echo b-out >> input

# then run the mm-multiply program for all segments
for segment in {0..3}
do
	./crazy-mmm-on-laptop $segment < input
done

# convert the output file from binary to text form to output validation
../../../tools/binary-to-text c c-$matrix_size

# finally check if the generated output differs from the output got from the multicore back-end compiler
echo "comparing generated c with reference"
diff c-$matrix_size data/c-$matrix_size

# do a validation of input file reading
echo "comparing generated a with reference"
diff a-out a
echo "comparing generated b with reference"
diff b-out b

