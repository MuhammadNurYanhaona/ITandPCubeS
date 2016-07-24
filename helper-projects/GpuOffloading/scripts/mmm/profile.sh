#!/bin/sh
#SBATCH --nodes 1
#SBATCH --tasks 1
#SBATCH --time=04:00:00
#SBATCH --output=output
#SBATCH --gres=gpu
#SBATCH --reservation=yan
#SBATCH --nodelist=artemis5

executable=../../gpu_offload.o
matrix_size=$1
block_size=$2
batch_size=$3

# to generate a log file for the summary mode profile
#nvprof --log-file timing-profile.log $executable $matrix_size $block_size $batch_size

# to generate a log file for the detail event mode profile
nvprof  --events all --metrics all --log-file event-profile.log $executable $matrix_size $block_size $batch_size

# to export the profiling output to a binary profile file for later viewing using the visual profiler
#nvprof --profile-from-start off --print-gpu-trace --print-api-trace --export-profile timing-profile.exp $executable $matrix_size $block_size $batch_size
#nvprof  --profile-from-start off --events all --metrics all --export-profile event-profile.exp $executable $matrix_size $block_size $batch_size

# to profile a code that has explicit profile start and end boundaries set up inside the code
#nvprof --profile-from-start off --print-gpu-trace --print-api-trace --log-file timing-profile.log $executable $matrix_size $block_size $batch_size
#nvprof  --profile-from-start off --events all --metrics all --log-file event-profile.log $executable $matrix_size $block_size $batch_size
