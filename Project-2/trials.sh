#! /usr/bin/env bash

# Number of samples for sdf.out trials
SAMPLES=1000000000

# Data file for mean.out and threaded.out
FILENAME=data.bin

# Configure the script to exit when Control-C is pressed
trap "exit" INT

# Helper function that takes (in parameter order):
#
#  * the test id (which is usually the number of threads)
#  * the command to run
#  * any command-line arguments for the command
#
run() {
    printf "% 3d: " $1
    /usr/bin/time -f "%e real\t%U user\t%S sys\t%M memory (KB)" $2 ${@:3} > /dev/null
}

# Time the single-threaded version of the program.  Uncomment the line for
#   the program you want to run
run 1 mean.out $FILENAME
# run 1 sdf.out -n $SAMPLES 

for i in {2..84..2} ; do
    run $i threaded.out -t $i -f $FILENAME
    # run $i sdf.out -t $i -n $SAMPLES
done
