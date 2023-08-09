#!/bin/sh

NUM_TREES=$1
NUM_CONCURRENT=2 # constant

readarray -d ',' -t RECOS < $2
IFS=',' read -ra PARAMS <<< $3

OUTFOLDER=$4
PREFIX=$5
SLIM_FILE=$6
SELECTION_STRENGTH=$7

function run_slim(){
    nohup slim -d reco=${RECOS[i]} -d N1=${PARAMS[0]} -d N2=${PARAMS[1]} \
        -d growth=${PARAMS[2]} -d T1=${PARAMS[3]} -d T2=${PARAMS[4]} \
        -d outfolder=\"$OUTFOLDER\" -d mut=$SELECTION_STRENGTH \
        -d prefix=\"$PREFIX\" -s $i -d i=$i $SLIM_FILE > $OUTFOLDER$i.out & }

i=0
while [ $i -lt $NUM_TREES ]
do
    for (( j=0; j<$NUM_CONCURRENT; j++ ))
    do
        run_slim
        i=$((i+1))
    done
    wait
done
