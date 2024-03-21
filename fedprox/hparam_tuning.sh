#!/bin/bash

if [ $# -ne 6 ]; then
    echo "Usage: $0 [DATASET] [lb] [up] [algo] [lr] [experiment number]"
    exit
fi

DATASET=$1
lb=$2
u=$3
algo=$4
lr=$5
i=$6

declare -A batches=( [femnist]=20 [celeba]=5 [synthetic]=5 [shakespeare]=20 [sent140]=10 )

DATASETS="femnist synthetic celeba shakespeare sent140"
if [[ ! "$DATASETS" =~ (^|[[:space:]])$DATASET($|[[:space:]]) ]]; then
    echo "Dataset must be in {$DATASETS}"
    exit
fi

log_and_execute () {
    local outfile=$1
    shift
    local toexec=("$@")
    mkdir -p $(dirname $outfile)
    echo "${toexec[@]}" > $outfile
    { time "${toexec[@]}" ; } &>> $outfile
}

build_command_and_run () {
    LOGDIR=../logs/hparam_tuning/${algo}/lr_tuning/${DATASET}/e${i}
    OUTFILENAME=${DATASET}-$(hostname).out
    OUTFILEPATH=$LOGDIR/$OUTFILENAME
    COMMAND=(python3 hparam_tuning.py -d ${DATASET} -l $LOGDIR -traindir ../leaf/data/${DATASET}/data/train/ -testdir ../leaf/data/${DATASET}/data/test/ -r $lr  -b ${batches[${DATASET}]} -lb $lb -up $u -n 20 -ee 3 -g 0 -sd 1
    )
    log_and_execute "$OUTFILEPATH" "${COMMAND[@]}"
}


build_command_and_run
        
