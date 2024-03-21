#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 [DATASET] [u]"
    exit
fi

DATASET=$1
u=$2
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
    EXPERIMENT=r${1}u${2}g${3}
    LOGDIR=${DATASET}-${EXPERIMENT}_$(git rev-parse --short HEAD)/$(date +%y%m%d-%H%M%S)
    OUTFILENAME=${DATASET}-${EXPERIMENT}_$(hostname).out
    OUTFILEPATH=$LOGDIR/$OUTFILENAME
    COMMAND=(python3 main.py -d ${DATASET} -l $LOGDIR -traindir ../leaf/data/${DATASET}/data/train/ -testdir ../leaf/data/${DATASET}/data/test/ -r 1 -scf ../samples/clients/${DATASET}/r${1}.txt -b ${batches[${DATASET}]} -lb $2 -up $2 -c fastest -n 20 -ee 3 -g $3
    )
    log_and_execute "$OUTFILEPATH" "${COMMAND[@]}"
}

PERCENTAGES="1.25 1 0.75 0.5 0.25"
for i in $(seq 0 4); do
    for p in $PERCENTAGES; do
        g=$(printf '%.*f\n' 0 $(bc <<< $p*$u))
        build_command_and_run $i $u $g
        build_command_and_run $i $(expr $u + $g) 0
    done
done

