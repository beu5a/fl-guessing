#!/bin/bash
LOGROOT=/mnt/nfs/$(whoami)/gel_logs/fedavg/guessing/
dataset=$1
r=$2
b=$3
l=$4
u=$5
g=$6
fr=$7
runl=$8
runr=$9

if [ $# -ne 9 ]; then
    echo "Usage: $0 [DATASET] [r] [b] [l] [u] [g] [fr] [runl] [runr]"
    exit
fi

DATASETS="femnist synthetic celeba shakespeare sent140"
if [[ ! "$DATASETS" =~ (^|[[:space:]])$dataset($|[[:space:]]) ]]; then
    echo "Dataset must be in {$DATASETS}"
    exit
fi

log_and_execute () {
    local outfile=$1
    shift
    local toexec=("$@")
    OUTDIR=$(dirname $outfile)
    mkdir -p $OUTDIR 
    echo -e "date and time: $(date +%d.%m.%y-%H:%M:%S)\nhostname: $(hostname)\ngit commit: $(git rev-parse --short HEAD)\ncommand: ${COMMAND[@]}" > $OUTDIR/metadata.txt
    { time "${toexec[@]}" ; } &> $outfile
}

for ((i=$runl; i<=$runr; i++))
do            
    LOGDIR=${LOGROOT}/${dataset}/${l}_${u}_g${g}_${r}/r${i}
    OUTFILEPATH=$LOGDIR/output.txt
    COMMAND=(python main.py -d $dataset -traindir ../leaf/data/${dataset}/data/train/ -testdir ../leaf/data/${dataset}/data/test/ -b $b -r $r -ee 5 -n 20 -g $g -lb $l -up $u -l ${LOGDIR} -mwf ../samples/models/${dataset}/m${i}.h5 -sd $i -f $fr)
    log_and_execute "$OUTFILEPATH" "${COMMAND[@]}"
done
