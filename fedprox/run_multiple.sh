#!/bin/bash
dataset=$1
r=$2
l=$3
u=$4
g=$5
mu=$6
f=$7
i=$8
ee=$9


declare -A batches=( [femnist]=20 [celeba]=5 [synthetic]=5 [shakespeare]=20 [sent140]=10 )

for ((j = 0; j <= $i; j++)); do
    python main.py -d $dataset -traindir ../leaf/data/${dataset}/data/train/ -testdir ../leaf/data/${dataset}/data/test/ -b ${batches[${dataset}]} -r $r -mu $mu -ee $ee -n 20 -lb $l -up $u -l ../logs/fedprox/guessing/${dataset}/${r}/${l}_${u}_g${g}/r${j} -mwf ../samples/models/${dataset}/m${j}.h5 -sd $j -f $f -g $g
done