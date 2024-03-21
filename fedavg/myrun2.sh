#!/bin/bash
dataset=$1
r=$2
l=$3
u=$4
g=$5
f=$6
i=$7
ee=$8

declare -A batches=( [femnist]=20 [celeba]=5 [synthetic]=5 [shakespeare]=20 [sent140]=10 [cifar10]=32)


python main.py -d $dataset -traindir ../leaf/data/${dataset}/data/train/ -testdir ../leaf/data/${dataset}/data/test/ -b ${batches[${dataset}]} -r $r -ee $ee -n 16 -lb $l -up $u -l ../logs/fedprox/guessing/${dataset}/${l}_${u}_g${g}/r${i} -sd 0 -f $f -g $g
