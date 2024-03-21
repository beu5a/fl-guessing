#!/bin/bash
dataset=$1
r=$2
b=$3
l=$4
u=$5
g=$6
fr=$7
runl=$8
runr=$9

LOGROOT=/mnt/nfs/$(whoami)/gel_logs/fednova/guessing_correct

for ((i=$runl; i<=$runr; i++))
	do	        
		python main.py -d $dataset -traindir /mnt/nfs/dhasade/leaf/data/${dataset}/data/train/ -testdir /mnt/nfs/dhasade/leaf/data/${dataset}/data/test/ -b $b -r $r -ee 5 -n 20 -g $g -lb $l -up $u -l ${LOGROOT}/${dataset}/${l}_${u}_g${g}_lr${r}/r${i} -mwf /mnt/nfs/dhasade/FL/models/${dataset}/m${i}.h5 -sd $i -f $fr
	done