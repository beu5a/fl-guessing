#!/bin/bash
ICPREFIX=iccluster
SACSPREFIX=sacs
LABOSPREFIX=labostrex
ICMACHINES=(103 108)
SACSMACHINES=(001 002 003 004 005)
LABOSMACHINES=(109 110 112 113 114 115 116 132 118 119 120 133 117 122 131 123 121)
ALLMACHINES="${SACSMACHINES[@]/#/$SACSPREFIX} ${LABOSMACHINES[@]/#/$LABOSPREFIX} ${ICMACHINES[@]/#/$ICPREFIX} labosgodzilla"

for host in $ALLMACHINES; do
    echo $host
    RESFILES=$(ssh rafaelpp@$host find gel/efficient-federated-learning/grad_guessing -name test.csv)
    for file in $RESFILES; do
        echo ">>> $file"
        ssh rafaelpp@$host tail -1 $file
    done
done

