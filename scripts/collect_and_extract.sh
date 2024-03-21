#!/bin/bash
TMPFILE=/tmp/out
./collect.sh > $TMPFILE
python3 extract.py $TMPFILE | sort > leaf_experiments.csv
#rm $TMPFILE
