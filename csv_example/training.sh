#!/usr/bin/bash
set -poe pipefail
DIR="./"
cd $DIR

PATTERN="global"
for cond in "case"
do
    echo writing to ${PATTERN}_${cond}.csv >&2
    reactIDR --csv  -e 0 --idr --case ${cond}.csv --core 5 --param default_parameters.txt --output output_${cond}.csv --${PATTERN} > ${PATTERN}_${cond}.out.txt
done


