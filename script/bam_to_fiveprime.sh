#!/bin/bash

#This script was written based on bam_to_ctss.sh in moirai packages
if [ $# -eq 0 ]                  
then
echo Usage is : $0 -q <mapping quality cutoff> <map1.bam> <map2.bam> … 
exit 1;
fi

QCUT=10 

if [ "${QCUT}" = "" ]; then QCUT=10; fi

for var in "$@"
do
if [[ $var =~ sam$ || $var =~ bam$ ]]; then
file=${var##*/}
base=${file%%.*}
option="S"
if [[ $var =~ bam$ ]]
then
option=""
fi

TMPFILE="/tmp/$($file).ctss.txt"
   samtools view  -F 4 -u -q $QCUT -${option}b $var | bamToBed -i stdin | cut -f1-3,6 > $TMPFILE
   cat  ${TMPFILE} \
| awk 'BEGIN{OFS="\t"}{if($6=="+"){print $1,$2,$5}}' \
| sort -k1,1 -k2,2n \
| groupBy -i stdin -g 1,2 -c 3 -o count \
| awk -v x="$base" 'BEGIN{OFS="\t"}{print $1,$2,$2+1,  x  ,$3,"+"}' #>> $var.ctss.bed

cat  ${TMPFILE} \
| awk 'BEGIN{OFS="\t"}{if($6=="-"){print $1,$3,$5}}' \
| sort -k1,1 -k2,2n \
| groupBy -i stdin -g 1,2 -c 3 -o count \
| awk -v x="$base" 'BEGIN{OFS="\t"}{print $1,$2-1,$2, x  ,$3,"-"}' #>> $var.ctss.bed

rm $TMPFILE
fi
done
