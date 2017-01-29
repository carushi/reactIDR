#!/bin/bash
if [ $# -eq 0 ]
then
echo "Usage is : $0 -q <mapping quality cutoff> <mapped.bam> "
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

TMPFILE="${file}_coverage.bam.tmp"
samtools view  -F 4 -q $QCUT -${option} ${var} | grep -v "XS:" | awk 'BEGIN{OFS='\t'}{if(NF == 3 || length($10) >= 15) { print }}' \
# for rRNA or repeat sequences
# samtools view  -F 4 -${option}h $var  | awk 'BEGIN{OFS='\t'}{if(NF == 3 || length($10) >= 15) { print }}' \
 | samtools view -Shb - > $TMPFILE
bedtools genomecov -bga -strand + -ibam $TMPFILE | awk '{printf("%s\t+\n", $0)}'
bedtools genomecov -bga -strand - -ibam $TMPFILE | awk '{printf("%s\t-\n", $0)}'

rm $TMPFILE

fi
done

