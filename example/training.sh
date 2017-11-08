#!/usr/bin/bash

DIR="./"
cd $DIR

# When you have bam files -> bed files
# https://github.com/carushi/RT_end_counter
# docker run -it carushi/rt_end_counter /bin/bash
# bash count_and_cov.sh /docker/directory/test.bam

# When you have bed files and fasta file -> tab files
#
# python ../script/bed_to_pars_format.py --offset -1 --fasta temp.fa temp_ctss.bed
# python ../script/bed_to_pars_format.py --offset  0 --fasta temp.fa temp_cov.bed

# You have bed files and gtf file -> tab files
#
# python ../script/bed_to_pars_format.py --offset -1 --gtf temp.gtf temp_ctss.bed
# python ../script/bed_to_pars_format.py --offset  0 --gtf temp.gtf temp_cov.bed


# You have large bed files -> tab files
#
# python ../script/bed_to_pars_format.py --offset -1 --bam temp.bam temp_ctss.bed
# python ../script/bed_to_pars_format.py --offset  0 --bam temp.bam temp_cov.bed

# read count merge
for suffix in "ctss" "cov"
do
python ../src/score_converter.py --merge --output vitro_${suffix} 2_${suffix}.bed.tab,3_${suffix}.bed.tab 0_${suffix}.bed.tab,1_${suffix}.bed.tab
python ../src/score_converter.py --merge --output vivo_${suffix} 4_${suffix}.bed.tab,5_${suffix}.bed.tab 0_${suffix}.bed.tab,1_${suffix}.bed.tab
done

# score computation
for cond in "vitro" "vivo"
do
python ../src/score_converter.py --score icshape --print_all --dir ./ --coverage ${cond}_cov_case.tab ${cond}_cov_cont.tab --skip_header --output ${cond} ${cond}_ctss_case.tab ${cond}_ctss_cont.tab
python ../src/score_converter.py --score icshape --print_all --integrated --dir ./ --coverage ${cond}_cov_case.tab ${cond}_cov_cont.tab --skip_header --output ${cond} ${cond}_ctss_case.tab ${cond}_ctss_cont.tab
done

#training
PATTERN="train"
for cond in "vitro" "vivo"
do
    python ../src/IDR_hmm.py --idr --case ${cond}_case.tab --cont ${cond}_cont.tab --time 10 --core 5 --param ${cond}_train.param.txt --output ${cond}.csv --ref rRNA_with_minus.fa --${PATTERN} > ${PATTERN}_${cond}.out.txt
done

PATTERN="test"
for cond in "vitro" "vivo"
do
    python ../src/IDR_hmm.py --idr --case ${cond}_case.tab --cont ${cond}_cont.tab --core 5 --param ${cond}_train.param.txt --output ${cond}.csv --ref rRNA_with_minus.fa --${PATTERN} > ${PATTERN}_${cond}.out.txt
done

PATTERN="global"
for cond in "vitro" "vivo"
do
    python ../src/IDR_hmm.py --idr --case ${cond}_case.tab --cont ${cond}_cont.tab --core 5 --param default_parameters.txt --output ${cond}.csv --ref rRNA_with_minus.fa --${PATTERN} > ${PATTERN}_${cond}.out.txt
done


# evaluation

for cond in "vitro" "vivo"
do
    cat test_${cond}.csv > test_${cond}_all.csv
    cat noHMM_${cond}.csv >> test_${cond}_all.csv
    python ../src/evaluate_IDR_csv.py --auc --score ${cond}_icshape_integ.tab -- test_${cond}_all.csv > AUC_test_${cond}.txt
    python ../src/evaluate_IDR_csv.py --parameter train_${cond}.out.txt
done
python ../src/plot_bargraph.py --header --window 1 --idr test_vivo_all.csv test_vitro_all.csv