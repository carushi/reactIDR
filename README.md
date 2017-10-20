# reactIDR
* A novel algorithm to classify each nucleotide into stem, loop, or unmappable regions from high-throughput structure probing data
* Input
	* Bam file
* Output
	* IDR (1-posterior probability of stem or loop class)
* Dataflow
	* bam -> (Docker) -> bed -> (scripts) -> tab file (score or raw read count) -> global and local IDR
* Target high-throughput structure analyses
	* PARS
	* SHAPE-Seq
	* icSHAPE
	* DMS-Seq (assumed to be enriched only at A or C)

## Requirement
* python3
<<<<<<< HEAD
=======
* numpy
>>>>>>> updated
* [idr package](https://github.com/nboley/idr) (if you would like to predict global IDR)
* Setup command (shown below)

```
cd src/
python setup.py  build_ext --inplace
# cython build
```

## Example

* Bam to read count or coverage
	* use docker image [https://hub.docker.com/r/carushi/rt_end_counter/](https://hub.docker.com/r/carushi/rt_end_counter/)
	* find scripts and how to use in [https://github.com/carushi/RT_end_counter](https://github.com/carushi/RT_end_counter)


* Preprocessing
	* You already have ctss.bed and cov.bed from a bam file using docker image.
```
python script/bed_to_pars_format.py --offset -1 --fasta fast.fa ctss.bed
# convert read count bed to tab and count the number of each base at modified sites
python script/bed_to_pars_format.py --offset 0 --fasta fast.fa cov.bed
# convert coverage bed to computation
python src/score_converter.py --merge --output control_ctss sample1.ctss.bed.tab sample2.ctss.bed.tab
# merge replicate data
python src/score_converter.py --score icshape --skip_header --integrated --ouput sample case_ctss.bed.tab control_ctss.bed.tab
# compute icshape score
```

* IDR computation
```
python src/IDR_hmm.py --train --time 10 --core 6 --grid --param sample_param.txt --ref ref.fa --output sample.csv
python src/IDR_hmm.py --test  --time 10 --core 6        --param sample_param.txt --ref ref.fa --output sample.csv
```

## Script
* read_collapse.py
	* collapse PCR duplicates and trim barcode
	* assume gawk
* read_truncate.py
	* extract consistent paired end reads
* bed_to_pars_format.py
	* write PARS-formatted 5' end coverage data based on gtf and gff annotation or sequence location
	* format: NAME <tab> 0;1;2;3;.....
* tab_to_csv.py
	* use to append raw count (read count, coverage, ...) to the output csv file

## TODO
* apply to MaP analyses
