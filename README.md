# reactIDR
* Pipeline to compare high-throughput structure probing dataset


## Script
* read_collapse.py
	* collpase PCR duplicates and trim barcode
	* assumes gawk
* read_truncate.py
	* extract consistent paired end reads
* bam_to_fiveprime.sh
	* write bed file of 5' end coverage
	* depends on samtools and bedtools
	* assumes gawk
* bam_to_coverage.sh
	* depends on samtools and bedtools
	* assumes gawk
* bed_to_pars_format.py
	* write PARS-formatted 5' end coverage data based on gtf and gff annotation or sequence location
	* format: NAME <tab> 0;1;2;3;.....
