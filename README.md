<img src="https://raw.githubusercontent.com/carushi/reactIDR/master/image/logo.png" width="280">

### reactIDR: evaluation of the statistical reproducibility of high-throughput structural analyses towards a robust RNA structure prediction

* Published in [BMC Bioinformatics](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2645-4)

<img src="https://raw.githubusercontent.com/carushi/reactIDR/master/image/workflow.png" width="500">

**reactIDR** is a Python package that evaluates statistical reproducibility across replicated high-throughput RNA structure profiling data (e.g., PARS, SHAPE-Seq, icSHAPE, DMS-Seq) to robustly infer loop and stem probabilities.

---

## 📥 Input
- Read count data (tabular format)
  - PARS
  - SHAPE-Seq
  - icSHAPE
  - DMS-Seq (assumed to enrich A/C only)

## 📤 Output
- Posterior probability for each site:
  - **Loop** (signal enriched in "case")
  - **Stem** (signal enriched in "control")

## 🧠 Algorithm
- [IDR](https://github.com/nboley/idr) (Irreproducible Discovery Rate)
- Hidden Markov Model

---

## 🔧 Requirements

```
python >= 3.9
numpy >= 2.0.2
scipy >= 1.13.1
pandas >= 2.2.3
```

Optional packages for visualization:

```
seaborn
jupyter notebook
```

## 🚀 Installation
```
pip install reactIDR
```

## ▶️ Getting Started
Test datasets are provided in the example and csv_example directories.
To run a demo using CSV input:
```
git clone https://github.com/carushi/reactIDR
cd reactIDR/csv_example
python -c "import reactIDR; reactIDR.run_reactIDR([
  '-e 0',
  '--csv',
  '--global',
  '--case', 'case.csv',
  '--output', 'test.csv',
  '--param', 'default_parameters.txt'
])"
```

📚 More usage examples and options are available in the [Wiki](https://github.com/carushi/reactIDR/wiki).


## 🛠️ Scripts

| Script                | Description                                                               |
|-----------------------|---------------------------------------------------------------------------|
| `read_collapse.py`    | Collapse PCR duplicates and trim barcodes (assumes `gawk`)                |
| `read_truncate.py`    | Extract consistent paired-end reads                                       |
| `bed_to_pars_format.py` | Convert BED coverage to PARS-style format based on annotations           |
|                         |  format: NAME <tab> 0;1;2;3;..... |
| `tab_to_csv.py`       | Append raw count data to output CSV | 


## 📖 Reference
* R. Kawaguchi, H. Kiryu, J. Iwakiri and J. Sese. ["reactIDR: evaluation of the statistical reproducibility of high-throughput structural analyses towards a robust RNA structure prediction"  BMC Bioinformatics 20 (Suppl 3) :130 (2019)"](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2645-4) ー Selected for APBC '19 proceedings

* [Docker image for reactIDR](https://hub.docker.com/r/carushi/rt_end_counter/)
	* Convert bam to read count data
	* Find scripts and how to use at [https://github.com/carushi/RT_end_counter](https://github.com/carushi/RT_end_counter)

* IDR
	* Li, Qunhua, et al. "Measuring reproducibility of high-throughput experiments", The annals of applied statistics, 2011.
	* [IDR in Python](https://github.com/nboley/idr)
	* [IDR in R](https://cran.r-project.org/web/packages/idr/index.html)


## TODO
* apply to MaP analyses
