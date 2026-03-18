## Transcriptomic signature for TB diagnosis in HIV-coinfected children
A minimal blood-based transcriptomic signature for diagnosing active tuberculosis in HIV-infected populations. Developed using data from Uganda and validated across two independent cohorts Southern Africa (Botswana and Eswatini) and India.

### Scripts
- discovery_pipeline.py : Main RFE pipeline
- validate_signature.py : performs validation in African and external cohort
- complete_covariate_comparison.py : compares performance of different features - transcripts, covariates and combination


### Dependencies
- Python 3.8+
- scikit-learn >= 1.0.0
- pandas >= 1.3.0
- numpy >= 1.21.0
- scipy >= 1.7.0

## Usage

### 1. Discovery and Validation
```bash
python scripts/discovery_pipeline.py
```

### 2. Covariate Comparison
```bash
python scripts/complete_covariate_comparison.py
```

## Methods

**Feature Selection:** Recursive Feature Elimination (RFE) with Random Forest classifier

**Model Parameters:**
- n_estimators: 500
- max_depth: 5
- min_samples_split: 10
- min_samples_leaf: 5
- random_state: 42

**Validation:** 10-fold stratified cross-validation (discovery), independent cohort validation

## Data Availability

Transcriptomic data will be deposited to GEO upon manuscript acceptance.Transcript count files are available at https://doi.org/10.5281/zenodo.17521611. The external validation data can be accessed Indian from the Gene Expression Omnibus (GEO) under accession GSE162164.
