# Rosaloid: Protein Optimization with ESM-2 & NGBoost

Rosaloid is a Bayesian Optimization pipeline for protein sequence engineering. It leverages **ESM-2** pre-trained language model embeddings and **NGBoost** for probabilistic surrogate modeling to efficiently navigate the protein fitness landscape.

## Key Features

- **State-of-the-Art Embeddings**: Uses Meta's ESM-2 (650M parameter) model to represent protein sequences.
- **Probabilistic Modeling**: NGBoost provides uncertainty estimates (natural gradients) crucial for the acquisition function.
- **Bayesian Optimization**: Implements Expected Improvement (EI) with diverse batch selection (Hamming distance penalty).
- **Efficiency**: Caches embeddings to disk to speed up multi-round experiments.

## Repository Structure

```
Rosaloid/
├─ src/             # Core library
│  ├─ embeddings.py # ESM-2 wrapper
│  ├─ surrogate.py  # NGBoost wrapper
│  ├─ bo_loop.py    # Main optimization loop
│  └─ ...
├─ notebooks/       # Interactive demo
│  └─ demo.ipynb    # Run the pipeline in Jupyter
├─ scripts/         # CLI tools
│  ├─ run_bo.py     # Command-line entry point
│  └─ analyze_results.py
├─ data/            # Datasets (DMS substitutions)
└─ results/         # Experiment outputs & plots
```

## Quickstart

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Run Demo (CLI)

Run a 5-round optimization campaign on Green Fluorescent Protein (GFP):

```bash
python scripts/run_bo.py --dms_id GFP_AEQVI_Sarkisyan_2016 --rounds 5 --device cuda
```

### 3. Run Demo (Notebook)

Open `notebooks/demo.ipynb` and execute the cells to visualize the optimization process interactively.

## Method Overview

1. **Seed**: Start with a small set of labeled sequences (e.g., N=96).
2. **Embed**: Convert sequences to fixed-size vectors using ESM-2 (layer 33).
3. **Train**: Fit NGBoost to predict fitness ($y$) from embeddings ($X$).
4. **Propose**: Score unlabeled pool with Expected Improvement (EI) and select a diverse batch.
5. **Evaluate**: Obtain ground truth labels (simulated via look-up).
6. **Repeat**: Add new data to training set and iterate.

## Contributors

 - [Theerapas Apinankul](https://github.com/theerapas)
 - [Thanakrit Weeraphatiwat](https://github.com/Champy2005)
 - [Keeratikorn Samutrnawin](https://github.com/yoksamutr)

## Future Work

- Tune acquisition β + normalization to reduce late-round overfitting
- Try fine-tuning ESM on related sequences (instead of fixed feature extractor)
- Expand to more multi-mutation datasets and add mechanistic interpretation
- Wet-lab validation hooks.
