# GlassBox Stacked NIDS

A stacked-ensemble Network Intrusion Detection System (NIDS) that pairs strong
gradient-boosted/tree base learners with a glass-box Explainable Boosting
Machine (EBM), and cross-checks SHAP vs. EBM explanations for agreement.
Evaluated on **CICIDS-2017** and validated cross-dataset on **UNSW-NB15**.

The stacking ensemble combines **CatBoost**, **ExtraTrees**, and an
**Explainable Boosting Machine (EBM)** base learners under a meta-learner,
with per-class decision-threshold calibration and SMOTE/ADASYN minority-class
remediation. Model explanations are evaluated for SHAP/EBM agreement
(Spearman/Kendall correlation, top-k feature/rank overlap) rather than taken
on faith — the "glass box" half of the name.

## Results summary

| Dataset | Model | Accuracy | F1 (weighted) | MCC |
|---|---|---|---|---|
| CICIDS-2017 | Baseline Ensemble | 0.8788 | 0.8899 | 0.6585 |
| CICIDS-2017 | **Stacking + Thresholds** | **0.9984** | **0.9984** | **0.9953** |
| UNSW-NB15 | Baseline Ensemble | 0.5476 | 0.5983 | 0.4813 |
| UNSW-NB15 | **Stacking + Thresholds** | **0.8567** | **0.8591** | **0.8172** |

Full component-by-component ablations, per-class threshold calibration,
minority-class (SMOTE/ADASYN) remediation, and SHAP-vs-EBM explanation
agreement are in [`results/`](results/) and visualized in
[`figures/`](figures/). Methodology and discussion are in
[`manuscript/`](manuscript/).

## Repository structure

```
glassbox-stacked-nids/
├── notebooks/
│   ├── NIDS_Upgraded_v4.ipynb          # CICIDS-2017 main pipeline
│   └── NIDS_UNSW_NB15_Native.ipynb     # UNSW-NB15 cross-dataset validation
├── data/                                # download instructions (no raw CSVs)
├── figures/                             # all figures saved by the notebooks
├── results/                             # ablation / calibration / agreement tables (CSV)
│   ├── Table_leakage_control_infold_FS*.csv   # in-fold feature selection leakage control
│   └── Table_SOTA_comparison_*.csv            # SOTA comparison (verified from source PDFs)
└── manuscript/
    ├── Glass-Box NIDS (JNCA).docx                          # main manuscript
    ├── Glass-Box NIDS - Highlights (JNCA).docx
    ├── Glass-Box NIDS - Cover Letter (JNCA).docx
    ├── Glass-Box NIDS - Graphical Abstract (JNCA).docx
    └── Glass-Box NIDS - Declaration of Competing Interest (JNCA).docx
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download the raw datasets per [`data/README.md`](data/README.md), then run
the notebooks in `notebooks/` top to bottom. Each notebook saves its own
figures into `figures/` and its own tables into `results/`.

## License

MIT — see [LICENSE](LICENSE).
