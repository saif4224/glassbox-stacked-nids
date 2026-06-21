# Data

The notebooks expect raw CSVs in this directory. They are **not** included in
this repository (CICIDS-2017 alone is ~900 MB) — download them from the
original sources below and place them as described.

## CICIDS-2017 (`notebooks/NIDS_Upgraded_v4.ipynb`)

1. Download the CSV version of CICIDS-2017 from the Canadian Institute for
   Cybersecurity: https://www.unb.ca/cic/datasets/ids-2017.html
2. The notebook reads the 8 daily capture CSVs merged into a single file.
   Concatenate the per-day CSVs (`Monday-WorkingHours.pcap_ISCX.csv`,
   `Tuesday-WorkingHours.pcap_ISCX.csv`, ... `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`)
   into one file and save it as:
   ```
   data/CICIDS-2017.csv
   ```
3. Update the `pd.read_csv(...)` path in the notebook's data-loading cell if
   you place the file elsewhere.

## UNSW-NB15 (`notebooks/NIDS_UNSW_NB15_Native.ipynb`)

1. Download UNSW-NB15 from the UNSW Canberra Cyber Range Lab:
   https://research.unsw.edu.au/projects/unsw-nb15-dataset
2. The notebook auto-discovers any CSV matching `*NB15*.csv` / `*nb15*.csv` in
   `DATA_DIR` (set at the top of the notebook, default `'./'`) and
   concatenates them. The two files used in this study are the official
   train/test partition:
   ```
   data/UNSW_NB15_training-set.csv
   data/UNSW_NB15_testing-set.csv
   ```
3. Set `DATA_DIR = 'data/'` (or wherever you place the files) in the
   notebook's setup cell before running.

## Notes

- Both notebooks are deterministic given the same raw data and the fixed
  random seeds set in their setup cells — re-running them reproduces the
  tables in `results/` and the figures in `figures/`.
- Saved model artifacts (scalers, label encoders, base/meta learners,
  per-class thresholds) are written by the notebooks via `joblib`/`pickle`
  but are not checked into this repo; re-run the notebooks to regenerate
  them.
