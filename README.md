## Tibetan Plateau Mixed-Support Water-Budget Closure

This repository contains the code used in a support-explicit study of
mixed-support water-budget closure over the Tibetan Plateau.

## Scope

The code path covers:

- preprocessing and common-domain construction (parameterized for multiple years)
- support-operator construction
- reference and misspecified synthetic-twin generation
- model training and evaluation for synthetic and real-data experiments
- figure and case export for the reported figures and tables

## Repository layout

```text
code/
    build_common_domain_and_masks.py
    preprocess_canonical_3h.py
    build_support_operators.py
    synthetic_twin_generate.py
    synthetic_twin_generate_misspecified.py
    dataset.py
    model.py
    losses.py
    train_real.py                  # training (synthetic and real-data modes)
    baseline_evaluation.py
    reproducibility.py
    real_checkpoint_diagnostics.py
    case_export.py
    case_export_real.py
    paper_figures.py
    paper_fig_e3_real_spatial.py
```

## Data

Third-party input data are not redistributed in this repository. The released
scripts expect the original source data to be placed under the repository root
using the directory names referenced in the code.

For the **2020** warm-season defaults in `build_common_domain_and_masks.py` and
`preprocess_canonical_3h.py`:

- `IMERG_V07B_TP_daily/`
- `Sun_TPW_2020_May_Sep/` (preferred for 2020; see code for fallbacks)
- `CMFD_2020_May_Sep_All/` (preferred for 2020; if missing, the code falls back
  to `CMFD_v2.0/`)

For a **non-2020** warm-season year (for example **2021**), defaults use:

- `IMERG_V07B_TP_<YEAR>_May_Sep_daily/`
- `Sun_TPW/<YEAR>/` (with the same fallback behavior as in the scripts)
- `CMFD_v2.0/`

The Tibetan Plateau TPW product, GPM IMERG precipitation, and the China
Meteorological Forcing Dataset should be obtained from their original public
sources.

### Preprocessed output directories

- **2020:** `build_common_domain_and_masks.py` and `preprocess_canonical_3h.py`
  write by default to `preprocessed/` (under the repository root).
- **Other single warm-season years** (for example 2021): both scripts write by
  default to `preprocessed_<YEAR>/` (for example `preprocessed_2021/`), **not**
  `preprocessed/`.

Downstream scripts default to `preprocessed/` unless you pass an explicit
directory:

- `build_support_operators.py` → `--preproc-dir` (default: `preprocessed`)
- `synthetic_twin_generate.py` → `--preproc-dir` (default: `preprocessed`)
- `train_real.py` (synthetic mode) → `--synthetic-dir` (default:
  `preprocessed/synthetic`)
- `baseline_evaluation.py` → `--synthetic-dir` (default: `preprocessed/synthetic`)

So for a **2021** rerun you must pass `preprocessed_2021` (or your chosen
output dir) consistently on those flags where applicable (see the 2021 bullet
under **Minimal workflow**).

**Real-data training** (`train_real.py --mode real`) loads
`preprocessed/common_domain.nc` and `preprocessed/support_operators.npz` from
the repository root layout above. The release scripts do not add a separate
`--preproc-dir` for real mode; a multi-year real experiment would require
compatible layout changes or merging outputs into `preprocessed/` yourself.

## Environment

Two environment files are provided:

- `environment.yml`: recommended full environment, including optional packages
  used by the map-based figure scripts
- `requirements.txt`: core Python dependencies

If you only need the training and evaluation pipeline, `requirements.txt` is
usually sufficient. For the real-data spatial figure workflow, the optional map
stack (`cartopy`, `geopandas`, `cmocean`) is recommended.

## Minimal workflow

The smallest end-to-end workflow for the **2020** warm season is:

1. Build the common domain and masks:  
   `python code/build_common_domain_and_masks.py`
2. Produce canonical 3-hour tensors:  
   `python code/preprocess_canonical_3h.py`
3. Build the support operators:  
   `python code/build_support_operators.py`
4. Generate synthetic twins for the released tiles:  
   `python code/synthetic_twin_generate.py --tile A_wet`  
   `python code/synthetic_twin_generate.py --tile C_mid`  
   `python code/synthetic_twin_generate.py --tile B_dry`
5. Train the main synthetic configuration (default run name
   `synthetic_<tile>`, so outputs go to `results/synthetic_A_wet` when
   `--tile A_wet` and `--run-name` is omitted):  
   `python code/train_real.py --mode synthetic --tile A_wet --epochs 50 --n-steps 1224 --lambda-W 10 --lambda-Pc 5 --lambda-R 0 --checkpoint-metric val_loss`
6. Evaluate baselines and the trained checkpoint:  
   `python code/baseline_evaluation.py --tile A_wet --n-steps 1224`  
   `python code/baseline_evaluation.py --tile A_wet --n-steps 1224 --model-dir results/synthetic_A_wet`  

Step 6 first line runs IMERG-lift and analytical baselines only; the second
line adds **trained-model** metrics using `best_model.pt` under the given
`results/...` directory (adjust the folder name if you set `--run-name`).

**2021 (or another non-2020 year):** run steps 1–2 with `--year 2021` so outputs
land under `preprocessed_2021/`. Then, for steps 3–5, pass that directory
explicitly, for example:

- `python code/build_support_operators.py --preproc-dir preprocessed_2021`
- `python code/synthetic_twin_generate.py --tile A_wet --preproc-dir preprocessed_2021`  
  (repeat for `C_mid`, `B_dry`)
- `python code/train_real.py ... --synthetic-dir preprocessed_2021/synthetic`  
- `python code/baseline_evaluation.py ... --synthetic-dir preprocessed_2021/synthetic`  
  and `--model-dir` pointing at your new `results/...` run.

Multi-seed or cross-tile studies: call `train_real.py` and
`baseline_evaluation.py` repeatedly with different `--seed`, `--tile`, and/or
`--run-name` values.

### Figure export

1. Create manifest files from the shipped examples and fill in your local
   result paths (from the repository root):  
   `copy case_export_manifest.example.json case_export_manifest.json`  
   `copy paper_figures_manifest.example.json paper_figures_manifest.json`  
   or, on Unix-like systems,  
   `cp case_export_manifest.example.json case_export_manifest.json`  
   `cp paper_figures_manifest.example.json paper_figures_manifest.json`
2. Export figures and figure source products:  
   `python code/case_export.py --export-all --config case_export_manifest.json`  
   `python code/case_export_real.py`  
   `python code/paper_figures.py --manifest paper_figures_manifest.json`  
   `python code/paper_fig_e3_real_spatial.py`

