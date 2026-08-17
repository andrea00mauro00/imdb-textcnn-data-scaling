# IMDB Text-CNN Data-Scaling Experiment

## Run

Install the dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run training, test evaluation, and checkpoint creation:

```bash
python3 run_experiment.py --skip-lime
```

Reuse the checkpoints and generate the fixed-sample LIME analysis:

```bash
python3 run_experiment.py --lime-only
```

Run the minimal deletion test on the existing checkpoints and LIME outputs:

```bash
python3 run_deletion_test.py
```

The complete run can also be performed in one command:

```bash
python3 run_experiment.py
```

## Fixed protocol

- Dataset: `IMDB Dataset.csv`.
- Stratified split: 40,000 training, 5,000 validation, and 5,000 test reviews.
- Training fractions: 1%, 5%, 10%, 25%, 50%, and 100%.
- Every training subset is balanced and nested in the next larger subset.
- The vocabulary is built once from the training pool and is shared across models; its 20,000-entry limit includes padding and unknown tokens.
- The architecture and hyperparameters are identical across fractions.
- LIME explains the same 100 balanced test reviews for every model.
- Each explanation stores the top five tokens ranked by absolute LIME weight.
- Shortcut tokens are stopwords or tokens consisting only of punctuation.

## Main outputs

- `outputs/metrics.csv`: accuracy, macro-F1, precision, recall, and training metadata.
- `outputs/lime_explanations.csv`: one row per top-five LIME token.
- `outputs/lime_summary.csv`: stopword, punctuation, and combined shortcut rates.
- `outputs/deletion_results.csv`: per-example probability change after deleting the LIME top-5 tokens.
- `outputs/deletion_summary.csv`: aggregate probability drop and prediction-flip rates.
- `outputs/lime_examples.csv`: fixed test examples used by LIME.
- `outputs/splits.csv`: row IDs and labels for the fixed data split.
- `outputs/models/`: one checkpoint for each training fraction.
- `outputs/figures/learning_curve.png`: accuracy and macro-F1 curves.
- `outputs/figures/lime_shortcuts_curve.png`: shortcut-rate curves.
- `outputs/figures/accuracy_vs_shortcuts.png`: combined accuracy and shortcut-rate plot.
- `outputs/figures/deletion_probability_drop.png`: deletion-test probability-drop curve.

The current run uses seed 42 and the Apple MPS device when available. For a CPU run, use
`--device cpu`.

## Multi-seed robustness check

Run the full pipeline (training + LIME + deletion test) for seeds 123 and 2024,
then aggregate all three seeds (42 is reused from `outputs/`):

```bash
./run_all_seeds.sh
```

Per-seed results go to `outputs_seed123/` and `outputs_seed2024/`. The
aggregation (mean ± std tables and figures with error bars) is written by
`aggregate_seeds.py` to `outputs_multiseed/`, including
`aggregate_report.md` with a report-ready summary table.
