# IMDB Text-CNN Data-Scaling Experiment

This repository contains a reproducible experiment on binary sentiment classification with a Text-CNN. The experiment studies how increasing the amount of labelled training data affects both predictive performance and the features highlighted by LIME explanations.

## Research question

Does providing more supervised training examples improve classification accuracy and reduce the use of surface-level features, such as stopwords and punctuation, in local explanations?

## Experimental design

- Dataset: IMDB movie reviews, with 40,000 training, 5,000 validation, and 5,000 test reviews per seed.
- Training fractions: 1%, 5%, 10%, 25%, 50%, and 100% of the training pool.
- Model: Text-CNN with a 20,000-token vocabulary, 100-dimensional embeddings, kernel widths 3, 4, and 5, global max pooling, and dropout 0.5.
- Evaluation: accuracy, macro-F1, precision, and recall on the fixed test set for each seed.
- Explanations: LIME explanations for 100 balanced test reviews, using 1,000 perturbations per review and retaining the five most salient tokens.
- Shortcut measure: the proportion of top-five LIME tokens that are stopwords or punctuation.
- Faithfulness diagnostic: deletion of the LIME top-five token types and measurement of the probability drop and prediction flips.
- Robustness: three random seeds, 42, 123, and 2024.

## Main results

Results below are means and standard deviations over the three seeds.

| Training data | Accuracy | Macro-F1 | LIME shortcut rate |
|---:|---:|---:|---:|
| 1% | 63.4% +/- 6.1% | 62.0% +/- 7.5% | 77.6% +/- 7.5% |
| 5% | 78.4% +/- 1.6% | 78.3% +/- 1.6% | 63.5% +/- 6.0% |
| 10% | 82.3% +/- 0.2% | 82.3% +/- 0.2% | 53.5% +/- 2.4% |
| 25% | 85.9% +/- 0.3% | 85.9% +/- 0.3% | 36.7% +/- 6.6% |
| 50% | 87.6% +/- 0.6% | 87.6% +/- 0.6% | 26.5% +/- 3.4% |
| 100% | 88.8% +/- 0.2% | 88.7% +/- 0.2% | 23.3% +/- 0.6% |

The results show a strong learning curve and a plateau at larger training fractions. More data is also associated with fewer stopwords and punctuation marks among the most salient LIME features. This does not prove human-like understanding: LIME is a local post-hoc approximation and the shortcut measure is intentionally simple.

## Installation

```bash
python3 -m pip install -r requirements.txt
```

The experiment expects an `IMDB Dataset.csv` file with `review` and `sentiment` columns. The dataset is intentionally not included in this repository. Obtain it from an authorized source and place it in the project root before running the experiment.

## Running the experiment

Train and evaluate the six fractions without LIME:

```bash
python3 run_experiment.py --skip-lime --device cpu
```

Generate LIME explanations from existing checkpoints:

```bash
python3 run_experiment.py --lime-only --device cpu
```

Run the deletion test:

```bash
python3 run_deletion_test.py --device cpu
```

Run the complete three-seed experiment and aggregate the results:

```bash
./run_all_seeds.sh
```

The complete run is computationally expensive. Generated checkpoints, per-seed outputs, and figures are excluded by `.gitignore`.

## Repository contents

- `run_experiment.py`: data preparation, Text-CNN training, evaluation, and LIME analysis.
- `run_deletion_test.py`: deletion-based diagnostic for LIME-selected tokens.
- `aggregate_seeds.py`: aggregation of metrics across the three seeds.
- `run_all_seeds.sh`: complete multi-seed pipeline.
- `report.tex`: LaTeX version of the report.
- `requirements.txt`: Python dependencies.

## Limitations

The experiment uses one dataset and a simplified shortcut definition. The deletion test has no random-token baseline, and LIME explanations are not a direct readout of the model's internal representations. The results therefore concern predictive behavior and local attribution patterns, not general language understanding or grounded cognition.
