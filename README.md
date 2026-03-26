## MOMENTA (`main/`) Reproducibility Guide

This directory contains the maintained implementation used to train and evaluate MOMENTA for multimodal misinformation detection.

The pipeline includes:
- modality-specific Mixture-of-Experts (MoE),
- bidirectional co-attention + discrepancy-aware fusion,
- temporal aggregation with drift and momentum,
- optional timestamp-aware Transformer over temporal windows,
- multi-term optimization (classification, alignment, contrastive, match, domain-adversarial, R-Drop, prototype, and prototype-memory terms).

This README is written for paper submission and reproducibility.

MOMENTA is designed for robust multimodal fake-news detection across heterogeneous sources. The model combines text-image semantic alignment with explicit inconsistency modeling, and then adds temporal reasoning to capture narrative evolution over time. Beyond in-domain accuracy, the training objective emphasizes cross-dataset stability through domain-adversarial learning and prototype-based alignment, which is important when transferring between platforms with different language style, image distributions, and class balance.

## 1) Project Layout

```text
main/
  README.md
  requirements.txt
  src/
    data/
      dataset.py
    models/
      momenta.py
      temporal.py
      loss.py
    scripts/
      train.py
      loso_eval.py
      calibration.py
      viz_tsne.py
  results/
    README.md
    generate_paper_figures.py
    figures_for_paper.tex
    figures_for_paper_extended.tex
    sota_tables.tex
    results_sota.json
    metrics/results_v2.json
    logs/lambda_logs/*.log
```

## 2) Environment Setup

Run from `main/`:

```bash
pip install -r requirements.txt
```

Recommended:
- Python 3.10+
- CUDA-enabled PyTorch for training speed

## 3) Data and Cache Expectations

The dataset loader (`src/data/dataset.py`) resolves paths relative to the project root.

Expected dataset roots:
- `datasets/Fakeddit-master/data/multimodal_only_samples/`
- `datasets/MMCoVaR-main/`
- `datasets/Weibo-dataset-main/`
- `datasets/XFacta-main/data/`

Expected cache roots:
- image cache: `deployment_cache/images/`
- model cache: `deployment_cache/models/`

Notes:
- Samples with missing cached images are dropped at load time.
- For Fakeddit, optional train-only subsampling is supported via `--fakeddit_train_frac`.

## 4) Core Training Command

Baseline multi-dataset training:

```bash
python -m src.scripts.train \
  --datasets fakeddit mmcovar weibo xfacta \
  --equalize --batches_per_ds 150
```

Fakeddit train-only subset regime (val/test unchanged):

```bash
python -m src.scripts.train \
  --datasets fakeddit mmcovar weibo xfacta \
  --fakeddit_train_frac 0.1 \
  --equalize --batches_per_ds 150
```

Outputs:
- checkpoints are saved under `checkpoints/` (default from `--out_dir`)
- best model name pattern: `best_<run_tag>.pt` (default: `best_main.pt`)

## 5) Evaluation and Analysis Commands

LOSO evaluation:

```bash
python -m src.scripts.loso_eval --gpu 0
```

Calibration analysis (MC-dropout reliability):

```bash
python -m src.scripts.calibration \
  --checkpoint checkpoints/best_main.pt \
  --datasets fakeddit mmcovar weibo xfacta \
  --mc_passes 30 \
  --gpu 0
```

t-SNE embedding visualization:

```bash
python -m src.scripts.viz_tsne \
  --checkpoint checkpoints/best_main.pt \
  --out_dir results \
  --gpu 0
```

Paper figure generation from JSON artifacts:

```bash
python results/generate_paper_figures.py
```

## 6) Default Hyperparameters in `train.py`

Key defaults currently used by `src/scripts/train.py`:

- model:
  - `d=256`, `K=4`, `n_heads=4`
  - `window_size=8`, `window_stride=4`
- optimization:
  - `epochs=20`, `lr=3e-5`, `backbone_lr=3e-6`
  - `weight_decay=1e-2`, `grad_accum=2`, `warmup_ratio=0.1`
- loss weights:
  - `lambda_align=0.05`
  - `lambda_tc=0.01`
  - `lambda_match=0.1`
  - `lambda_contrast=0.05`
  - `lambda_domain=0.03`
  - `lambda_proto=0.05`
  - `lambda_proto_memory=0.10`
  - `lambda_rdrop=0.5` (R-Drop enabled by default; disable with `--no_rdrop`)
  - `lambda_tc_lstm=0.01`
  - `lambda_reg=0.0`
- loss shape parameters:
  - `tau=0.2`, `gamma=0.9`
  - `focal_gamma=2.5`
  - `label_smoothing=0.05`

## 7) Paper-Alignment Notes

The implementation in `main/` is aligned with the current methodology details on:
- temporal window labeling and overlapping-window inference aggregation,
- focal classification factor form,
- image projection path in the multimodal encoder,
- default values for the compact hyperparameter block.

One explicit caveat remains:
- the total objective includes an explicit `lambda_reg * ||Theta||^2` term, but the compact hyperparameter block does not specify a numeric `lambda_reg`.
- current defaults are `--lambda_reg=0.0` and AdamW `weight_decay=1e-2`.

## 8) Result Artifacts for Manuscript

Prepared artifacts are stored in `results/`:
- tables: `sota_tables.tex`
- figure snippets for inclusion:
  - `figures_for_paper.tex`
  - `figures_for_paper_extended.tex`
- metrics JSON:
  - `metrics/results_v2.json`
  - `results_sota.json`
- run logs:
  - `logs/lambda_logs/*.log`

## 9) Practical Checks Before Running

- verify dataset directories exist and are populated
- verify `deployment_cache/images/` contains resolved image files
- verify model cache paths or internet/model access for backbone loading
- verify output directory permissions for `checkpoints/` and `results/`

## Authors

- Yeganeh Abdollahinejad
- Ahmad Mousavi
- Naeemul Hassan
- Kai Shu
- Nathalie Japkowicz
- Shahriar Khosravi
- Amir Karami

## License

This project is released under the MIT License.

