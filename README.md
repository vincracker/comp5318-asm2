# COMP5318/COMP4318 Assignment 2: PathMNIST Image Classification

Classifying 28×28 colour histology patches into 9 tissue types with three models:

| Model | Owner | Status |
|---|---|---|
| SVM (RBF kernel on PCA features) | Member A | Done (test accuracy 0.688, macro F1 0.679) |
| Fully connected neural network (MLP) | Member B | In progress |
| Convolutional neural network (CNN) | Member C | In progress |

Everything lives in one notebook, `notebook.ipynb`, which is the submitted file.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13.

```bash
uv sync                  # creates .venv with every dependency from uv.lock
uv run jupyter lab       # then open notebook.ipynb
```

In VS Code, pick `.venv` as the notebook kernel instead.

## Data

The dataset is **not in the repo**. Copy the four files from the assignment
download into `Assignment2Data/` next to the notebook:

```
code/
├── Assignment2Data/
│   ├── X_train.npy   (32000, 28, 28, 3)
│   ├── y_train.npy   (32000,)
│   ├── X_test.npy    (8000, 28, 28, 3)
│   └── y_test.npy    (8000,)
└── notebook.ipynb
```

`.gitignore` excludes this folder and all `.npy` files. Never commit the data
(it is also not part of the submission).

## Project layout

| Path | Contents |
|---|---|
| `notebook.ipynb` | Exploration, preprocessing, all three models, tuning, final results |
| `figures/` | PNGs saved by the notebook for the report |
| `results/` | Tuning grids (`*_tuning*.csv`) and final test metrics (`final_*.json`) |
| `pyproject.toml`, `uv.lock` | Pinned dependencies |

## Running the notebook

- Run top to bottom. Section 1 must run first: it loads the data and builds
  the shared split used by every model.
- **Hyperparameter searches are slow** (the SVM coarse search took about 29 min).
  Those cells are tagged `tuning` / `tuning-fine` and their outputs are saved in
  the notebook. You can skip them: the plotting cells read the saved CSVs from
  `results/`, and the final-model cells hard-code the best hyperparameters.

## Conventions for B and C

Reuse what Section 1 and 2 already define so all models are compared fairly.

**Data (same stratified 80/20 split, pixels scaled to [0, 1]):**

| Variable | Shape | Use |
|---|---|---|
| `X_tr_img`, `X_val_img`, `X_test_img` | (n, 28, 28, 3) | CNN |
| `X_tr_flat`, `X_val_flat`, `X_test_flat` | (n, 2352) | MLP |
| `y_tr`, `y_val`, `y_te` | (n,) | labels 0–8 |

Tune on `(X_tr_*, y_tr)` vs `(X_val_*, y_val)`. Touch the test set only in
Section 4.

**Helpers:**

- `set_seeds()`: call before building or training any network
- `timed_fit(model, X, y, **fit_kwargs)` and `timed_predict(predict_fn, X)`: record runtimes
- `evaluate_classifier(y_true, y_pred, model_name, train_time, predict_time)`:
  returns accuracy plus macro precision, recall and F1
- `plot_confusion(y_true, y_pred, title, filename)`: row-normalised confusion matrix
- `save_fig(fig, name)`: writes to `figures/`

**Other conventions:**

- Tag slow search cells `tuning` (cell metadata → tags).
- Save tuning results to `results/<model>_tuning.csv` and final metrics to
  `results/final_<model>.json`, matching `final_svm.json`.
- Use `SERIES_BLUE` / `SERIES_ORANGE` for line plots so figures match.

## Deadline

Friday 30 October 2026, 11:59 pm: notebook (`.ipynb`) and report (PDF) on Canvas.
