# concrete-strength-prediction
Machine learning prediction of concrete compressive strength using Linear Regression and Random Forest.
# Concrete Compressive Strength Prediction Using Machine Learning

Machine learning mini-project comparing **Linear Regression** (baseline) and **Random Forest Regression** for predicting concrete compressive strength (MPa) from mixture proportions and curing age.

## Research question

Can a nonlinear Random Forest model predict concrete compressive strength more accurately than a linear regression baseline, including on **mixtures not seen during training**?

**Inputs (8):** Cement, Blast Furnace Slag, Fly Ash, Water, Superplasticizer, Coarse Aggregate, Fine Aggregate (all in kg/m³), and Age (days).  
**Target:** Concrete compressive strength (MPa).

## Dataset

- **Source:** [UCI Machine Learning Repository — Concrete Compressive Strength (dataset 165)](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength).
- **Original observations:** 1,030; **exact duplicate records removed:** 25; **remaining observations:** 1,005.
- **Missing values:** None in the dataset used.
- Data are downloaded **at runtime** by `ucimlrepo` (no manual dataset download required). Internet access is needed when fetching the dataset.
- Duplicate removal is a conservative preprocessing choice to prevent identical complete rows from appearing in both training and test data. The dataset does not reveal whether all duplicate entries represent data-entry duplication or independent repeated tests.

**Original dataset reference:** Yeh, I.-C. (1998). *Modeling of strength of high-performance concrete using artificial neural networks*. Cement and Concrete Research, 28(12), 1797–1808. https://doi.org/10.1016/S0008-8846(98)00165-3

## Methods

Two scikit-learn regression models are compared with identical input features:

1. **Linear Regression:** ordinary least squares baseline.
2. **Random Forest Regression:** 200 decision trees, `random_state=42` (initial fixed configuration; no systematic hyperparameter search claimed).

Evaluation uses MAE (MPa), RMSE (MPa), and R², with three complementary designs:

- **Random holdout:** 80% training / 20% test, fixed seed `42` (804 train, 201 test).
- **Mixture-grouped holdout:** grouping by all seven mixture-proportion variables (excluding Age), then assigning each group wholly to either training or testing (800 train, 205 test; 0 overlapping groups).
- **5-fold GroupKFold:** validates on mixture groups not present in each corresponding training fold; all 1,005 observations are used across folds. This is **cross-validation**, not a separate untouched final test set.

Permutation feature importance is computed on the mixture-grouped test set using the change in MAE after shuffling each feature. Importance indicates model reliance; it does **not** establish causation.

## Results

The following values are from the course notebook run. Values reproduced in a different Python/scikit-learn environment may vary slightly.

### Random 80/20 holdout

| Model             | MAE (MPa)  | RMSE (MPa) | R²         |
| ----------------- | ----------:| ----------:| ----------:|
| Linear Regression | 8.8953     | 11.1913    | 0.5802     |
| Random Forest     | **3.5329** | **5.2034** | **0.9092** |

### Mixture-grouped 80/20 holdout

| Model             | MAE (MPa)  | RMSE (MPa) | R²         |
| ----------------- | ----------:| ----------:| ----------:|
| Linear Regression | 8.4762     | 10.7238    | 0.6372     |
| Random Forest     | **4.1752** | **6.0067** | **0.8862** |

### Five-fold mixture-grouped cross-validation

| Model             | Mean MAE ± SD (MPa) | Mean RMSE (MPa) | Mean R² ± SD        |
| ----------------- | -------------------:| ---------------:| -------------------:|
| Linear Regression | 8.2417 ± 0.1765     | 10.4033         | 0.5834 ± 0.0571     |
| Random Forest     | **4.4147 ± 0.3023** | **6.1461**      | **0.8534 ± 0.0338** |

Random Forest shows notably lower prediction error than the linear baseline. The lower R² under mixture-grouped evaluation highlights the importance of checking generalization to unseen compositions. Differences between random and grouped results may also reflect differences in test-set difficulty.

Among the studied inputs, **Age** and **Cement** were most important in the permutation analysis. These are predictive associations, not evidence of causal effect.

## Installation

Use Python 3.11+ (Python 3.13 was used in the original Colab session). Create an environment, then install dependencies:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

The requirements specify compatible package ranges, not an exact lockfile. For bit-for-bit software-environment reproduction, record installed versions with `python -m pip freeze`.

## Reproduce training and evaluation

```bash
python train.py
```

The script downloads the UCI dataset, removes exact duplicate rows, performs all three evaluation protocols, calculates permutation importance, and then trains a **final inference model on all 1,005 cleaned observations**.

It produces:

- `concrete_rf_model.joblib` — model + feature order
- `metrics.json` — evaluation results and settings
- `Actual_vs_Predicted.png` — model comparison on random holdout
- `Feature_Importance.png` — permutation importance on grouped holdout
- `permutation_importance.csv` — numerical importance values

The final full-data model was trained **after** evaluation. The reported held-out evaluation scores belong to the respective earlier train/test models, **not** to the final all-data refit.

The companion notebook, `Concrete_Strength_Prediction.ipynb`, documents the interactive data exploration and training process. Colab-specific download cells may require Google Colab and are not necessary for the standalone `train.py` workflow.

## Run inference on example or new input

Once `train.py` has produced the model file:

```bash
python predict.py
# or use a JSON file with all 8 input features:
python predict.py --input example_input.json
```

Example input: Cement 300, Slag 100, Fly Ash 0, Water 180, Superplasticizer 6, Coarse Aggregate 1000, Fine Aggregate 800 (kg/m³); Age 28 days. The original Colab run predicted **41.19 MPa** with the final full-data Random Forest.

The example is a **model prediction**, not a measured strength or proof of engineering suitability. Apply engineering judgment and code-required laboratory testing for real designs.

## Files

```text
README.md
requirements.txt
train.py
predict.py
example_input.json
Concrete_Strength_Prediction.ipynb  # upload your downloaded notebook
concrete_rf_model.joblib             # optional: auto-generated by train.py
Actual_vs_Predicted.png              # optional: auto-generated by train.py
Feature_Importance.png              # optional: auto-generated by train.py
report/                              # final LaTeX report, to be added
```

## Limitations and reproducibility notes

- This is a small public dataset from a particular experimental compilation and may not represent every concrete mixture, curing condition, or construction environment.
- Exact-mixture grouping uses equality of the seven recorded mixture variables. Nominally equivalent mixtures with rounding differences may appear in separate groups.
- Five-fold GroupKFold variation is not a confidence interval. There was no nested hyperparameter search or independent external validation dataset.
- Do not interpret feature importance as causal effect.
- Code uses fixed `random_state=42`; minor numerical variation can occur across versions and platforms.

## Attribution and AI disclosure

- **Data:** UCI Concrete Compressive Strength dataset; original research by Yeh (1998).
- **Software:** Python, NumPy, pandas, scikit-learn, matplotlib, joblib, ucimlrepo.
- **Generative AI:** ChatGPT assisted with explanations, debugging, code organization, documentation drafts, and interpretation of results. The student ran the notebook experiments and supplied the reported numerical outputs. All claims and outputs should be reviewed before submission.
