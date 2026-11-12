# DBA 5106 – Project 2 (Version 2: With Data)

Classification project for DBA 5106, Overfitting in Validation & Calibration (Long Zhao, NUS DAO).
Our team has chosen **Version 2: With Data**.

---

## Logistics

| Item | Detail |
|---|---|
| Group size | 2 |
| Format | 15-minute live presentation in English, via Zoom |
| Slide limit | **6 slides maximum** |
| Grading | Based **only** on the presentation. Expect many questions. |
| Schedule | Presentation slots (after Quiz 2) will be released in Week 13 |

---

## Required Components

### 1. Problem framing and evaluation loss
- [x] Pick a **classification** problem with a clear real-world use.
- [x] Explain **how the predictions are used**. We can define our own setting.
- [x] Define a **customized evaluation loss** that **deviates from misclassification loss**, based on the context. For example, false negatives and false positives have different costs.

### 2. Models: interpretability vs. performance
- [x] Implement **at least one glassbox model**. Course options: LASSO logistic regression, shallow tree, EBM, FIGS.
- [x] Implement **at least one blackbox model**. Course options: random forest, extra trees, gradient boost.

### 3. Training losses: two versions per model
- [x] Train **each model in two versions**:
  - **L2 loss**
  - **Another surrogate loss**, such as binomial deviance, hinge, or exponential
- [x] **Tune the hyperparameters separately** for each version.

### 4. Model selection
- [x] Compare **all** model versions using **validation AUC**, not training loss.
- [x] Make and justify a decision on the final model.

### 5. Threshold selection
- [x] For the chosen model, **select the classification threshold that minimizes the evaluation loss** from step 1.

### 6. Calibration
- [x] Check whether the models are calibrated, using bin evaluation, a reliability diagram, and ECE.
- [x] If they are not calibrated, adjust the probability predictions, for example with Platt scaling.

### 7. Interpretability with SHAP
- [x] Apply **Tree SHAP to a tree-based blackbox model**. This is required **even if the final chosen model is glassbox**.
- [x] Show both **local SHAP** (an individual prediction) and **global SHAP** (overall feature effects).

---

## Dataset Guidance

**Sources**
- Kaggle: https://www.kaggle.com/. Prefer datasets with **high usability ratings**.
- data.gov.sg: https://beta.data.gov.sg/datasets. Singapore-related datasets are encouraged.

**Choose carefully and avoid:**
- Severe class imbalance
- Very large datasets
- Heavy data cleaning
- Lots of missing data

**Do not copy Kaggle example notebooks.** It will hurt the grade, and it is detectable when teams share similar mistakes or code.

---

## Common Pitfalls

- **Unclear use of predictions.** Always state how the predictions drive a decision. This is what motivates the customized loss.
- **Leaving the threshold at 0.5.** Tune the threshold to minimize the customized evaluation loss.
- **Near-perfect prediction.** If results are almost perfect, consider removing some features to make the task harder.
- **Overfitting the validation set.** Don't over-tune or try too many models. Think more, try less.

---

## Suggested Workflow

1. Choose a dataset and define the decision context and evaluation loss.
2. Split the data into train, validation, and test sets.
3. Build the models: glassbox and blackbox, each with an L2 version and a surrogate-loss version, tuned separately.
4. Compare all versions on validation AUC and choose the final model.
5. Tune the threshold of the chosen model to minimize the evaluation loss.
6. Check calibration and apply Platt scaling if needed.
7. Run Tree SHAP on a tree-based blackbox model, with local and global views.
8. Build the presentation (6 slides or fewer) and prepare for Q&A.

---

## Team Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Dataset | HDB resale flat prices, Jan 2017 to Nov 2026 (data.gov.sg), `resaleprice.csv`. 239,015 rows used. | Singapore data, no missing values, no cleaning beyond parsing lease and storey. |
| Target / classes | `premium` = 1 if the flat sold above the median price of comparable flats (same town, flat type and quarter), else 0. 49% positive. | Balanced by construction, robust to rising prices, and not near-perfectly predictable (a fixed price cutoff gave AUC 0.99). |
| Decision context | A property agency scores each new listing; flagged listings get a senior agent and a paid marketing package. | States how the prediction drives a decision before the price is known. |
| Evaluation loss (cost matrix) | FN = 5, FP = 1, correct = 0. Loss = (5·FN + 1·FP) / n. | A missed premium flat loses commission; a wrongly flagged flat only wastes marketing spend. The 5:1 ratio is our assumption. |
| Glassbox model(s) | LASSO: L2 loss (α = 0.0001) and L1 logistic (C = 0.3). | Sparse, readable coefficients. |
| Blackbox model(s) | Gradient boosting (`HistGradientBoosting`): L2 loss and deviance, both learning rate 0.15, 31 leaves. | Tree-based, so Tree SHAP applies. |
| Surrogate loss(es) | Binomial deviance (log loss), alongside L2. | Tuned separately per version, 4 settings each, on validation AUC. |
| Final model | Gradient boost with deviance loss. Validation AUC 0.907, test AUC 0.896. | Highest validation AUC and best-calibrated raw scores (validation ECE 0.015), so no Platt scaling applied. |
| Chosen threshold | 0.16 (theory: 1/(1+5) = 0.167). | Minimises the evaluation loss on validation. Test cost per listing 0.361, against 0.600 at 0.5. |

## Results at a glance

| Model version | Type | Train AUC | Valid AUC | Test AUC | Test ECE raw | Test ECE after Platt |
|---|---|---|---|---|---|---|
| LASSO, L2 loss | Glassbox | 0.817 | 0.841 | 0.850 | 0.050 | 0.017 |
| LASSO logistic, deviance | Glassbox | 0.820 | 0.844 | 0.852 | 0.027 | 0.016 |
| Gradient boost, L2 loss | Blackbox | 0.922 | 0.906 | 0.895 | 0.018 | 0.030 |
| Gradient boost, deviance (final) | Blackbox | 0.921 | 0.907 | 0.896 | 0.017 | 0.015 |

Split by time: train 2017 to 2023, validation 2024, test 2025 to 2026.

## Reproducing

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python project2_analysis.py
```

The script takes about two minutes and writes the charts to `figures/` and the tables to `results/` (`summary.json` holds every number quoted above).
