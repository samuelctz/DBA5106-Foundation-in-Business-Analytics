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
- [ ] Pick a **classification** problem with a clear real-world use.
- [ ] Explain **how the predictions are used**. We can define our own setting.
- [ ] Define a **customized evaluation loss** that **deviates from misclassification loss**, based on the context. For example, false negatives and false positives have different costs.

### 2. Models: interpretability vs. performance
- [ ] Implement **at least one glassbox model**. Course options: LASSO logistic regression, shallow tree, EBM, FIGS.
- [ ] Implement **at least one blackbox model**. Course options: random forest, extra trees, gradient boost.

### 3. Training losses: two versions per model
- [ ] Train **each model in two versions**:
  - **L2 loss**
  - **Another surrogate loss**, such as binomial deviance, hinge, or exponential
- [ ] **Tune the hyperparameters separately** for each version.

### 4. Model selection
- [ ] Compare **all** model versions using **validation AUC**, not training loss.
- [ ] Make and justify a decision on the final model.

### 5. Threshold selection
- [ ] For the chosen model, **select the classification threshold that minimizes the evaluation loss** from step 1.

### 6. Calibration
- [ ] Check whether the models are calibrated, using bin evaluation, a reliability diagram, and ECE.
- [ ] If they are not calibrated, adjust the probability predictions, for example with Platt scaling.

### 7. Interpretability with SHAP
- [ ] Apply **Tree SHAP to a tree-based blackbox model**. This is required **even if the final chosen model is glassbox**.
- [ ] Show both **local SHAP** (an individual prediction) and **global SHAP** (overall feature effects).

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

## Team Decisions (to fill in)

| Decision | Choice | Rationale |
|---|---|---|
| Dataset | | |
| Target / classes | | |
| Decision context | | |
| Evaluation loss (cost matrix) | | |
| Glassbox model(s) | | |
| Blackbox model(s) | | |
| Surrogate loss(es) | | |
| Final model | | |
| Chosen threshold | | |
