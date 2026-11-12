"""DBA 5106 Project 2 (Version 2: With Data) - HDB resale "premium listing" classifier.

Run:  .venv/bin/python project2_analysis.py
Outputs: figures/*.png and results/*.csv, results/summary.json

Decision context
----------------
A property agency triages every newly listed HDB resale flat. Listings predicted to be
"premium" (will transact above the median price of comparable flats: same town, same
flat type, same quarter) are routed to a senior agent with a paid marketing package.
  - False negative: a premium flat gets standard handling -> undersold / lost commission. Cost 5.
  - False positive: marketing budget and senior-agent time spent on an ordinary flat.   Cost 1.
Evaluation loss = (5 * FN + 1 * FP) / n   (average cost per listing).
"""
import json
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
SEED = 5106
C_FN, C_FP = 5.0, 1.0
ECE_OK = 0.02   # validation ECE at or below this counts as "calibrated": no adjustment
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10})
BLUE, ORANGE, GREY = "#2a6fb0", "#d9731a", "#8a8f98"

# --------------------------------------------------------------------------- 1. data
df = pd.read_csv("resaleprice.csv")
df["year"] = df["month"].str[:4].astype(int)
df["month_of_year"] = df["month"].str[5:].astype(int)
df["quarter"] = df["year"].astype(str) + "Q" + ((df["month_of_year"] - 1) // 3 + 1).astype(str)
lease = df["remaining_lease"].str.extract(r"(\d+) years?(?: (\d+) months?)?").astype(float).fillna(0)
df["remaining_lease_yrs"] = lease[0] + lease[1] / 12
df["storey_mid"] = df["storey_range"].str[:2].astype(int) + 1
df["months_since_2017"] = (df["year"] - 2017) * 12 + df["month_of_year"]

# Target: sold above the median of comparable flats (town x flat type x quarter)
grp = df.groupby(["quarter", "town", "flat_type"])["resale_price"]
df["premium"] = (df["resale_price"] > grp.transform("median")).astype(int)
df = df[grp.transform("size") >= 10].reset_index(drop=True)  # need a meaningful peer group

NUM = ["floor_area_sqm", "remaining_lease_yrs", "storey_mid", "months_since_2017"]
CAT = ["town", "flat_type", "flat_model"]
# Dropped on purpose: resale_price (defines the label), block / street_name (near-unique
# identifiers that leak the building), lease_commence_date (duplicate of remaining lease).

# Temporal split: the model is used on future listings, so validate and test on later periods.
train = df[df["year"] <= 2023]
valid = df[df["year"] == 2024]
test = df[df["year"] >= 2025]
ytr, yva, yte = train["premium"].values, valid["premium"].values, test["premium"].values

pre = ColumnTransformer([
    ("num", StandardScaler(), NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=50), CAT),
])
Xtr = pre.fit_transform(train)
Xva = pre.transform(valid)
Xte = pre.transform(test)
feat_names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
split_info = {k: {"n": int(len(v)), "pos_rate": round(float(v["premium"].mean()), 4)}
              for k, v in {"train (2017-2023)": train, "validation (2024)": valid,
                           "test (2025-2026)": test}.items()}
print(split_info, "features:", Xtr.shape[1])


def clip01(s):
    return np.clip(s, 0.0, 1.0)


# --------------------------------------------------------------------------- 2-3. models
# Each family is trained with L2 loss (regression on the 0/1 label) and with binomial
# deviance (log loss). Hyperparameters are tuned separately per version, on validation AUC,
# over a deliberately small grid ("think more, try less").
tuning_rows, models = [], {}


def tune(name, family, loss, grid, build, score):
    best = (-1, None, None)
    for params in grid:
        m = build(**params).fit(Xtr, ytr)
        auc = roc_auc_score(yva, score(m, Xva))
        tuning_rows.append({"model": name, "params": json.dumps(params), "valid_auc": round(auc, 4)})
        if auc > best[0]:
            best = (auc, params, m)
    models[name] = {"family": family, "loss": loss, "params": best[1], "model": best[2],
                    "score": score,
                    # Platt scaling input: log-odds for deviance models, raw output for L2 models
                    "margin": (lambda m, X: m.decision_function(X)) if loss != "L2"
                    else (lambda m, X: m.predict(X))}
    print(f"{name:32s} best {best[1]}  valid AUC {best[0]:.4f}")


reg_score = lambda m, X: clip01(m.predict(X))
clf_score = lambda m, X: m.predict_proba(X)[:, 1]

tune("LASSO (L2 loss)", "glassbox", "L2",
     [{"alpha": a} for a in (1e-4, 1e-3, 3e-3, 1e-2)],
     lambda alpha: Lasso(alpha=alpha, max_iter=5000), reg_score)

tune("LASSO logistic (deviance)", "glassbox", "binomial deviance",
     [{"C": c} for c in (0.003, 0.03, 0.3, 3.0)],
     lambda C: LogisticRegression(penalty="l1", C=C, solver="liblinear", max_iter=300), clf_score)

gb_grid = [{"learning_rate": lr, "max_leaf_nodes": ln} for lr in (0.05, 0.15) for ln in (8, 31)]
gb_common = dict(max_iter=400, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
                 l2_regularization=1.0, random_state=SEED)
tune("Gradient boost (L2 loss)", "blackbox", "L2", gb_grid,
     lambda **p: HistGradientBoostingRegressor(loss="squared_error", **gb_common, **p), reg_score)
tune("Gradient boost (deviance)", "blackbox", "binomial deviance", gb_grid,
     lambda **p: HistGradientBoostingClassifier(loss="log_loss", **gb_common, **p), clf_score)
pd.DataFrame(tuning_rows).to_csv("results/tuning_grid.csv", index=False)

# --------------------------------------------------------------------------- helpers


def eval_loss(y, p, thr):
    pred = p >= thr
    fn = np.sum((y == 1) & ~pred)
    fp = np.sum((y == 0) & pred)
    return (C_FN * fn + C_FP * fp) / len(y)


def bin_table(y, p, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum():
            rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(mask.sum()),
                         "mean_predicted": p[mask].mean(), "observed_rate": y[mask].mean()})
    t = pd.DataFrame(rows)
    t["gap"] = (t["mean_predicted"] - t["observed_rate"]).abs()
    return t


def ece(y, p, n_bins=10):
    t = bin_table(y, p, n_bins)
    return float((t["n"] / t["n"].sum() * t["gap"]).sum())


def platt(s_fit, y_fit):
    """Platt scaling: logistic regression of the label on the raw score."""
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(s_fit.reshape(-1, 1), y_fit)
    return lambda s: lr.predict_proba(s.reshape(-1, 1))[:, 1]


# --------------------------------------------------------------------------- 4. selection
for name, m in models.items():
    m["s_tr"], m["s_va"], m["s_te"] = (m["score"](m["model"], X) for X in (Xtr, Xva, Xte))
    m["auc_tr"], m["auc_va"], m["auc_te"] = (roc_auc_score(y, s) for y, s in
                                             ((ytr, m["s_tr"]), (yva, m["s_va"]), (yte, m["s_te"])))
    m["cal"] = platt(m["margin"](m["model"], Xva), yva)   # calibrator fitted on validation only
    m["prob"] = lambda X, m=m: m["cal"](m["margin"](m["model"], X))
    m["p_va"], m["p_te"] = m["prob"](Xva), m["prob"](Xte)
    # Calibration decision is made on validation: keep raw scores if already calibrated.
    m["ece_va_raw"] = ece(yva, m["s_va"])
    m["use_platt"] = m["ece_va_raw"] > ECE_OK
    m["f_va"], m["f_te"] = (m["p_va"], m["p_te"]) if m["use_platt"] else (m["s_va"], m["s_te"])

comparison = pd.DataFrame([{
    "model": n, "type": m["family"], "training_loss": m["loss"], "best_params": json.dumps(m["params"]),
    "train_auc": m["auc_tr"], "valid_auc": m["auc_va"], "test_auc": m["auc_te"],
    "ece_valid_raw": ece(yva, m["s_va"]), "ece_test_raw": ece(yte, m["s_te"]),
    "ece_test_platt": ece(yte, m["p_te"]), "platt_applied": bool(m["use_platt"]),
} for n, m in models.items()]).round(4)
comparison.to_csv("results/model_comparison.csv", index=False)
print(comparison.drop(columns="best_params").to_string(index=False))

final_name = comparison.sort_values("valid_auc", ascending=False).iloc[0]["model"]
best_glass = comparison[comparison["type"] == "glassbox"].sort_values("valid_auc").iloc[-1]["model"]
F = models[final_name]
print("FINAL MODEL:", final_name)

fig, ax = plt.subplots(figsize=(5.2, 4.4))
for (n, m), c, ls in zip(models.items(), (GREY, GREY, ORANGE, BLUE), ("--", "-", "--", "-")):
    fpr, tpr, _ = roc_curve(yva, m["s_va"])
    ax.plot(fpr, tpr, color=c, ls=ls, lw=1.8, label=f"{n}  AUC {m['auc_va']:.3f}")
ax.plot([0, 1], [0, 1], color="#cccccc", lw=1)
ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Validation ROC (2024 listings)")
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig("figures/roc_validation.png"); plt.close(fig)

# --------------------------------------------------------------------------- 5. threshold
# Chosen on validation, using the final model's probabilities (Platt-scaled only if the
# validation check above said the raw scores were miscalibrated).
grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
loss_va = np.array([eval_loss(yva, F["f_va"], t) for t in grid])
thr = float(grid[loss_va.argmin()])
thr_theory = C_FP / (C_FP + C_FN)


def confusion(y, p, t):
    pred = p >= t
    return {"TP": int(np.sum((y == 1) & pred)), "FP": int(np.sum((y == 0) & pred)),
            "FN": int(np.sum((y == 1) & ~pred)), "TN": int(np.sum((y == 0) & ~pred))}


threshold_summary = {
    "chosen_threshold": thr, "theoretical_threshold_c_fp/(c_fp+c_fn)": round(thr_theory, 3),
    "valid_loss_at_chosen": round(float(loss_va.min()), 4),
    "valid_loss_at_0.5": round(eval_loss(yva, F["f_va"], 0.5), 4),
    "test_loss_at_chosen": round(eval_loss(yte, F["f_te"], thr), 4),
    "test_loss_at_0.5": round(eval_loss(yte, F["f_te"], 0.5), 4),
    "test_loss_flag_everything": round(eval_loss(yte, np.ones_like(F["f_te"]), 0.5), 4),
    "test_loss_flag_nothing": round(eval_loss(yte, np.zeros_like(F["f_te"]), 0.5), 4),
    "test_loss_best_glassbox_own_threshold": None,
    "test_confusion_at_chosen": confusion(yte, F["f_te"], thr),
    "test_confusion_at_0.5": confusion(yte, F["f_te"], 0.5),
    "test_misclassification_at_chosen": round(float(np.mean((F["f_te"] >= thr) != yte)), 4),
    "test_misclassification_at_0.5": round(float(np.mean((F["f_te"] >= 0.5) != yte)), 4),
}
G = models[best_glass]
g_thr = float(grid[np.argmin([eval_loss(yva, G["f_va"], t) for t in grid])])
threshold_summary["test_loss_best_glassbox_own_threshold"] = {
    "model": best_glass, "threshold": g_thr, "loss": round(eval_loss(yte, G["f_te"], g_thr), 4)}
print(json.dumps(threshold_summary, indent=1))

fig, ax = plt.subplots(figsize=(5.6, 4.0))
ax.plot(grid, loss_va, color=BLUE, lw=2, label="Validation (2024)")
ax.plot(grid, [eval_loss(yte, F["f_te"], t) for t in grid], color=GREY, lw=1.5, ls="--",
        label="Test (2025-2026)")
ax.axvline(thr, color=ORANGE, lw=1.5)
ax.annotate(f"chosen threshold {thr:.2f}\nloss {loss_va.min():.3f}", (thr, loss_va.min()),
            xytext=(thr + 0.08, loss_va.min() + 0.25), color=ORANGE,
            arrowprops=dict(arrowstyle="->", color=ORANGE))
ax.scatter([0.5], [eval_loss(yva, F["f_va"], 0.5)], color="black", zorder=5, s=25)
ax.annotate(f"default 0.50\nloss {eval_loss(yva, F['f_va'], 0.5):.3f}",
            (0.5, eval_loss(yva, F["f_va"], 0.5)), xytext=(0.56, 0.35), fontsize=9)
ax.set(xlabel="Classification threshold", ylabel=f"Average cost per listing ({C_FN:.0f}·FN + {C_FP:.0f}·FP) / n",
       title=f"Evaluation loss vs threshold - {final_name}")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig("figures/threshold_loss.png"); plt.close(fig)

# --------------------------------------------------------------------------- 6. calibration
fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
for ax, (n, m) in zip(axes, models.items()):
    for p, c, lab in ((m["s_te"], ORANGE, "raw"), (m["p_te"], BLUE, "Platt-scaled")):
        t = bin_table(yte, p)
        ax.plot(t["mean_predicted"], t["observed_rate"], "o-", color=c, ms=4, lw=1.6,
                label=f"{lab}  ECE {ece(yte, p):.3f}")
    ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=1)
    ax.set(title=n, xlabel="Mean predicted probability", xlim=(0, 1), ylim=(0, 1))
    ax.title.set_fontsize(9.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
axes[0].set_ylabel("Observed premium rate")
fig.suptitle("Reliability diagrams on test (2025-2026), 10 equal-width bins; Platt scaling fitted on validation",
             fontsize=10)
fig.tight_layout(); fig.savefig("figures/reliability_all_models.png"); plt.close(fig)

bins = bin_table(yte, F["s_te"]).merge(bin_table(yte, F["p_te"]), on="bin", how="outer",
                                       suffixes=("_raw", "_platt")).round(4)
bins.to_csv("results/calibration_bins_final_model.csv", index=False)
print(bins.to_string(index=False))

# --------------------------------------------------------------------------- 7. Tree SHAP
# Required on a tree-based blackbox model: the deviance gradient boost (log-odds scale).
tree_model = models["Gradient boost (deviance)"]["model"]
rng = np.random.default_rng(SEED)
idx = rng.choice(len(test), 3000, replace=False)
Xs, rows = Xte[idx], test.iloc[idx].reset_index(drop=True)
explainer = shap.TreeExplainer(tree_model)
sv = explainer.shap_values(Xs)
sv = sv[1] if isinstance(sv, list) else sv
if sv.ndim == 3:
    sv = sv[:, :, -1]
base = float(np.ravel(explainer.expected_value)[-1])

# One-hot columns are summed back to their original feature (SHAP values are additive).
orig = NUM + CAT
col_group = [next(o for o in orig if f == o or f.startswith(o + "_")) for f in feat_names]
grouped = np.column_stack([sv[:, [i for i, g in enumerate(col_group) if g == o]].sum(axis=1) for o in orig])
pretty = {"floor_area_sqm": "Floor area (sqm)", "remaining_lease_yrs": "Remaining lease (years)",
          "storey_mid": "Storey", "months_since_2017": "Sale month (time trend)",
          "town": "Town", "flat_type": "Flat type", "flat_model": "Flat model"}
recon = np.abs(grouped.sum(1) + base - tree_model.decision_function(Xs)).max()
print("SHAP additivity check, max abs error:", recon)

glob = pd.Series(np.abs(grouped).mean(0), index=[pretty[o] for o in orig]).sort_values()
glob.to_csv("results/shap_global_importance.csv", header=["mean_abs_shap_log_odds"])
fig, ax = plt.subplots(figsize=(5.6, 3.6))
ax.barh(glob.index, glob.values, color=BLUE)
for i, v in enumerate(glob.values):
    ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)
ax.set(xlabel="Mean |SHAP value| (log-odds of premium)", title="Global SHAP: overall feature importance")
fig.tight_layout(); fig.savefig("figures/shap_global_bar.png"); plt.close(fig)

# Global direction of effect for the numeric features (beeswarm)
num_idx = [orig.index(o) for o in NUM]
shap.summary_plot(grouped[:, num_idx], rows[NUM].values, feature_names=[pretty[o] for o in NUM],
                  show=False, plot_size=(6.2, 3.2))
plt.title("Global SHAP: direction of effect (numeric features)", fontsize=10)
plt.tight_layout(); plt.savefig("figures/shap_global_beeswarm.png"); plt.close()

# Local explanation: a listing flagged premium with high confidence, and one borderline case
p_s = F["prob"](Xs) if F["use_platt"] else F["score"](F["model"], Xs)
local = {}
for tag, i in (("flagged", int(np.argsort(p_s)[int(0.95 * len(p_s))])),
               ("borderline", int(np.argmin(np.abs(p_s - thr))))):
    r = rows.iloc[i]
    data = np.array([f"{r[o]:.0f}" if o in NUM else str(r[o]) for o in orig], dtype=object)
    exp = shap.Explanation(values=grouped[i], base_values=base, data=data,
                           feature_names=[pretty[o] for o in orig])
    shap.plots.waterfall(exp, max_display=7, show=False)
    plt.title(f"Local SHAP ({tag} listing): P(premium) = {p_s[i]:.2f}, actual = {int(r['premium'])}",
              fontsize=10)
    plt.tight_layout(); plt.savefig(f"figures/shap_local_{tag}.png"); plt.close()
    local[tag] = {"listing": {o: (float(r[o]) if o in NUM else str(r[o])) for o in orig},
                  "month": r["month"], "resale_price": float(r["resale_price"]),
                  "calibrated_p": round(float(p_s[i]), 3), "actual_premium": int(r["premium"]),
                  "shap_log_odds": {pretty[o]: round(float(v), 3) for o, v in zip(orig, grouped[i])},
                  "base_value": round(base, 3)}

# Glassbox counterpart: the non-zero LASSO-logistic coefficients
coef = pd.Series(models["LASSO logistic (deviance)"]["model"].coef_.ravel(), index=feat_names)
coef[coef != 0].sort_values(key=np.abs, ascending=False).round(4).to_csv(
    "results/lasso_logistic_coefficients.csv", header=["coefficient"])

summary = {"cost_matrix": {"FN": C_FN, "FP": C_FP}, "splits": split_info, "n_features_encoded": int(Xtr.shape[1]),
           "final_model": final_name, "final_model_platt_applied": bool(F["use_platt"]), "final_params": F["params"],
           "models": json.loads(comparison.to_json(orient="records")),
           "threshold": threshold_summary, "shap_global": glob.round(4).to_dict(), "shap_local": local,
           "lasso_nonzero_coefs": int((coef != 0).sum())}
with open("results/summary.json", "w") as fh:
    json.dump(summary, fh, indent=1)
print("done")
