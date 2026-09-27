"""
CardioAI — Deliverable 4: Final Visualisation & Insight Dashboard
=================================================================
Produces the final set of figures that answer the original project
questions and link data -> model -> insight for a non-technical
audience. All figures are saved as PNGs in figures/.

Run:
    python visualisation.py

Inputs  : heart_disease_cleaned.csv  (output of Deliverable 2)
Outputs : figures/*.png              (embedded into the final report)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    confusion_matrix, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
)

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# Colour palette (colour-blind safe, consistent across charts)
C_NEG  = "#4C9FD4"   # no disease
C_POS  = "#E45756"   # disease
PAL2   = [C_NEG, C_POS]

# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA (cleaned output of Deliverable 2)
# ─────────────────────────────────────────────────────────────

df = pd.read_csv("data/heart_disease_cleaned.csv")

# `thal` is a text label from D2; encode numerically for modelling.
thal_map = {"normal": 0, "fixed defect": 1, "reversable defect": 2}
df["thal"] = df["thal"].map(thal_map).fillna(0).astype(int)
for c in ["cp", "restecg", "slope"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

# Engineered features from D2
df["hypertension_flag"] = (df["trestbps"] >= 140).astype(int)
df["high_chol_flag"]    = (df["chol"] >= 240).astype(int)
df["hr_reserve"]        = (220 - df["age"]) - df["thalch"]

# Friendly labels for plotting
df_plot = df.copy()
df_plot["Diagnosis"] = df_plot["target"].map({0: "No disease", 1: "Disease"})
df_plot["Sex"]       = df_plot["sex"].map({0: "Female", 1: "Male"})

# ─────────────────────────────────────────────────────────────
# 2. TRAIN / TEST SPLIT + SCALING (consistent with D3)
# ─────────────────────────────────────────────────────────────

feature_cols = [c for c in df.columns if c != "target"]
X = df[feature_cols].values
y = df["target"].values

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
scaler = StandardScaler().fit(X_tr)
X_tr_s = scaler.transform(X_tr)
X_te_s = scaler.transform(X_te)

models = {
    "Logistic Regression":  LogisticRegression(C=1/0.01, max_iter=2000, random_state=42),
    "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=7),
    "Decision Tree":        DecisionTreeClassifier(max_depth=5, min_samples_split=10,
                                                   random_state=42),
}

results = {}
for name, m in models.items():
    m.fit(X_tr_s, y_tr)
    y_pred  = m.predict(X_te_s)
    y_proba = m.predict_proba(X_te_s)[:, 1]
    results[name] = dict(
        model=m, y_pred=y_pred, y_proba=y_proba,
        accuracy=accuracy_score(y_te, y_pred),
        precision=precision_score(y_te, y_pred),
        recall=recall_score(y_te, y_pred),
        f1=f1_score(y_te, y_pred),
        auc=roc_auc_score(y_te, y_proba),
    )

print("Test-set performance")
print(f"{'Model':<25}{'Acc':>7}{'Prec':>8}{'Rec':>8}{'F1':>8}{'AUC':>8}")
for name, r in results.items():
    print(f"{name:<25}{r['accuracy']:>7.3f}{r['precision']:>8.3f}"
          f"{r['recall']:>8.3f}{r['f1']:>8.3f}{r['auc']:>8.3f}")


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)


# ─────────────────────────────────────────────────────────────
# FIGURE 1 — Target class balance
# ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6.5, 4))
counts = df_plot["Diagnosis"].value_counts()
bars = ax.bar(counts.index, counts.values, color=[C_NEG, C_POS], edgecolor="black")
for b, v in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width()/2, v + 5, f"{v}  ({v/len(df)*100:.1f}%)",
            ha="center", fontsize=11)
ax.set_title("Who is in the dataset?  — Heart-disease diagnosis")
ax.set_ylabel("Number of patients")
ax.set_ylim(0, max(counts.values) + 70)
style_ax(ax)
plt.tight_layout()
plt.savefig(f"{OUT}/fig01_target_balance.png")
plt.close()

# ─────────────────────────────────────────────────────────────
# FIGURE 2 — Age distribution by diagnosis (stacked histogram)
# ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 4.2))
bins = np.linspace(df["age"].min(), df["age"].max(), 26)
ax.hist([df_plot.loc[df_plot["target"] == 0, "age"],
         df_plot.loc[df_plot["target"] == 1, "age"]],
        bins=bins, stacked=True, color=PAL2, edgecolor="white",
        label=["No disease", "Disease"])
ax.set_title("Age profile by diagnosis")
ax.set_xlabel("Age (years)")
ax.set_ylabel("Number of patients")
ax.legend(frameon=False)
style_ax(ax)
plt.tight_layout()
plt.savefig(f"{OUT}/fig02_age_distribution.png")
plt.close()

# ─────────────────────────────────────────────────────────────
# FIGURE 3 — Disease rate by sex & age band
# ─────────────────────────────────────────────────────────────

bin_edges = [28, 40, 50, 60, 70, 80]
band_labels = ["<40", "40-49", "50-59", "60-69", "70+"]
df_plot["Age group"] = pd.cut(df_plot["age"], bins=bin_edges, labels=band_labels, right=False)
rate = (df_plot.groupby(["Age group", "Sex"])["target"]
        .mean().unstack())

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(band_labels))
w = 0.36
ax.bar(x - w/2, rate["Female"].values, w, label="Female", color="#F4B942", edgecolor="black")
ax.bar(x + w/2, rate["Male"].values,   w, label="Male",   color="#3E78B2", edgecolor="black")
for xi, (f, mv) in enumerate(zip(rate["Female"], rate["Male"])):
    ax.text(xi - w/2, f + 0.02,  f"{f*100:.0f}%", ha="center", fontsize=9)
    ax.text(xi + w/2, mv + 0.02, f"{mv*100:.0f}%", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(band_labels)
ax.set_ylim(0, 1.0)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
ax.set_ylabel("Disease rate")
ax.set_title("Share of patients diagnosed with heart disease — by age band and sex")
ax.legend(frameon=False)
style_ax(ax)
plt.tight_layout()
plt.savefig(f"{OUT}/fig03_disease_by_age_sex.png")
plt.close()

# ─────────────────────────────────────────────────────────────
# FIGURE 4 — Key clinical markers (boxplots)
# ─────────────────────────────────────────────────────────────

key = [("thalch",   "Max heart rate achieved (bpm)"),
       ("oldpeak",  "ST depression (exercise)"),
       ("chol",     "Cholesterol (mg/dL)"),
       ("trestbps", "Resting blood pressure (mmHg)")]

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for (col, title), ax in zip(key, axes.flat):
    data = [df.loc[df["target"] == 0, col], df.loc[df["target"] == 1, col]]
    bp = ax.boxplot(data, patch_artist=True, widths=0.45,
                    labels=["No disease", "Disease"])
    for patch, c in zip(bp["boxes"], PAL2):
        patch.set_facecolor(c); patch.set_alpha(0.85); patch.set_edgecolor("black")
    for med in bp["medians"]:
        med.set_color("black"); med.set_linewidth(1.5)
    ax.set_title(title)
    style_ax(ax)
fig.suptitle("How the four strongest clinical markers differ between groups",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT}/fig04_clinical_markers.png")
plt.close()

# ─────────────────────────────────────────────────────────────
# FIGURE 5 — Correlation heat-map
# ─────────────────────────────────────────────────────────────

corr_cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
             "thalch", "exang", "oldpeak", "slope", "ca", "thal",
             "hr_reserve", "target"]
corr = df[corr_cols].corr().values
n = len(corr_cols)

fig, ax = plt.subplots(figsize=(8.5, 7.5))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
# annotate lower triangle + diagonal
for i in range(n):
    for j in range(n):
        if j <= i:
            ax.text(j, i, f"{corr[i, j]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if abs(corr[i, j]) > 0.55 else "black")
ax.set_xticks(range(n)); ax.set_xticklabels(corr_cols, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(n)); ax.set_yticklabels(corr_cols, fontsize=9)
ax.set_title("Feature correlations (Pearson)")
cb = plt.colorbar(im, ax=ax, shrink=0.7)
cb.set_label("Correlation")
ax.grid(False)
plt.tight_layout()
plt.savefig(f"{OUT}/fig05_correlation_heatmap.png")
plt.close()

# ─────────────────────────────────────────────────────────────
# FIGURE 6 — Model comparison bar chart
# ─────────────────────────────────────────────────────────────

metrics = ["accuracy", "precision", "recall", "f1", "auc"]
metric_names = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
metric_colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E"]

fig, ax = plt.subplots(figsize=(9, 4.8))
x = np.arange(len(models))
w = 0.16
for i, (m_key, m_name, c) in enumerate(zip(metrics, metric_names, metric_colors)):
    vals = [results[n_][m_key] for n_ in models]
    bars = ax.bar(x + (i - 2) * w, vals, w, label=m_name, color=c, edgecolor="black")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.012, f"{v:.2f}",
                ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(list(models.keys()))
ax.set_ylim(0, 1.08)
ax.set_ylabel("Score")
ax.set_title("Model performance on the held-out test set (n = 184)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=5, frameon=False)
style_ax(ax)
plt.tight_layout()
plt.savefig(f"{OUT}/fig06_model_comparison.png")
plt.close()

# ─────────────────────────────────────────────────────────────
# FIGURE 7 — ROC curves
# ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6.5, 5.5))
model_colors = {"Logistic Regression": "#2E86AB",
                "K-Nearest Neighbours": "#A23B72",
                "Decision Tree":        "#F18F01"}
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_te, r["y_proba"])
    ax.plot(fpr, tpr, color=model_colors[name], lw=2.2,
            label=f"{name}  (AUC = {r['auc']:.2f})")
ax.plot([0, 1], [0, 1], ls="--", color="grey", label="Random (AUC = 0.50)")
ax.set_title("How well does each model separate disease from no-disease?")
ax.set_xlabel("False-positive rate")
ax.set_ylabel("True-positive rate (Recall)")
ax.legend(loc="lower right", frameon=False)
style_ax(ax)
plt.tight_layout()
plt.savefig(f"{OUT}/fig07_roc_curves.png")
plt.close()

# ─────────────────────────────────────────────────────────────
# FIGURE 8 — Confusion matrix (Logistic Regression — primary model)
# ─────────────────────────────────────────────────────────────

best_name = "Logistic Regression"
cm = confusion_matrix(y_te, results[best_name]["y_pred"])
labels_text = np.array([
    [f"True negatives\n{cm[0, 0]}",  f"False positives\n{cm[0, 1]}"],
    [f"False negatives\n{cm[1, 0]}", f"True positives\n{cm[1, 1]}"],
])

fig, ax = plt.subplots(figsize=(6.2, 4.6))
im = ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, labels_text[i, j], ha="center", va="center",
                fontsize=11, color="white" if cm[i, j] > cm.max() * 0.6 else "black")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Predicted No disease", "Predicted Disease"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual No disease", "Actual Disease"])
ax.set_title(f"Confusion matrix — {best_name}  (test set)")
ax.grid(False)
plt.tight_layout()
plt.savefig(f"{OUT}/fig08_confusion_matrix.png")
plt.close()

# ─────────────────────────────────────────────────────────────
# FIGURE 9 — Feature importance (LR coefficients, absolute)
# ─────────────────────────────────────────────────────────────

lr = results["Logistic Regression"]["model"]
coefs = pd.Series(lr.coef_[0], index=feature_cols)
top = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(12)

fig, ax = plt.subplots(figsize=(8, 5.5))
colors_fi = [C_POS if v > 0 else C_NEG for v in top.values]
ax.barh(top.index[::-1], top.values[::-1], color=colors_fi[::-1], edgecolor="black")
ax.axvline(0, color="black", lw=0.8)
ax.set_title("Which factors drive the model's prediction?")
ax.set_xlabel("Logistic Regression coefficient  (→ higher risk  ⁄  ← lower risk)")
ax.legend(handles=[Patch(color=C_POS, label="Increases disease risk"),
                   Patch(color=C_NEG, label="Decreases disease risk")],
          loc="lower right", frameon=False)
style_ax(ax)
plt.tight_layout()
plt.savefig(f"{OUT}/fig09_feature_importance.png")
plt.close()

# ─────────────────────────────────────────────────────────────
# FIGURE 10 — Cross-validation stability
# ─────────────────────────────────────────────────────────────

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = {name: [] for name in models}
for name, m in models.items():
    for tr, te in cv.split(X, y):
        sc = StandardScaler().fit(X[tr])
        m2 = type(m)(**m.get_params())
        m2.fit(sc.transform(X[tr]), y[tr])
        cv_scores[name].append(f1_score(y[te], m2.predict(sc.transform(X[te]))))

fig, ax = plt.subplots(figsize=(7.5, 4.5))
data = [cv_scores[n_] for n_ in models]
bp = ax.boxplot(data, patch_artist=True, widths=0.45,
                labels=list(models.keys()))
for patch, c in zip(bp["boxes"], ["#2E86AB", "#A23B72", "#F18F01"]):
    patch.set_facecolor(c); patch.set_alpha(0.75); patch.set_edgecolor("black")
for med in bp["medians"]:
    med.set_color("black"); med.set_linewidth(1.5)
# overlay individual fold scores
for i, vals in enumerate(data, start=1):
    ax.scatter([i] * len(vals), vals, color="black", s=22, zorder=3, alpha=0.7)
ax.set_ylabel("F1-score across 5 folds")
ax.set_title("Are the results stable across different data splits?  (5-fold CV)")
ax.set_ylim(0.7, 1.0)
style_ax(ax)
plt.tight_layout()
plt.savefig(f"{OUT}/fig10_cv_stability.png")
plt.close()

# ─────────────────────────────────────────────────────────────
# FIGURE 11 — Predicted probability distribution
# ─────────────────────────────────────────────────────────────

probs = results["Logistic Regression"]["y_proba"]
fig, ax = plt.subplots(figsize=(7.5, 4.2))
bins = np.linspace(0, 1, 21)
ax.hist(probs[y_te == 0], bins=bins, alpha=0.75, label="No disease",
        color=C_NEG, edgecolor="white")
ax.hist(probs[y_te == 1], bins=bins, alpha=0.75, label="Disease",
        color=C_POS, edgecolor="white")
ax.axvline(0.5, color="black", ls="--", lw=1.1, label="Decision threshold (0.5)")
ax.set_title("Model-assigned risk scores on the test set")
ax.set_xlabel("Predicted probability of heart disease")
ax.set_ylabel("Number of patients")
ax.legend(frameon=False)
style_ax(ax)
plt.tight_layout()
plt.savefig(f"{OUT}/fig11_risk_scores.png")
plt.close()

# ─────────────────────────────────────────────────────────────
# FIGURE 12 — Decision Tree visualisation
# ─────────────────────────────────────────────────────────────

dt = results["Decision Tree"]["model"]
fig, ax = plt.subplots(figsize=(15, 8))
plot_tree(dt, feature_names=feature_cols,
          class_names=["No disease", "Disease"],
          filled=True, impurity=False, rounded=True,
          max_depth=3, fontsize=8, ax=ax)
ax.set_title("Decision Tree — top three levels of clinical rules")
plt.tight_layout()
plt.savefig(f"{OUT}/fig12_decision_tree.png")
plt.close()

# ─────────────────────────────────────────────────────────────
# FIGURE 13 — Patient risk segmentation
# ─────────────────────────────────────────────────────────────

seg = pd.DataFrame({"p": probs, "y": y_te})
seg_bins = [0, 0.25, 0.5, 0.75, 1.01]
seg_labels = ["Low (<25%)", "Moderate (25-50%)", "High (50-75%)", "Very high (>75%)"]
seg["band"] = pd.cut(seg["p"], bins=seg_bins, labels=seg_labels, right=False)
summary = seg.groupby("band").agg(n=("y", "size"), actual_rate=("y", "mean")).reset_index()

fig, ax = plt.subplots(figsize=(8, 4.6))
bars = ax.bar(summary["band"].astype(str), summary["n"],
              color=["#6A994E", "#A7C957", "#F18F01", "#E45756"], edgecolor="black")
for b, n_, r in zip(bars, summary["n"], summary["actual_rate"]):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5,
            f"{int(n_)} patients\n{r*100:.0f}% actually sick",
            ha="center", fontsize=9)
ax.set_title("Risk stratification — how the model sorts the test cohort")
ax.set_ylabel("Number of test patients")
ax.set_xlabel("Predicted risk band")
ax.set_ylim(0, max(summary["n"]) * 1.35)
style_ax(ax)
plt.tight_layout()
plt.savefig(f"{OUT}/fig13_risk_segmentation.png")
plt.close()

# ─────────────────────────────────────────────────────────────
# Persist numeric summary
# ─────────────────────────────────────────────────────────────

mdf = pd.DataFrame({m: [results[n_][m] for n_ in results] for m in metrics},
                   index=list(results.keys()))
mdf.to_csv("model_results_viz.csv")

print(f"\n{len(os.listdir(OUT))} figures saved to {OUT}/")
print("Risk segmentation:\n", summary)
