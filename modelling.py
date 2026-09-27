"""
CardioAI — Deliverable 3: Modelling & Analysis
Supervised Learning Pipeline for Heart Disease Classification
"""

import csv
import math
import random

random.seed(42)

# ─────────────────────────────────────────────
# 1. LOAD THE CLEANED DATASET
# ─────────────────────────────────────────────

filename = "data/heart_disease_cleaned.csv"

with open(filename) as f:
    reader = csv.DictReader(f)
    raw_data = list(reader)

print(f"Loaded {len(raw_data)} rows from {filename}")
print(f"Columns: {list(raw_data[0].keys())}")

# Convert all values to float
data = []
for row in raw_data:
    new_row = {}
    for k, v in row.items():
        try:
            new_row[k] = float(v)
        except ValueError:
            if k == "thal":
                new_row[k] = {"normal": 0.0, "fixed defect": 1.0, "reversable defect": 2.0}[v]
            else:
                raise ValueError(f"Unexpected non-numeric value in {k}: {v!r}")
    data.append(new_row)

# Separate features and target
feature_cols = [k for k in data[0].keys() if k != "target"]
X = [[row[c] for c in feature_cols] for row in data]
y = [int(row["target"]) for row in data]

print(f"\nFeatures: {len(feature_cols)}")
print(f"Target distribution — 0: {y.count(0)}, 1: {y.count(1)}")


# ─────────────────────────────────────────────
# 2. TRAIN / TEST SPLIT  (80 / 20, stratified)
# ─────────────────────────────────────────────

def train_test_split_stratified(X, y, test_size=0.20, seed=42):
    random.seed(seed)
    idx_0 = [i for i, v in enumerate(y) if v == 0]
    idx_1 = [i for i, v in enumerate(y) if v == 1]
    random.shuffle(idx_0)
    random.shuffle(idx_1)
    n_test_0 = round(len(idx_0) * test_size)
    n_test_1 = round(len(idx_1) * test_size)
    test_idx  = idx_0[:n_test_0] + idx_1[:n_test_1]
    train_idx = idx_0[n_test_0:] + idx_1[n_test_1:]
    random.shuffle(train_idx); random.shuffle(test_idx)
    X_train = [X[i] for i in train_idx]
    X_test  = [X[i] for i in test_idx]
    y_train = [y[i] for i in train_idx]
    y_test  = [y[i] for i in test_idx]
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)
print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# ─────────────────────────────────────────────
# 3. SCALE ON TRAINING DATA ONLY
# ─────────────────────────────────────────────

def fit_scaler(X):
    n_feat = len(X[0])
    means = [sum(row[j] for row in X) / len(X) for j in range(n_feat)]
    stds  = [
        math.sqrt(sum((row[j] - means[j])**2 for row in X) / len(X)) or 1.0
        for j in range(n_feat)
    ]
    return means, stds

def apply_scaler(X, means, stds):
    return [[(row[j] - means[j]) / stds[j] for j in range(len(row))] for row in X]

means, stds = fit_scaler(X_train)
X_train_sc  = apply_scaler(X_train, means, stds)
X_test_sc   = apply_scaler(X_test,  means, stds)

# ─────────────────────────────────────────────
# 4. METRICS HELPERS
# ─────────────────────────────────────────────

def confusion(y_true, y_pred):
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    return tp, tn, fp, fn

def metrics(y_true, y_pred):
    tp, tn, fp, fn = confusion(y_true, y_pred)
    acc  = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn) if (tp + fn) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}

def roc_auc(y_true, scores):
    """Approximate AUC via trapezoidal rule over sorted score thresholds."""
    pairs = sorted(zip(scores, y_true), reverse=True)
    pos = sum(y_true); neg = len(y_true) - pos
    tp = fp = 0; prev_tpr = prev_fpr = 0; auc = 0
    for _, label in pairs:
        if label == 1: tp += 1
        else:          fp += 1
        tpr = tp / pos; fpr = fp / neg
        auc += (fpr - prev_fpr) * (prev_tpr + tpr) / 2
        prev_tpr, prev_fpr = tpr, fpr
    return auc


# ─────────────────────────────────────────────
# 5. MODEL 1 — LOGISTIC REGRESSION (from scratch)
# ─────────────────────────────────────────────

def sigmoid(z):
    return 1 / (1 + math.exp(-max(-500, min(500, z))))

def logistic_train(X, y, lr=0.05, epochs=500, lam=0.01):
    n, p = len(X), len(X[0])
    w = [0.0] * p; b = 0.0
    for _ in range(epochs):
        dw = [0.0] * p; db = 0.0
        for xi, yi in zip(X, y):
            pred = sigmoid(sum(w[j] * xi[j] for j in range(p)) + b)
            err  = pred - yi
            for j in range(p):
                dw[j] += err * xi[j]
            db += err
        w = [w[j] - lr * (dw[j] / n + lam * w[j]) for j in range(p)]
        b -= lr * (db / n)
    return w, b

def logistic_predict(X, w, b, threshold=0.5):
    probs = [sigmoid(sum(w[j] * xi[j] for j in range(len(w))) + b) for xi in X]
    preds = [1 if p >= threshold else 0 for p in probs]
    return preds, probs

print("\n" + "="*55)
print("MODEL 1: Logistic Regression")
w_lr, b_lr = logistic_train(X_train_sc, y_train)
preds_lr, probs_lr = logistic_predict(X_test_sc, w_lr, b_lr)
m_lr = metrics(y_test, preds_lr)
auc_lr = roc_auc(y_test, probs_lr)
m_lr["auc"] = auc_lr
print(f"  Accuracy : {m_lr['accuracy']:.4f}")
print(f"  Precision: {m_lr['precision']:.4f}")
print(f"  Recall   : {m_lr['recall']:.4f}")
print(f"  F1-Score : {m_lr['f1']:.4f}")
print(f"  AUC-ROC  : {m_lr['auc']:.4f}")


# ─────────────────────────────────────────────
# 6. MODEL 2 — K-NEAREST NEIGHBOURS
# ─────────────────────────────────────────────

def euclidean(a, b):
    return math.sqrt(sum((x - y)**2 for x, y in zip(a, b)))

def knn_predict(X_train, y_train, X_test, k=7):
    preds, probs = [], []
    for xi in X_test:
        dists = sorted([(euclidean(xi, xt), yt) for xt, yt in zip(X_train, y_train)])
        neighbours = [d[1] for d in dists[:k]]
        p = sum(neighbours) / k
        probs.append(p)
        preds.append(1 if p >= 0.5 else 0)
    return preds, probs

print("\n" + "="*55)
print("MODEL 2: K-Nearest Neighbours (k=7)")
preds_knn, probs_knn = knn_predict(X_train_sc, y_train, X_test_sc, k=7)
m_knn = metrics(y_test, preds_knn)
auc_knn = roc_auc(y_test, probs_knn)
m_knn["auc"] = auc_knn
print(f"  Accuracy : {m_knn['accuracy']:.4f}")
print(f"  Precision: {m_knn['precision']:.4f}")
print(f"  Recall   : {m_knn['recall']:.4f}")
print(f"  F1-Score : {m_knn['f1']:.4f}")
print(f"  AUC-ROC  : {m_knn['auc']:.4f}")


# ─────────────────────────────────────────────
# 7. MODEL 3 — DECISION TREE (CART-style, from scratch)
# ─────────────────────────────────────────────

def gini(labels):
    n = len(labels)
    if n == 0: return 0
    p = sum(labels) / n
    return 1 - p**2 - (1 - p)**2

def best_split(X, y):
    best_g, best_feat, best_thresh = float("inf"), None, None
    n = len(y)
    for j in range(len(X[0])):
        vals = sorted(set(row[j] for row in X))
        for t in vals:
            left_y  = [y[i] for i in range(n) if X[i][j] <= t]
            right_y = [y[i] for i in range(n) if X[i][j] >  t]
            if not left_y or not right_y: continue
            g = (len(left_y) * gini(left_y) + len(right_y) * gini(right_y)) / n
            if g < best_g:
                best_g, best_feat, best_thresh = g, j, t
    return best_feat, best_thresh

def build_tree(X, y, depth=0, max_depth=5, min_samples=10):
    if depth >= max_depth or len(y) < min_samples or len(set(y)) == 1:
        return {"leaf": True, "pred": round(sum(y) / len(y)), "prob": sum(y) / len(y)}
    feat, thresh = best_split(X, y)
    if feat is None:
        return {"leaf": True, "pred": round(sum(y) / len(y)), "prob": sum(y) / len(y)}
    left_mask  = [i for i in range(len(y)) if X[i][feat] <= thresh]
    right_mask = [i for i in range(len(y)) if X[i][feat] >  thresh]
    return {
        "leaf": False, "feat": feat, "thresh": thresh,
        "left":  build_tree([X[i] for i in left_mask],  [y[i] for i in left_mask],  depth+1, max_depth, min_samples),
        "right": build_tree([X[i] for i in right_mask], [y[i] for i in right_mask], depth+1, max_depth, min_samples),
    }

def tree_predict_one(node, xi):
    if node["leaf"]:
        return node["pred"], node["prob"]
    if xi[node["feat"]] <= node["thresh"]:
        return tree_predict_one(node["left"], xi)
    return tree_predict_one(node["right"], xi)

def tree_predict(tree, X):
    preds, probs = zip(*[tree_predict_one(tree, xi) for xi in X])
    return list(preds), list(probs)

print("\n" + "="*55)
print("MODEL 3: Decision Tree (max_depth=5)")
tree = build_tree(X_train_sc, y_train, max_depth=5)
preds_dt, probs_dt = tree_predict(tree, X_test_sc)
m_dt = metrics(y_test, preds_dt)
auc_dt = roc_auc(y_test, probs_dt)
m_dt["auc"] = auc_dt
print(f"  Accuracy : {m_dt['accuracy']:.4f}")
print(f"  Precision: {m_dt['precision']:.4f}")
print(f"  Recall   : {m_dt['recall']:.4f}")
print(f"  F1-Score : {m_dt['f1']:.4f}")
print(f"  AUC-ROC  : {m_dt['auc']:.4f}")


# ─────────────────────────────────────────────
# 8. 5-FOLD CROSS-VALIDATION (Logistic Regression)
# ─────────────────────────────────────────────

def cross_validate_lr(X, y, k=5, lr=0.05, epochs=500, lam=0.01, seed=42):
    random.seed(seed)
    idx = list(range(len(y)))
    random.shuffle(idx)
    folds = [idx[i::k] for i in range(k)]
    fold_f1s = []
    for i in range(k):
        test_idx  = folds[i]
        train_idx = [j for fi, fold in enumerate(folds) if fi != i for j in fold]
        Xtr = [X[j] for j in train_idx]; ytr = [y[j] for j in train_idx]
        Xte = [X[j] for j in test_idx];  yte = [y[j] for j in test_idx]
        ms, ss = fit_scaler(Xtr)
        Xtr_s = apply_scaler(Xtr, ms, ss)
        Xte_s = apply_scaler(Xte, ms, ss)
        w, b = logistic_train(Xtr_s, ytr, lr=lr, epochs=epochs, lam=lam)
        preds, _ = logistic_predict(Xte_s, w, b)
        m = metrics(yte, preds)
        fold_f1s.append(m["f1"])
    return fold_f1s

print("\n" + "="*55)
print("5-Fold Cross-Validation (Logistic Regression)")
cv_f1s = cross_validate_lr(X, y)
mean_f1 = sum(cv_f1s) / len(cv_f1s)
std_f1  = math.sqrt(sum((x - mean_f1)**2 for x in cv_f1s) / len(cv_f1s))
for i, f in enumerate(cv_f1s, 1):
    print(f"  Fold {i}: F1 = {f:.4f}")
print(f"  Mean F1 = {mean_f1:.4f} ± {std_f1:.4f}")


# ─────────────────────────────────────────────
# 9. FEATURE IMPORTANCE (Logistic Regression coefficients)
# ─────────────────────────────────────────────

abs_weights = [(abs(w_lr[j]), feature_cols[j], w_lr[j]) for j in range(len(w_lr))]
abs_weights.sort(reverse=True)

print("\n" + "="*55)
print("Top 10 Feature Importances (|LR coefficients|)")
for rank, (abs_w, name, raw_w) in enumerate(abs_weights[:10], 1):
    direction = "+" if raw_w > 0 else "-"
    print(f"  {rank:2}. {name:<25} |coef| = {abs_w:.4f}  ({direction})")


# ─────────────────────────────────────────────
# 10. SAVE RESULTS SUMMARY
# ─────────────────────────────────────────────

results = {
    "Logistic Regression": m_lr,
    "K-Nearest Neighbours": m_knn,
    "Decision Tree": m_dt,
}

with open("model_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "Accuracy", "Precision", "Recall", "F1", "AUC-ROC"])
    for name, m in results.items():
        writer.writerow([name,
                         round(m["accuracy"], 4), round(m["precision"], 4),
                         round(m["recall"], 4),   round(m["f1"], 4),
                         round(m["auc"], 4)])

print("\n" + "="*55)
print("Results saved to model_results.csv")
print("="*55)
