# %% [markdown]
# # 05 — Status Report Experiments (Deliverable 3)
#
# This notebook/script produces every number and figure in the Interim Status Report.
# It answers the feedback from Interim Progress Report 1 and covers the new rubric items:
#
# 1. **Data release check** — compare our file with UNSW's published training/testing counts.
# 2. **De-duplication by default** — drop exact feature duplicates and conflicting-label rows before any split.
# 3. **Host-grouped evaluation** — no source host appears in both training and test (StratifiedGroupKFold on `src_ip`),
#    compared with an ordinary stratified 5-fold split.
# 4. **Port / host fingerprint ablation** — raw ports vs. no ports vs. bucketed ports vs. adding IP addresses.
# 5. **Model comparison** — majority class, Logistic Regression, Random Forest, Histogram Gradient Boosting,
#    and an unsupervised Isolation Forest, with validation-based tuning.
# 6. **Error analysis, per-group performance, SHAP explanations and explanation stability.**
# 7. **Adversarial evasion test** (attacker-controlled feature manipulation) and an adversarial-training defense.
# 8. **Generalization** to an external dataset (UNSW-NB15) through a shared feature schema, plus drift (PSI).
#
# Run from the repository root or from `notebooks/`. Total runtime is roughly 45–60 minutes on a laptop.
# Results are cached in `results/status_report/cache/`, so re-running skips finished steps.
# Delete that folder to recompute everything.

# %%
from pathlib import Path
import hashlib
import itertools
import json
import pickle
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name in ("notebooks", "src") else Path.cwd()
CONFIG = json.loads((PROJECT_ROOT / "configs" / "status_report_config.json").read_text())
SEED = CONFIG["random_seed"]
RNG = np.random.default_rng(SEED)

DATA_PATH = PROJECT_ROOT / CONFIG["data_path"]
EXT_DIR = PROJECT_ROOT / CONFIG["external_dir"]
import os
SMOKE = bool(os.environ.get("SMOKE"))  # quick test mode on a small sample
OUT = PROJECT_ROOT / "results" / ("status_report_smoke" if SMOKE else "status_report")
MET = OUT / "metrics"
FIG = OUT / "figures"
CACHE = OUT / "cache"
for folder in (MET, FIG, CACHE):
    folder.mkdir(parents=True, exist_ok=True)


def cached(name, compute):
    """Run compute() once and store the result, so long steps are not repeated."""
    path = CACHE / f"{name}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    start = time.perf_counter()
    result = compute()
    path.write_bytes(pickle.dumps(result))
    print(f"[{name}] finished in {time.perf_counter() - start:.0f}s")
    return result


print("Project root:", PROJECT_ROOT.name)

# %% [markdown]
# ## 1. Data release check
# UNSW publishes the class counts of the official training/testing network file (461,043 rows, 300,000 normal).
# We compare our copy class by class and record its SHA-256 so the exact file can be identified later.

# %%
raw = pd.read_csv(DATA_PATH)
sha256 = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
official = CONFIG["official_train_test_counts"]
ours = raw["type"].value_counts().to_dict()
release = pd.DataFrame(
    {
        "class": list(official),
        "official_train_test": [official[k] for k in official],
        "our_file": [ours.get(k, 0) for k in official],
    }
)
release["matches_official"] = release["official_train_test"] == release["our_file"]
release.loc[len(release)] = ["TOTAL", sum(official.values()), len(raw), sum(official.values()) == len(raw)]
release.to_csv(MET / "data_release_check.csv", index=False)
(MET / "data_release_sha256.txt").write_text(f"{DATA_PATH.name}\n{sha256}\nrows={len(raw)} cols={raw.shape[1]}\n")
print("SHA-256:", sha256)
release

# %% [markdown]
# **Reading:** every attack class matches the official counts exactly; only the normal class differs
# (50,000 in our file vs. 300,000 officially). Our file is the official attack set with a reduced normal sample.
# Because false-positive rate and recall do not depend on the class mix, we report those directly and also
# re-weight precision to the official mix (34.9% attacks) so results stay comparable with the full release.

# %% [markdown]
# ## 2. De-duplicate before any split

# %%
FEATURES = [c for c in raw.columns if c not in ("label", "type", "src_ip", "dst_ip")]
row_hash = pd.util.hash_pandas_object(raw[FEATURES], index=False)
types_per_hash = raw.assign(h=row_hash).groupby("h")["type"].nunique()
conflict = row_hash.isin(types_per_hash[types_per_hash > 1].index)
keep = (~row_hash.duplicated()) & (~conflict)
df = raw[keep.values].reset_index(drop=True)
if SMOKE:
    df = df.groupby("type", group_keys=False).sample(frac=0.08, random_state=SEED).reset_index(drop=True)
dedup_summary = pd.DataFrame(
    [
        {"step": "raw rows", "rows": len(raw)},
        {"step": "exact feature duplicates removed", "rows": int(row_hash.duplicated().sum())},
        {"step": "conflicting-label rows removed", "rows": int((conflict & ~row_hash.duplicated()).sum())},
        {"step": "rows used for modelling", "rows": len(df)},
    ]
)
dedup_summary.to_csv(MET / "dedup_summary.csv", index=False)
print(dedup_summary.to_string(index=False))
print(df["type"].value_counts().to_string())

# %% [markdown]
# ## 3. Feature sets
# * `all_40` — the 40 non-identifier fields used in Report 1.
# * `no_ports` — drops `src_port` and `dst_port`.
# * `engineered` — no raw ports; adds port buckets (well-known / registered / ephemeral) and bytes-per-packet ratios.
#   This is the main feature set because it removes lab-specific port numbers.
# * `with_ips` — `all_40` plus `src_ip` and `dst_ip` (leakage demonstration only).

# %%
def port_bucket(ports):
    ports = np.asarray(ports)
    return np.select([ports < 1024, ports < 49152], ["well_known", "registered"], "ephemeral")


def add_engineered(frame):
    frame = frame.copy()
    frame["src_port_bucket"] = port_bucket(frame["src_port"])
    frame["dst_port_bucket"] = port_bucket(frame["dst_port"])
    frame["bytes_per_src_pkt"] = frame["src_ip_bytes"] / frame["src_pkts"].clip(lower=1)
    frame["bytes_per_dst_pkt"] = frame["dst_ip_bytes"] / frame["dst_pkts"].clip(lower=1)
    return frame


df = add_engineered(df)
NO_PORTS = [c for c in FEATURES if c not in ("src_port", "dst_port")]
FEATURE_SETS = {
    "all_40": FEATURES,
    "no_ports": NO_PORTS,
    "engineered": NO_PORTS + ["src_port_bucket", "dst_port_bucket", "bytes_per_src_pkt", "bytes_per_dst_pkt"],
    "with_ips": FEATURES + ["src_ip", "dst_ip"],
}
MAIN_SET = "engineered"


def split_types(cols, frame):
    cat = [c for c in cols if not pd.api.types.is_numeric_dtype(frame[c])]
    num = [c for c in cols if c not in cat]
    return cat, num


def tree_preprocessor(cols, frame):
    cat, num = split_types(cols, frame)
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)
    return ColumnTransformer([("cat", encoder, cat), ("num", "passthrough", num)])


def linear_preprocessor(cols, frame):
    cat, num = split_types(cols, frame)
    numeric = Pipeline([("log", FunctionTransformer(np.log1p)), ("scale", StandardScaler())])
    onehot = OneHotEncoder(handle_unknown="ignore", min_frequency=10)
    return ColumnTransformer([("cat", onehot, cat), ("num", numeric, num)])


def make_model(name, cols, frame, params=None):
    params = params or {}
    if name == "majority":
        return Pipeline([("pre", tree_preprocessor(cols, frame)), ("model", DummyClassifier(strategy="most_frequent"))])
    if name == "logreg":
        lr = LogisticRegression(max_iter=CONFIG["logreg"]["max_iter"], C=CONFIG["logreg"]["C"], class_weight="balanced")
        return Pipeline([("pre", linear_preprocessor(cols, frame)), ("model", lr)])
    if name == "random_forest":
        rf = RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=SEED, **params)
        return Pipeline([("pre", tree_preprocessor(cols, frame)), ("model", rf)])
    if name == "hist_gb":
        hgb = HistGradientBoostingClassifier(
            max_iter=200, early_stopping=False, class_weight="balanced", random_state=SEED, **params
        )
        return Pipeline([("pre", tree_preprocessor(cols, frame)), ("model", hgb)])
    raise ValueError(name)


# %% [markdown]
# ## 4. Evaluation helpers
# Binary metrics treat every attack family as "attack". The attack score is `1 − P(normal)`.
# `precision_official_mix` re-weights precision to the official release's class mix.

# %%
ATTACK_SHARE = CONFIG["official_attack_share"]


def binary_metrics(y_bin, pred_bin, score):
    tn, fp, fn, tp = confusion_matrix(y_bin, pred_bin, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if tp + fn else np.nan
    fpr = fp / (fp + tn) if fp + tn else np.nan
    precision = tp / (tp + fp) if tp + fp else np.nan
    denom = recall * ATTACK_SHARE + fpr * (1 - ATTACK_SHARE)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "fpr": fpr,
        "fnr": 1 - recall,
        "roc_auc": roc_auc_score(y_bin, score) if len(set(y_bin)) > 1 else np.nan,
        "pr_auc": average_precision_score(y_bin, score) if len(set(y_bin)) > 1 else np.nan,
        "precision_official_mix": recall * ATTACK_SHARE / denom if denom else np.nan,
    }


def evaluate(model, X, y):
    start = time.perf_counter()
    proba = model.predict_proba(X)
    latency = (time.perf_counter() - start) / len(X) * 1000 * 1000  # ms per 1,000 flows
    classes = np.asarray(model.classes_)
    pred = classes[proba.argmax(axis=1)]
    score = 1 - proba[:, list(classes).index("normal")]
    y_bin = (np.asarray(y) != "normal").astype(int)
    result = binary_metrics(y_bin, (pred != "normal").astype(int), score)
    present = sorted(set(y))
    report = classification_report(y, pred, labels=present, output_dict=True, zero_division=0)
    result.update(
        {
            "macro_f1": report["macro avg"]["f1-score"],
            "mitm_precision": report.get("mitm", {}).get("precision", np.nan),
            "mitm_recall": report.get("mitm", {}).get("recall", np.nan),
            "latency_ms_per_1k": latency,
        }
    )
    return result, pred, score


def log_numeric(frame, cols):
    cat, num = split_types(cols, frame)
    out = frame[cols].copy()
    out[num] = np.log1p(out[num].astype(float))
    return out


def run_isolation_forest(train, test, cols):
    """Unsupervised baseline: learn normal traffic only, flag the most unusual 5% of normal behaviour."""
    pre = tree_preprocessor(cols, train)
    tr = log_numeric(train, cols)
    te = log_numeric(test, cols)
    normal = train["type"].values == "normal"
    Xtr = pre.fit_transform(tr[normal])
    Xte = pre.transform(te)
    forest = IsolationForest(n_estimators=CONFIG["isolation_forest"]["n_estimators"], random_state=SEED, n_jobs=-1)
    start = time.perf_counter()
    forest.fit(Xtr)
    threshold = np.quantile(-forest.score_samples(Xtr), CONFIG["isolation_forest"]["normal_quantile_threshold"])
    t0 = time.perf_counter()
    score = -forest.score_samples(Xte)
    latency = (time.perf_counter() - t0) / len(Xte) * 1e6
    y_bin = (test["type"].values != "normal").astype(int)
    result = binary_metrics(y_bin, (score > threshold).astype(int), score)
    result.update({"macro_f1": np.nan, "mitm_precision": np.nan, "mitm_recall": np.nan, "latency_ms_per_1k": latency})
    return result, score


# %% [markdown]
# ## 5. Two evaluation designs
# * **Random 5-fold** — ordinary stratified folds (what Report 1 used).
# * **Host-grouped 5-fold** — `StratifiedGroupKFold` with `src_ip` as the group: every flow from a host is in the
#   same fold, so the model is always tested on machines it never saw in training.

# %%
y_all = df["type"].values
groups = df["src_ip"].values
random_folds = list(StratifiedKFold(n_splits=CONFIG["n_folds"], shuffle=True, random_state=SEED).split(df, y_all))
host_folds = list(
    StratifiedGroupKFold(n_splits=CONFIG["n_folds"], shuffle=True, random_state=SEED).split(df, y_all, groups)
)
SPLITS = {"random": random_folds, "host_grouped": host_folds}

fold_rows = []
for i, (tr, te) in enumerate(host_folds):
    test_attack_hosts = sorted(set(df.iloc[te].loc[lambda d: d["type"] != "normal", "src_ip"]))
    fold_rows.append(
        {
            "fold": i,
            "test_rows": len(te),
            "test_hosts": df.iloc[te]["src_ip"].nunique(),
            "test_attacker_hosts": ", ".join(test_attack_hosts),
            "classes_missing_from_train": ", ".join(sorted(set(y_all[te]) - set(y_all[tr]))) or "none",
            "shared_hosts_train_test": len(set(groups[tr]) & set(groups[te])),
        }
    )
host_fold_table = pd.DataFrame(fold_rows)
host_fold_table.to_csv(MET / "host_fold_composition.csv", index=False)
host_fold_table

# %% [markdown]
# ## 6. Hyperparameter tuning (validation only)
# Tuning uses only the **training part of host-grouped fold 0**, split again into 3 host-grouped inner folds.
# The candidate with the best mean inner macro-F1 is then used everywhere. Test folds are never used for tuning.

# %%
def tune(model_name, candidates):
    tr_idx = host_folds[0][0]
    inner = df.iloc[tr_idx].reset_index(drop=True)
    inner_folds = list(
        StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED).split(inner, inner["type"], inner["src_ip"])
    )
    rows = []
    cols = FEATURE_SETS[MAIN_SET]
    for k, params in enumerate(candidates):
        scores = []
        for a, b in inner_folds:
            model = make_model(model_name, cols, inner, params).fit(inner.iloc[a][cols], inner.iloc[a]["type"])
            pred = model.predict(inner.iloc[b][cols])
            present = sorted(set(inner.iloc[b]["type"]))
            scores.append(f1_score(inner.iloc[b]["type"], pred, labels=present, average="macro"))
        rows.append({"model": model_name, "candidate": k, **params, "inner_macro_f1_mean": np.mean(scores), "inner_macro_f1_sd": np.std(scores)})
    return pd.DataFrame(rows)


tuning = cached(
    "tuning",
    lambda: pd.concat(
        [tune("random_forest", CONFIG["random_forest_candidates"]), tune("hist_gb", CONFIG["hist_gb_candidates"])],
        ignore_index=True,
    ),
)
tuning.to_csv(MET / "tuning_results.csv", index=False)
BEST = {}
for name, cands in (("random_forest", CONFIG["random_forest_candidates"]), ("hist_gb", CONFIG["hist_gb_candidates"])):
    best_k = int(tuning[tuning.model == name].sort_values("inner_macro_f1_mean", ascending=False).iloc[0]["candidate"])
    BEST[name] = cands[best_k]
print(json.dumps(BEST, indent=2))
tuning

# %% [markdown]
# ## 7. Model comparison under both evaluation designs

# %%
def cross_validate(model_name, feature_set, split_name, keep_predictions=False):
    cols = FEATURE_SETS[feature_set]
    rows, oof_pred, oof_score = [], np.empty(len(df), dtype=object), np.zeros(len(df))
    for fold, (tr, te) in enumerate(SPLITS[split_name]):
        train, test = df.iloc[tr], df.iloc[te]
        start = time.perf_counter()
        if model_name == "isolation_forest":
            metrics, score = run_isolation_forest(train, test, cols)
            pred = np.full(len(te), "", dtype=object)
        else:
            model = make_model(model_name, cols, train, BEST.get(model_name)).fit(train[cols], train["type"])
            fit_seconds = time.perf_counter() - start
            metrics, pred, score = evaluate(model, test[cols], test["type"].values)
            metrics["fit_seconds"] = fit_seconds
        oof_pred[te], oof_score[te] = pred, score
        rows.append({"model": model_name, "features": feature_set, "split": split_name, "fold": fold, **metrics})
    out = {"folds": pd.DataFrame(rows)}
    if keep_predictions:
        out["pred"], out["score"] = oof_pred, oof_score
    return out


MODELS = ["majority", "logreg", "random_forest", "hist_gb", "isolation_forest"]
runs = {}
for split_name in SPLITS:
    for model_name in MODELS:
        key = f"{model_name}__{MAIN_SET}__{split_name}"
        runs[key] = cached(key, lambda m=model_name, s=split_name: cross_validate(m, MAIN_SET, s, keep_predictions=True))

fold_metrics = pd.concat([r["folds"] for r in runs.values()], ignore_index=True)
fold_metrics.to_csv(MET / "model_comparison_folds.csv", index=False)
METRIC_COLS = ["precision", "recall", "f1", "fpr", "roc_auc", "pr_auc", "precision_official_mix", "macro_f1", "mitm_recall", "latency_ms_per_1k"]
summary = fold_metrics.groupby(["split", "model"])[METRIC_COLS].agg(["mean", "std"]).round(4)
summary.to_csv(MET / "model_comparison_summary.csv")
summary

# %% [markdown]
# ## 8. Port and host-fingerprint ablation (Histogram Gradient Boosting)
# If performance collapses when hosts are held out, or depends on raw port numbers, the model is partly
# recognising machines and lab ports instead of attack behaviour.

# %%
ablation_frames = []
for feature_set in ("all_40", "no_ports", "engineered", "with_ips"):
    for split_name in SPLITS:
        key = f"hist_gb__{feature_set}__{split_name}"
        result = runs.get(key) or cached(key, lambda f=feature_set, s=split_name: cross_validate("hist_gb", f, s))
        ablation_frames.append(result["folds"])
ablation = pd.concat(ablation_frames, ignore_index=True)
ablation_summary = ablation.groupby(["features", "split"])[["macro_f1", "recall", "fpr", "mitm_recall"]].agg(["mean", "std"]).round(4)
ablation_summary.to_csv(MET / "fingerprint_ablation.csv")
ablation_summary

# %% [markdown]
# ## 9. Error analysis (host-grouped, Random Forest — best model on unseen hosts — out-of-fold predictions)

# %%
main = runs[f"random_forest__{MAIN_SET}__host_grouped"]
err = df[["src_ip", "type", "proto", "service", "conn_state", "duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts", "dst_port"]].copy()
err["pred"] = main["pred"]
err["attack_score"] = main["score"]
labels = sorted(df["type"].unique())
cm = pd.DataFrame(confusion_matrix(err["type"], err["pred"], labels=labels), index=labels, columns=labels)
cm.to_csv(MET / "confusion_matrix_host_grouped_rf.csv")

false_pos = err[(err["type"] == "normal") & (err["pred"] != "normal")]
false_neg = err[(err["type"] != "normal") & (err["pred"] == "normal")]
fp_patterns = false_pos.groupby(["pred", "proto", "service", "conn_state"]).size().sort_values(ascending=False).head(8).rename("flows").reset_index()
fn_patterns = false_neg.groupby(["type", "proto", "service", "conn_state"]).size().sort_values(ascending=False).head(8).rename("flows").reset_index()
fp_patterns.to_csv(MET / "false_positive_patterns.csv", index=False)
fn_patterns.to_csv(MET / "false_negative_patterns.csv", index=False)
confusions = err[err["type"] != err["pred"]].groupby(["type", "pred"]).size().sort_values(ascending=False).head(12).rename("flows").reset_index()
confusions.to_csv(MET / "top_confusions.csv", index=False)
print("False positives:", len(false_pos), "of", int((err["type"] == "normal").sum()), "normal flows")
print("False negatives:", len(false_neg), "of", int((err["type"] != "normal").sum()), "attack flows")
print(fp_patterns.to_string(index=False))
print(fn_patterns.to_string(index=False))
print(confusions.to_string(index=False))

# %% [markdown]
# ## 10. Per-group performance (responsible-AI check)
# There are no demographic attributes in network flows, so we check for unequal performance across
# attack type, protocol, and individual source hosts (devices).

# %%
per_class = pd.DataFrame(classification_report(err["type"], err["pred"], output_dict=True, zero_division=0)).T.loc[labels, ["precision", "recall", "f1-score", "support"]]
per_class.to_csv(MET / "per_class_host_grouped_rf.csv")

normal_rows = err[err["type"] == "normal"]
fpr_by_proto = normal_rows.groupby("proto").apply(lambda g: pd.Series({"normal_flows": len(g), "fpr": (g["pred"] != "normal").mean()}))
fpr_by_host = normal_rows.groupby("src_ip").apply(lambda g: pd.Series({"normal_flows": len(g), "fpr": (g["pred"] != "normal").mean()}))
fpr_by_host = fpr_by_host[fpr_by_host["normal_flows"] >= 100].sort_values("fpr", ascending=False)
detect_by_host = err[err["type"] != "normal"].groupby("src_ip").apply(lambda g: pd.Series({"attack_flows": len(g), "detection_rate": (g["pred"] != "normal").mean()}))
fpr_by_proto.to_csv(MET / "fpr_by_protocol.csv")
fpr_by_host.to_csv(MET / "fpr_by_normal_host.csv")
detect_by_host.to_csv(MET / "detection_by_attacker_host.csv")
print(per_class.round(3).to_string())
print(fpr_by_proto.round(4).to_string())
print(fpr_by_host.head(10).round(4).to_string())
print(detect_by_host.round(4).to_string())

# %% [markdown]
# ## 11. Explainability: SHAP and explanation stability
# A binary Histogram Gradient Boosting model (attack vs. normal) is trained on each host-grouped training fold.
# We compute mean |SHAP| on 2,000 test flows per fold and check how stable the top-10 features are across folds.

# %%
def shap_by_fold():
    import shap

    cols = FEATURE_SETS[MAIN_SET]
    tops, importances = [], []
    for fold, (tr, te) in enumerate(host_folds):
        train, test = df.iloc[tr], df.iloc[te]
        pre = tree_preprocessor(cols, train)
        Xtr = pre.fit_transform(train[cols])
        names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
        model = HistGradientBoostingClassifier(max_iter=200, early_stopping=False, class_weight="balanced", random_state=SEED, **BEST["hist_gb"])
        model.fit(Xtr, (train["type"] != "normal").astype(int))
        sample = test.sample(min(2000, len(test)), random_state=SEED)
        values = shap.TreeExplainer(model).shap_values(pre.transform(sample[cols]))
        values = values[1] if isinstance(values, list) else values
        mean_abs = pd.Series(np.abs(values).mean(axis=0), index=names, name=f"fold_{fold}")
        importances.append(mean_abs)
        tops.append(set(mean_abs.sort_values(ascending=False).head(10).index))
    jaccard = [len(a & b) / len(a | b) for a, b in itertools.combinations(tops, 2)]
    return pd.concat(importances, axis=1), jaccard


shap_table, shap_jaccard = cached("shap", shap_by_fold)
shap_table["mean"] = shap_table.mean(axis=1)
shap_table = shap_table.sort_values("mean", ascending=False)
shap_table.to_csv(MET / "shap_mean_abs_by_fold.csv")
stability = pd.DataFrame([{"top_k": 10, "mean_pairwise_jaccard": np.mean(shap_jaccard), "min_pairwise_jaccard": np.min(shap_jaccard)}])
stability.to_csv(MET / "shap_stability.csv", index=False)
print(shap_table.head(12).round(4).to_string())
print(stability.round(3).to_string(index=False))

# %% [markdown]
# ## 12. Adversarial evasion test
# **Threat:** an attacker who controls their own traffic tries to make attack flows look normal.
# They cannot change the target service or how the victim responds, but they can:
# * **pad** payloads (+64 to +1,024 bytes, spread over extra packets),
# * **slow down** (stretch duration 2–10×),
# * **randomise the source port** (ephemeral range),
# * or **combine** all three.
#
# We take attack flows that the model detects on a held-out test set, perturb them, and measure how many are now
# classified as normal (evasion rate). **Defense:** adversarial training — add perturbed copies of 30% of the
# training attacks (with their true labels) and retrain.

# %%
ADV = CONFIG["adversarial"]


def perturb(frame, kind, rng):
    frame = frame.copy()
    n = len(frame)
    if kind in ("pad", "combined"):
        pad = rng.integers(ADV["pad_bytes_min"], ADV["pad_bytes_max"] + 1, n)
        extra_pkts = np.ceil(pad / 1460).astype(int)
        frame["src_bytes"] = frame["src_bytes"] + pad
        frame["src_pkts"] = frame["src_pkts"] + extra_pkts
        frame["src_ip_bytes"] = frame["src_ip_bytes"] + pad + 40 * extra_pkts
    if kind in ("slow", "combined"):
        frame["duration"] = frame["duration"] * rng.uniform(ADV["slowdown_min"], ADV["slowdown_max"], n)
    if kind in ("sport", "combined"):
        frame["src_port"] = rng.integers(49152, 65536, n)
    return add_engineered(frame)


def adversarial_experiment():
    train, test = train_test_split(df, test_size=0.2, stratify=df["type"], random_state=SEED)
    rng = np.random.default_rng(SEED)
    attacks = train[train["type"] != "normal"]
    sampled = attacks.sample(frac=ADV["augment_fraction"], random_state=SEED)
    kinds = rng.choice(["pad", "slow", "sport", "combined"], len(sampled))
    augmented = pd.concat([perturb(sampled[kinds == k], k, rng) for k in np.unique(kinds)])
    train_aug = pd.concat([train, augmented], ignore_index=True)

    variants = {
        "HGB, raw ports (all_40)": ("all_40", train),
        "HGB, engineered features": ("engineered", train),
        "HGB, engineered + adversarial training": ("engineered", train_aug),
    }
    rows = []
    for label, (feature_set, training) in variants.items():
        cols = FEATURE_SETS[feature_set]
        model = make_model("hist_gb", cols, training, BEST["hist_gb"]).fit(training[cols], training["type"])
        clean, _, _ = evaluate(model, test[cols], test["type"].values)
        detected = test[(test["type"] != "normal") & (model.predict(test[cols]) != "normal")]
        row = {"model": label, "clean_macro_f1": clean["macro_f1"], "clean_fpr": clean["fpr"], "clean_recall": clean["recall"]}
        for kind in ("pad", "slow", "sport", "combined"):
            moved = perturb(detected, kind, np.random.default_rng(SEED + 1))
            row[f"evasion_{kind}"] = float((model.predict(moved[cols]) == "normal").mean())
        rows.append(row)
    return pd.DataFrame(rows)


adversarial = cached("adversarial", adversarial_experiment)
adversarial.to_csv(MET / "adversarial_evasion.csv", index=False)
adversarial.round(4)

# %% [markdown]
# ## 13. Generalization to an external dataset (UNSW-NB15)
# UNSW-NB15 is a different testbed (2015, Argus + Bro tooling, different attack taxonomy). Only a handful of
# fields can be mapped onto TON_IoT's schema, so we train a **shared-schema** binary model:
# protocol, service, connection state, duration, packets and IP-level bytes in each direction, and bytes per packet.
# * UNSW `sbytes`/`dbytes` include headers → mapped to `src_ip_bytes`/`dst_ip_bytes`.
# * Argus states are mapped approximately to Zeek states (FIN/CLO→SF, REQ→S0, CON/ACC→S1, RST→RSTO, INT/other→OTH).
# * Attack categories do not line up (Fuzzers, Exploits, Generic, …), so only attack vs. normal is evaluated.

# %%
UNSW_PROTO = {4: "tcp", 2: "udp"}
UNSW_SERVICE = {999: "-", 2: "http", 3: "ftp", 4: "ftp", 7: "dns", 9: "ssl"}
UNSW_STATE = {2: "SF", 7: "SF", 3: "S0", 5: "S1", 4: "S1", 6: "RSTO", 1: "OTH"}
SHARED = ["proto", "service", "conn_state", "duration", "src_pkts", "dst_pkts", "src_ip_bytes", "dst_ip_bytes", "bytes_per_src_pkt", "bytes_per_dst_pkt"]


def load_unsw(name):
    u = pd.read_csv(EXT_DIR / name)
    out = pd.DataFrame(
        {
            "proto": u["xProt"].map(UNSW_PROTO).fillna("other"),
            "service": u["xServ"].map(UNSW_SERVICE).fillna("other"),
            "conn_state": u["xState"].map(UNSW_STATE).fillna("OTH"),
            "duration": u["dur"],
            "src_pkts": u["spkts"],
            "dst_pkts": u["dpkts"],
            "src_ip_bytes": u["sbytes"],
            "dst_ip_bytes": u["dbytes"],
        }
    )
    out["bytes_per_src_pkt"] = out["src_ip_bytes"] / out["src_pkts"].clip(lower=1)
    out["bytes_per_dst_pkt"] = out["dst_ip_bytes"] / out["dst_pkts"].clip(lower=1)
    out["is_attack"] = u["label"].astype(int)
    return out


def binary_shared_model(train_frame, y):
    return make_model("hist_gb", SHARED, train_frame, BEST["hist_gb"]).fit(train_frame[SHARED], y)


def binary_eval(model, frame, y):
    score = model.predict_proba(frame[SHARED])[:, 1]
    return binary_metrics(np.asarray(y), (score >= 0.5).astype(int), score)


def generalization_experiment():
    unsw_a = load_unsw("UNSW_NB15_training-set.csv")  # 82,332 rows
    unsw_b = load_unsw("UNSW_NB15_testing-set.csv")   # 175,341 rows (external test set)
    ton_y = (df["type"] != "normal").astype(int).values
    rows = []
    # In-domain reference: TON_IoT shared-schema model, host-grouped folds
    fold_scores = []
    for tr, te in host_folds:
        model = binary_shared_model(df.iloc[tr], ton_y[tr])
        fold_scores.append(binary_eval(model, df.iloc[te], ton_y[te]))
    rows.append({"setting": "TON_IoT → TON_IoT (host-grouped CV)", "test_rows": len(df), **pd.DataFrame(fold_scores).mean().to_dict()})
    # Zero-shot transfer
    ton_model = binary_shared_model(df, ton_y)
    rows.append({"setting": "TON_IoT → UNSW-NB15 (no adaptation)", "test_rows": len(unsw_b), **binary_eval(ton_model, unsw_b, unsw_b["is_attack"])})
    # Light adaptation: add 5% of labelled UNSW file A
    few = unsw_a.sample(frac=CONFIG["external_adaptation_fraction"], random_state=SEED)
    mixed = pd.concat([df[SHARED].assign(is_attack=ton_y), few], ignore_index=True)
    adapted = binary_shared_model(mixed, mixed["is_attack"].values)
    rows.append({"setting": f"TON_IoT + {int(CONFIG['external_adaptation_fraction']*100)}% UNSW labels → UNSW-NB15", "test_rows": len(unsw_b), **binary_eval(adapted, unsw_b, unsw_b["is_attack"])})
    # In-domain ceiling for UNSW
    ceiling = binary_shared_model(unsw_a, unsw_a["is_attack"].values)
    rows.append({"setting": "UNSW-NB15 → UNSW-NB15 (reference ceiling)", "test_rows": len(unsw_b), **binary_eval(ceiling, unsw_b, unsw_b["is_attack"])})

    schema = pd.DataFrame(
        [
            {"item": "rows", "ton_iot": len(raw), "unsw_nb15": len(unsw_a) + len(unsw_b)},
            {"item": "attack share", "ton_iot": round(float((raw["type"] != "normal").mean()), 3), "unsw_nb15": round(float(pd.concat([unsw_a, unsw_b])["is_attack"].mean()), 3)},
            {"item": "original feature columns", "ton_iot": len(FEATURES) + 2, "unsw_nb15": 43},
            {"item": "shared (mappable) features", "ton_iot": len(SHARED), "unsw_nb15": len(SHARED)},
        ]
    )
    return pd.DataFrame(rows), schema, unsw_b


generalization, schema_table, unsw_b = cached("generalization", generalization_experiment)
generalization.to_csv(MET / "generalization_external.csv", index=False)
schema_table.to_csv(MET / "external_schema_comparison.csv", index=False)
print(schema_table.to_string(index=False))
generalization.round(4)

# %% [markdown]
# ## 14. Drift check (Population Stability Index)
# PSI compares a feature's distribution in new traffic with training traffic. Rule of thumb:
# < 0.1 stable, 0.1–0.25 moderate shift, > 0.25 major shift (investigate / retrain).

# %%
def psi(reference, current, bins=10):
    reference, current = np.log1p(np.asarray(reference, float)), np.log1p(np.asarray(current, float))
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, edges)[0] / len(reference)
    cur = np.histogram(current, edges)[0] / len(current)
    ref, cur = np.clip(ref, 1e-6, None), np.clip(cur, 1e-6, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


tr0, te0 = host_folds[0]
reference = df.iloc[tr0]
psi_rows = []
for col in ["duration", "src_pkts", "dst_pkts", "src_ip_bytes", "dst_ip_bytes", "bytes_per_src_pkt"]:
    psi_rows.append(
        {
            "feature": col,
            "psi_unseen_TON_hosts": psi(reference[col], df.iloc[te0][col]),
            "psi_UNSW_NB15": psi(reference[col], unsw_b[col]),
        }
    )
psi_table = pd.DataFrame(psi_rows)
psi_table.to_csv(MET / "drift_psi.csv", index=False)
psi_table.round(3)

# %% [markdown]
# ## 15. Figures for the report

# %%
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10, "axes.titleweight": "bold"})
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#9a9a9a"

# Figure 1: random vs host-grouped macro-F1, plus fingerprint ablation
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.15, 1]})
names = {"majority": "Majority", "logreg": "Logistic\nRegression", "random_forest": "Random\nForest", "hist_gb": "Gradient\nBoosting"}
x = np.arange(len(names))
for offset, (split_name, color, label) in enumerate([("random", GREY, "Random 5-fold"), ("host_grouped", BLUE, "Host-grouped 5-fold")]):
    sub = fold_metrics[fold_metrics["split"] == split_name].groupby("model")["macro_f1"]
    means = [sub.mean()[m] for m in names]
    sds = [sub.std()[m] for m in names]
    axes[0].bar(x + (offset - 0.5) * 0.38, means, 0.36, yerr=sds, color=color, label=label, capsize=2, error_kw={"elinewidth": 0.8})
    for xi, m, sd in zip(x, means, sds):
        axes[0].text(xi + (offset - 0.5) * 0.38, m + sd + 0.02, f"{m:.2f}", ha="center", fontsize=7)
axes[0].set_xticks(x, list(names.values()), fontsize=7.5)
axes[0].set_ylim(0, 1.18)
axes[0].set_ylabel("10-class macro-F1 (mean ± SD)")
axes[0].set_title("(a) Models: random vs. unseen hosts", loc="left")
axes[0].legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.95, -0.2), ncol=2)

set_names = {"with_ips": "+ IP addresses", "all_40": "Raw ports", "no_ports": "No ports", "engineered": "Bucketed ports\n+ ratios (main)"}
y = np.arange(len(set_names))
for offset, (split_name, color) in enumerate([("random", GREY), ("host_grouped", BLUE)]):
    sub = ablation[ablation["split"] == split_name].groupby("features")["macro_f1"]
    vals = [sub.mean()[k] for k in set_names]
    errs = [sub.std()[k] for k in set_names]
    axes[1].barh(y + (offset - 0.5) * 0.38, vals, 0.36, color=color, xerr=errs, error_kw={"elinewidth": 0.8})
    for yi, v, e in zip(y, vals, errs):
        axes[1].text(v + e + 0.02, yi + (offset - 0.5) * 0.38, f"{v:.2f}", va="center", fontsize=7)
axes[1].set_yticks(y, list(set_names.values()), fontsize=7.5)
axes[1].set_xlim(0, 1.2)
axes[1].invert_yaxis()
axes[1].set_xlabel("10-class macro-F1 (Gradient Boosting)")
axes[1].set_title("(b) Fingerprint ablation", loc="left")
fig.tight_layout()
fig.savefig(FIG / "fig1_model_and_ablation.png", dpi=220)
plt.close(fig)

# Figure 2: normalised confusion matrix (host-grouped, HGB)
norm = cm.div(cm.sum(axis=1), axis=0)
fig, ax = plt.subplots(figsize=(4.9, 3.7))
im = ax.imshow(norm.values, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=7.5)
ax.set_yticks(range(len(labels)), labels, fontsize=7.5)
for i in range(len(labels)):
    for j in range(len(labels)):
        v = norm.values[i, j]
        if v >= 0.02:
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=6.5, color="white" if v > 0.55 else "#222")
ax.set_xlabel("Predicted class")
ax.set_ylabel("True class")
ax.set_title("Unseen hosts, Random Forest (row %)", fontsize=9)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cb.ax.tick_params(labelsize=7)
fig.tight_layout()
fig.savefig(FIG / "fig2_confusion_host_grouped.png", dpi=220)
plt.close(fig)
print("Figures saved to", FIG.relative_to(PROJECT_ROOT))

# %% [markdown]
# ## 16. Operating point: recall when false alarms are capped at 1%
# ROC-AUC stays near 1.0 on unseen hosts while the default 0.5 threshold produces many false alarms.
# That means the ranking is good but the threshold is wrong. Below: attack recall at a 1% false-positive
# budget, per fold. (Choosing the actual threshold on validation hosts is a final-phase task.)

# %%
from sklearn.metrics import roc_curve


def recall_at_fpr(y_bin, score, budget=0.01):
    fpr, tpr, _ = roc_curve(y_bin, score)
    return float(tpr[fpr <= budget].max()) if (fpr <= budget).any() else 0.0


op_rows = []
for split_name, folds in SPLITS.items():
    for model_name in ("logreg", "random_forest", "hist_gb", "isolation_forest"):
        scores = runs[f"{model_name}__{MAIN_SET}__{split_name}"]["score"]
        for fold, (_, te) in enumerate(folds):
            y_bin = (y_all[te] != "normal").astype(int)
            op_rows.append({"split": split_name, "model": model_name, "fold": fold, "recall_at_1pct_fpr": recall_at_fpr(y_bin, scores[te])})
operating_point = pd.DataFrame(op_rows)
op_summary = operating_point.groupby(["split", "model"])["recall_at_1pct_fpr"].agg(["mean", "std", "min"]).round(4)
op_summary.to_csv(MET / "recall_at_1pct_fpr.csv")
op_summary
