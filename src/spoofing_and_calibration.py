# %% [markdown]
# # 06 — MITM Spoofing Detection and Alert-Threshold Calibration
#
# **Why this notebook exists.** Report 3 showed two gaps:
# 1. On unseen devices, the default 0.5 threshold raises false alarms on ~19% of normal flows,
#    even though ROC-AUC stays near 0.999 (the ranking is good; the threshold is wrong).
# 2. MITM is the rarest class (1,039 flows) and the instructor asked whether spoofing can be used to detect it.
#
# **What we build.**
# * Four **spoofing detectors** that look for the footprints of a man-in-the-middle attack chain:
#   forged routes (ICMP Redirect, TTL-exceeded), forged DNS answers (rogue resolver), and stripped encryption
#   (TLS stripping). They use flow behaviour only, never IP addresses.
# * A **hybrid detector**: the tuned Random Forest (no raw ports) plus the four rule flags as features,
#   with high-specificity rules able to escalate an alert on their own.
# * **Threshold calibration on validation hosts**: inside each training fold, an inner host-grouped split
#   picks the score threshold that keeps false alarms at 1%; the outer test fold is never touched.
#
# Evaluation uses the same host-grouped 5-fold split as notebook 05 (no device in both training and test).

# %%
from pathlib import Path
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name in ("notebooks", "src") else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from ids_common import (  # noqa: E402
    HIGH_SPECIFICITY_RULES,
    RULE_COLUMNS,
    SEED,
    SPOOF_RULES,
    attack_score,
    decide,
    feature_sets,
    load_dedup,
    random_forest,
)

CONFIG = json.loads((PROJECT_ROOT / "configs" / "status_report_config.json").read_text())
OUT = PROJECT_ROOT / "results" / "spoofing"
MET, FIG = OUT / "metrics", OUT / "figures"
for folder in (MET, FIG):
    folder.mkdir(parents=True, exist_ok=True)
TARGET_FPR = 0.01

df, FEATURES = load_dedup(PROJECT_ROOT / CONFIG["data_path"])
SETS = feature_sets(FEATURES)
y = df["type"].values
host_folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(df, y, df["src_ip"].values))
print("Flows:", len(df), "| MITM flows:", int((y == "mitm").sum()))

# %% [markdown]
# ## 1. How often does each spoofing detector fire?
# Rules are fixed (no training), so they can be measured on all flows. "Other attacks" are non-MITM attack flows:
# those are still correct attack alerts, just not MITM.

# %%
rule_rows = []
mitm, normal = y == "mitm", y == "normal"
any_rule = df[RULE_COLUMNS].sum(axis=1).values > 0
for col in RULE_COLUMNS + ["any_rule"]:
    hit = any_rule if col == "any_rule" else df[col].values == 1
    meta = SPOOF_RULES.get(col.replace("rule_", ""), {"title": "Any spoofing rule", "layer": "All"})
    rule_rows.append(
        {
            "rule": meta["title"],
            "layer": meta["layer"],
            "flows_flagged": int(hit.sum()),
            "mitm_caught": int((hit & mitm).sum()),
            "mitm_recall": (hit & mitm).sum() / mitm.sum(),
            "normal_flagged": int((hit & normal).sum()),
            "normal_fpr": (hit & normal).sum() / normal.sum(),
            "other_attacks_flagged": int((hit & ~mitm & ~normal).sum()),
        }
    )
rule_table = pd.DataFrame(rule_rows)
rule_table.to_csv(MET / "spoof_rule_hits.csv", index=False)
rule_table.round(4)

# %% [markdown]
# **Sanity check for the DNS rule.** It assumes the network's normal resolver does not advertise recursion.
# We confirm this from *training* normal traffic in every fold (not from test data).

# %%
dns_baseline = []
for fold, (tr, _) in enumerate(host_folds):
    train_normal_dns = df.iloc[tr].query("type == 'normal' and service == 'dns'")
    dns_baseline.append({"fold": fold, "normal_dns_replies": len(train_normal_dns), "share_with_RA_flag": float((train_normal_dns["dns_RA"] == "T").mean())})
dns_baseline = pd.DataFrame(dns_baseline)
dns_baseline.to_csv(MET / "dns_resolver_baseline.csv", index=False)
dns_baseline.round(4)

# %% [markdown]
# ## 2. Rules vs. ML vs. hybrid, with and without calibrated thresholds

# %%
def calibrate_threshold(train: pd.DataFrame, cols: list[str]) -> float:
    """Pick the attack-score threshold that gives TARGET_FPR on held-out *training* hosts."""
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train))
    for a, b in inner.split(train, train["type"], train["src_ip"]):
        model = random_forest(cols, train.iloc[a], n_estimators=100).fit(train.iloc[a][cols], train.iloc[a]["type"])
        oof[b] = attack_score(model, train.iloc[b][cols])[0]
    normal_scores = oof[train["type"].values == "normal"]
    return float(np.quantile(normal_scores, 1 - TARGET_FPR))


def score_predictions(y_true, predicted):
    y_true, predicted = np.asarray(y_true), np.asarray(predicted)
    present = sorted(set(y_true))
    rep = classification_report(y_true, predicted, labels=present, output_dict=True, zero_division=0)
    is_attack_true, is_attack_pred = y_true != "normal", predicted != "normal"
    return {
        "macro_f1": rep["macro avg"]["f1-score"],
        "mitm_precision": rep.get("mitm", {}).get("precision", np.nan),
        "mitm_recall": rep.get("mitm", {}).get("recall", np.nan),
        "mitm_f1": rep.get("mitm", {}).get("f1-score", np.nan),
        "attack_recall": (is_attack_pred & is_attack_true).sum() / is_attack_true.sum(),
        "fpr": (is_attack_pred & ~is_attack_true).sum() / (~is_attack_true).sum(),
    }


rows, oof_final = [], pd.DataFrame(index=df.index, columns=["fold", "attack_score", "predicted", "threshold"])
pooled = {}
for fold, (tr, te) in enumerate(host_folds):
    start = time.perf_counter()
    train, test = df.iloc[tr], df.iloc[te]
    rules_te = test[RULE_COLUMNS]
    no_rules = rules_te * 0
    results = {}

    # (a) ML only, default threshold
    ml = random_forest(SETS["no_ports"], train).fit(train[SETS["no_ports"]], train["type"])
    results["ML only (RF, no ports), threshold 0.5"] = decide(ml, test[SETS["no_ports"]], no_rules, 0.5)["predicted"]

    # (b) Rules only: any rule -> MITM, else normal
    results["Spoofing rules only"] = np.where(rules_te.sum(axis=1) > 0, "mitm", "normal")

    # (c) Hybrid, default threshold
    hyb = random_forest(SETS["hybrid"], train).fit(train[SETS["hybrid"]], train["type"])
    results["Hybrid (RF + rule features), threshold 0.5"] = decide(hyb, test[SETS["hybrid"]], rules_te, 0.5)["predicted"]

    # (d) ML only, calibrated threshold
    t_ml = calibrate_threshold(train, SETS["no_ports"])
    results["ML only, calibrated threshold"] = decide(ml, test[SETS["no_ports"]], no_rules, t_ml)["predicted"]

    # (e) Hybrid, calibrated threshold  = final system
    t_h = calibrate_threshold(train, SETS["hybrid"])
    final = decide(hyb, test[SETS["hybrid"]], rules_te, t_h)
    results["Hybrid, calibrated threshold (final system)"] = final["predicted"]
    oof_final.loc[test.index, ["fold", "attack_score", "predicted", "threshold"]] = np.c_[np.full(len(te), fold), final["attack_score"], final["predicted"], np.full(len(te), t_h)]

    for name, pred in results.items():
        pooled.setdefault(name, np.empty(len(df), dtype=object))[te] = np.asarray(pred)
        rows.append({"approach": name, "fold": fold, "threshold": {"ML only, calibrated threshold": t_ml, "Hybrid, calibrated threshold (final system)": t_h}.get(name, 0.5), **score_predictions(test["type"], pred)})
    print(f"fold {fold}: thresholds ML={t_ml:.3f} hybrid={t_h:.3f}  ({time.perf_counter() - start:.0f}s)")

comparison = pd.DataFrame(rows)
comparison.to_csv(MET / "spoofing_comparison_folds.csv", index=False)
ORDER = list(results)
summary = comparison.groupby("approach")[["macro_f1", "mitm_precision", "mitm_recall", "mitm_f1", "attack_recall", "fpr", "threshold"]].agg(["mean", "std"]).reindex(ORDER).round(4)
summary.to_csv(MET / "spoofing_comparison_summary.csv")

# Pooled view: every flow is predicted exactly once (by the fold that held its host out).
# For a rare class like MITM this is steadier than averaging folds, one of which has no MITM at all.
pooled_table = pd.DataFrame([{"approach": name, **score_predictions(y, pooled[name])} for name in ORDER])
pooled_table.to_csv(MET / "spoofing_comparison_pooled.csv", index=False)
pooled_table.round(4)

# %% [markdown]
# ## 3. Final system on unseen hosts: confusion matrix and which rule caught which MITM flow

# %%
oof_final["true"] = y
labels = sorted(df["type"].unique())
cm = pd.DataFrame(confusion_matrix(oof_final["true"], oof_final["predicted"].astype(str), labels=labels), index=labels, columns=labels)
cm.to_csv(MET / "final_system_confusion.csv")

mitm_rows = df[mitm]
mitm_pred = oof_final.loc[mitm, "predicted"].astype(str)
caught_by = pd.DataFrame(
    {
        "caught_by_rules": mitm_rows[RULE_COLUMNS].sum(axis=1).values > 0,
        "labelled_mitm_by_system": (mitm_pred == "mitm").values,
        "flagged_as_attack": (mitm_pred != "normal").values,
    }
)
coverage = pd.DataFrame(
    [
        {"group": "MITM flows", "count": len(caught_by)},
        {"group": "flagged as an attack by the final system", "count": int(caught_by["flagged_as_attack"].sum())},
        {"group": "labelled MITM by the final system", "count": int(caught_by["labelled_mitm_by_system"].sum())},
        {"group": "hit by at least one spoofing rule", "count": int(caught_by["caught_by_rules"].sum())},
    ]
)
coverage.to_csv(MET / "mitm_coverage.csv", index=False)
print(coverage.to_string(index=False))

# Example flows per rule (for the report and the demo app)
examples = []
for col in RULE_COLUMNS:
    hit = df[(df[col] == 1) & mitm]
    if len(hit):
        row = hit.iloc[0]
        examples.append({"rule": SPOOF_RULES[col.replace("rule_", "")]["title"], "proto": row["proto"], "service": row["service"], "conn_state": row["conn_state"],
                         "icmp_type_or_src_port": row["src_port"], "dns_RA": row["dns_RA"], "dns_rcode": row["dns_rcode"], "ssl_version": row["ssl_version"],
                         "src_bytes": row["src_bytes"], "dst_bytes": row["dst_bytes"]})
pd.DataFrame(examples).to_csv(MET / "spoof_rule_examples.csv", index=False)
pd.DataFrame(examples)

# %% [markdown]
# ## 4. Figure for the report

# %%
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10, "axes.titleweight": "bold"})
short = {
    "ML only (RF, no ports), threshold 0.5": "ML only\n(t = 0.5)",
    "Spoofing rules only": "Spoofing\nrules only",
    "Hybrid (RF + rule features), threshold 0.5": "Hybrid\n(t = 0.5)",
    "ML only, calibrated threshold": "ML only\n(calibrated)",
    "Hybrid, calibrated threshold (final system)": "Hybrid\n(calibrated)",
}
means = pooled_table.set_index("approach").reindex(ORDER)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
x = np.arange(len(ORDER))
colors = ["#9a9a9a", "#1baf7a", "#2a78d6", "#9a9a9a", "#2a78d6"]
for ax, metric, title in ((axes[0], "mitm_f1", "(a) MITM F1, unseen hosts"), (axes[1], "fpr", "(b) False alarms, unseen hosts")):
    vals = means[metric].values * (100 if metric == "fpr" else 1)
    ax.bar(x, vals, 0.62, color=colors)
    for xi, v in zip(x, vals):
        ax.text(xi, v + (0.02 if metric != "fpr" else 0.4), f"{v:.2f}" if metric != "fpr" else f"{v:.1f}%", ha="center", fontsize=7)
    ax.set_xticks(x, [short[k] for k in ORDER], fontsize=7)
    ax.set_title(title, loc="left")
axes[0].set_ylim(0, 1.0)
axes[0].set_ylabel("MITM F1 (pooled over 5 folds)")
axes[1].set_ylabel("% of normal flows alerted")
fig.tight_layout()
fig.savefig(FIG / "fig3_spoofing_hybrid.png", dpi=220)
plt.close(fig)
print("Saved", (FIG / "fig3_spoofing_hybrid.png").relative_to(PROJECT_ROOT))
