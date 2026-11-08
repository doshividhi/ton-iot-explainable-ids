"""Build everything the Streamlit demo needs (run once, about 3 minutes).

    python app/build_demo_assets.py

Creates app/assets/ with:
  * fold_model_{k}.joblib  - hybrid Random Forest trained on host-grouped fold k (its test hosts are unseen)
  * replay_flows.pkl       - ~8,000 unseen-host flows, each pre-scored by the model that never saw its host
  * sim_standard.joblib / sim_adversarial.joblib + sim_pool.pkl - attack simulator models and example attacks
  * drift_reference.json   - training-traffic baseline for the drift monitor (and UNSW-NB15 PSI if available)
  * manifest.json          - thresholds and summary numbers
Needs data/train_test_network.csv (and optionally data/external/unsw_nb15/*.csv).
"""
from pathlib import Path
import json
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from ids_common import RULE_COLUMNS, SEED, feature_sets, load_dedup, perturb, psi, random_forest, tree_preprocessor  # noqa: E402

ASSETS = APP_DIR / "assets"
ASSETS.mkdir(exist_ok=True)
DATA = PROJECT_ROOT / "data" / "train_test_network.csv"
DRIFT_FEATURES = ["duration", "src_pkts", "dst_pkts", "src_ip_bytes", "dst_ip_bytes", "src_bytes", "dst_bytes"]
SHOW_COLUMNS = ["src_ip", "dst_ip", "proto", "service", "conn_state", "src_port", "dst_port", "duration", "src_bytes",
                "dst_bytes", "src_pkts", "dst_pkts", "src_ip_bytes", "dst_ip_bytes", "dns_query", "dns_RA", "dns_rcode",
                "ssl_version", "ssl_subject", "http_method", "http_uri"]

start = time.perf_counter()
if not DATA.exists():
    sys.exit(f"Missing {DATA}. Put train_test_network.csv in the data/ folder first.")
df, FEATURES = load_dedup(DATA)
SETS = feature_sets(FEATURES)
COLS = SETS["hybrid"]
y = df["type"].values
folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(df, y, df["src_ip"].values))
print(f"Loaded {len(df):,} de-duplicated flows")

# Calibrated thresholds from notebook 06 (per fold); fall back to 0.9 if the notebook has not been run.
thr_file = PROJECT_ROOT / "results" / "spoofing" / "metrics" / "spoofing_comparison_folds.csv"
thresholds = {k: 0.9 for k in range(5)}
if thr_file.exists():
    t = pd.read_csv(thr_file)
    t = t[t["approach"].str.contains("final system")]
    thresholds = {int(r.fold): float(r.threshold) for r in t.itertuples()}
print("Calibrated thresholds:", {k: round(v, 3) for k, v in thresholds.items()})

# 1) Fold models + replay traffic -------------------------------------------------
rng = np.random.default_rng(SEED)
replay_parts = []
for k, (tr, te) in enumerate(folds):
    train, test = df.iloc[tr], df.iloc[te]
    model = random_forest(COLS, train, n_estimators=100).fit(train[COLS], train["type"])
    joblib.dump({"model": model, "features": COLS, "threshold": thresholds[k]}, ASSETS / f"fold_model_{k}.joblib", compress=3)

    picks = []
    for cls, group in test.groupby("type"):
        n = 450 if cls == "normal" else (len(group) if cls == "mitm" else 160)
        picks.append(group.sample(min(n, len(group)), random_state=SEED))
    sample = pd.concat(picks)
    proba = model.predict_proba(sample[COLS])
    classes = np.array(model.classes_)
    attack_idx = [i for i, c in enumerate(classes) if c != "normal"]
    score = 1 - proba[:, list(classes).index("normal")]
    family = classes[attack_idx][proba[:, attack_idx].argmax(axis=1)]
    high_rule = sample[["rule_icmp_redirect", "rule_icmp_ttl_exceeded"]].sum(axis=1).values > 0
    family = np.where(high_rule, "mitm", family)
    out = sample[list(dict.fromkeys(SHOW_COLUMNS + COLS + ["type"]))].copy()
    out["fold"] = k
    out["attack_score"] = score
    out["family_guess"] = family
    out["high_rule"] = high_rule
    out["predicted"] = np.where((score >= thresholds[k]) | high_rule, family, "normal")
    out["predicted_default"] = np.where((score >= 0.5) | high_rule, family, "normal")
    replay_parts.append(out)
    print(f"  fold {k}: model saved, {len(out):,} replay flows")

replay = pd.concat(replay_parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
replay.insert(0, "flow_id", np.arange(1, len(replay) + 1))
replay.to_pickle(ASSETS / "replay_flows.pkl")

# 2) Attack simulator: standard vs adversarially trained model ----------------------
train, test = train_test_split(df, test_size=0.2, stratify=df["type"], random_state=SEED)


def hgb_pipeline(frame):
    hgb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=15, l2_regularization=1.0,
                                         early_stopping=False, class_weight="balanced", random_state=SEED)
    return Pipeline([("pre", tree_preprocessor(COLS, frame)), ("model", hgb)])


attacks = train[train["type"] != "normal"].sample(frac=0.3, random_state=SEED)
augmented = perturb(attacks, rng.integers(64, 1025, len(attacks)), rng.uniform(2, 10, len(attacks)))
train_aug = pd.concat([train, augmented], ignore_index=True)
sim_std = hgb_pipeline(train).fit(train[COLS], train["type"])
sim_adv = hgb_pipeline(train_aug).fit(train_aug[COLS], train_aug["type"])
joblib.dump(sim_std, ASSETS / "sim_standard.joblib", compress=3)
joblib.dump(sim_adv, ASSETS / "sim_adversarial.joblib", compress=3)

candidates = test[(test["type"] != "normal")]
candidates = candidates[sim_std.predict(candidates[COLS]) != "normal"]
pool = pd.concat([g.sample(min(60, len(g)), random_state=SEED) for _, g in candidates.groupby("type")])
worst = perturb(pool, 1024, 10)
pool = pool.assign(evades_standard=sim_std.predict(worst[COLS]) == "normal", evades_adversarial=sim_adv.predict(worst[COLS]) == "normal")
pool.to_pickle(ASSETS / "sim_pool.pkl")
print(f"Simulator: {len(pool)} example attacks, {int(pool['evades_standard'].sum())} can evade the standard model at max padding")

# 3) Drift reference ------------------------------------------------------------------
reference = df.iloc[folds[0][0]]
drift = {"features": {}, "unsw_psi": None}
for col in DRIFT_FEATURES:
    values = np.log1p(reference[col].astype(float).values)
    edges = np.unique(np.quantile(values, np.linspace(0, 1, 11)))
    edges[0], edges[-1] = -1e18, 1e18
    props = np.histogram(values, edges)[0] / len(values)
    drift["features"][col] = {"edges": edges.tolist(), "props": props.tolist()}
unsw_dir = PROJECT_ROOT / "data" / "external" / "unsw_nb15"
if (unsw_dir / "UNSW_NB15_testing-set.csv").exists():
    u = pd.read_csv(unsw_dir / "UNSW_NB15_testing-set.csv")
    mapped = {"duration": u["dur"], "src_pkts": u["spkts"], "dst_pkts": u["dpkts"], "src_ip_bytes": u["sbytes"], "dst_ip_bytes": u["dbytes"]}
    drift["unsw_psi"] = {col: psi(reference[col], vals) for col, vals in mapped.items()}
(ASSETS / "drift_reference.json").write_text(json.dumps(drift))

# 4) Manifest -------------------------------------------------------------------------
manifest = {
    "flows_total": int(len(df)),
    "replay_flows": int(len(replay)),
    "thresholds": thresholds,
    "features": COLS,
    "rule_columns": RULE_COLUMNS,
    "built_seconds": round(time.perf_counter() - start),
}
(ASSETS / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(f"Done in {manifest['built_seconds']}s -> {ASSETS.relative_to(PROJECT_ROOT)}")
