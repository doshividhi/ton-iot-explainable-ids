"""Shared building blocks for the TON_IoT explainable IDS.

Used by notebook 06 (src/spoofing_and_calibration.py) and by the Streamlit demo app (app/).
Everything here is plain pandas / scikit-learn so it runs the same in VS Code, Jupyter and Streamlit.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

SEED = 42
TARGETS = ("label", "type")
IDENTIFIERS = ("src_ip", "dst_ip")

# ---------------------------------------------------------------------------
# Spoofing detectors (MITM "kill chain" footprints)
# ---------------------------------------------------------------------------
# A typical man-in-the-middle attack chain (e.g. Ettercap / bettercap) works in layers:
#   1. Re-route victims through the attacker      -> ICMP Redirect spoofing, TTL anomalies
#   2. Answer their name lookups with fake replies -> DNS spoofing
#   3. Strip or intercept encryption               -> TLS stripping / interception
# Each rule below looks for the footprint of one layer using only flow behaviour,
# never IP addresses (IPs were shown to leak host identity in Report 3).
# For ICMP, Zeek stores the ICMP type in src_port and the code in dst_port.

SPOOF_RULES = {
    "icmp_redirect": {
        "title": "ICMP Redirect spoofing",
        "layer": "Routing (L3)",
        "why": "An ICMP type-5 Redirect tells a device to send its traffic through a different gateway. "
        "Attackers forge these to place themselves in the path. Normal IoT traffic never needs it.",
        "severity": "high",
    },
    "icmp_ttl_exceeded": {
        "title": "TTL-exceeded / forwarding anomaly",
        "layer": "Routing (L3)",
        "why": "ICMP type-11 'time exceeded' bursts appear when an extra hop (the attacker) forwards and "
        "re-writes packets. Almost absent from normal traffic.",
        "severity": "medium",
    },
    "dns_spoof": {
        "title": "Rogue DNS responder",
        "layer": "Name resolution (L7)",
        "why": "A successful DNS answer that advertises recursion (RA flag). This network's normal resolver "
        "does not set RA, so an answer that does was produced by a different, unexpected responder.",
        "severity": "medium",
    },
    "tls_strip": {
        "title": "TLS stripping / interception",
        "layer": "Encryption (L6)",
        "why": "A completed session on the TLS port where Zeek saw no TLS version and no certificate: "
        "the encrypted handshake never happened, which is what SSL stripping produces.",
        "severity": "medium",
    },
}
RULE_COLUMNS = [f"rule_{name}" for name in SPOOF_RULES]
HIGH_SPECIFICITY_RULES = ["rule_icmp_redirect", "rule_icmp_ttl_exceeded"]


def spoof_rule_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one 0/1 column per spoofing rule for every flow."""
    icmp = frame["proto"].astype(str) == "icmp"
    service = frame["service"].astype(str)
    flags = pd.DataFrame(index=frame.index)
    flags["rule_icmp_redirect"] = icmp & (frame["src_port"] == 5)
    flags["rule_icmp_ttl_exceeded"] = icmp & (frame["src_port"] == 11)
    flags["rule_dns_spoof"] = (service == "dns") & (frame["dns_RA"].astype(str) == "T") & (frame["dns_rcode"] == 0)
    flags["rule_tls_strip"] = (
        service.str.contains("ssl")
        & (frame["ssl_version"].astype(str) == "-")
        & (frame["ssl_subject"].astype(str) == "-")
        & frame["conn_state"].astype(str).isin(["SF", "RSTR", "RSTO", "S2", "S3"])
        & (frame["dst_bytes"] > 0)
    )
    return flags.astype(int)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_dedup(data_path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load the TON_IoT train/test network file, drop exact feature duplicates and
    conflicting-label rows (same rule as notebook 05), and add rule flags."""
    raw = pd.read_csv(data_path)
    features = [c for c in raw.columns if c not in TARGETS + IDENTIFIERS]
    row_hash = pd.util.hash_pandas_object(raw[features], index=False)
    types_per_hash = raw.assign(h=row_hash).groupby("h")["type"].nunique()
    conflict = row_hash.isin(types_per_hash[types_per_hash > 1].index)
    keep = (~row_hash.duplicated()) & (~conflict)
    df = raw[keep.values].reset_index(drop=True)
    df = pd.concat([df, spoof_rule_flags(df)], axis=1)
    return df, features


def feature_sets(features: list[str]) -> dict[str, list[str]]:
    no_ports = [c for c in features if c not in ("src_port", "dst_port")]
    return {"no_ports": no_ports, "hybrid": no_ports + RULE_COLUMNS}


def tree_preprocessor(cols: list[str], frame: pd.DataFrame) -> ColumnTransformer:
    cat = [c for c in cols if not pd.api.types.is_numeric_dtype(frame[c])]
    num = [c for c in cols if c not in cat]
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)
    return ColumnTransformer([("cat", encoder, cat), ("num", "passthrough", num)])


def random_forest(cols: list[str], frame: pd.DataFrame, n_estimators: int = 200) -> Pipeline:
    """Tuned settings from notebook 05 (max_depth 20, min_samples_leaf 2, sqrt features)."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=20,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=SEED,
    )
    return Pipeline([("pre", tree_preprocessor(cols, frame)), ("model", model)])


def attack_score(model: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (attack score = 1 - P(normal), class probabilities)."""
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    return 1 - proba[:, classes.index("normal")], proba


def decide(model: Pipeline, X: pd.DataFrame, rules: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Final system decision for each flow.

    * attack if the calibrated attack score >= threshold, OR a high-specificity spoofing rule fires;
    * family = most likely attack class; forced to 'mitm' when a high-specificity spoofing rule fires.
    """
    score, proba = attack_score(model, X)
    classes = np.array(model.classes_)
    attack_classes = [i for i, c in enumerate(classes) if c != "normal"]
    family = classes[attack_classes][proba[:, attack_classes].argmax(axis=1)]
    high_rule = rules[HIGH_SPECIFICITY_RULES].sum(axis=1).values > 0
    is_attack = (score >= threshold) | high_rule
    family = np.where(high_rule, "mitm", family)
    return pd.DataFrame(
        {
            "attack_score": score,
            "is_attack": is_attack,
            "predicted": np.where(is_attack, family, "normal"),
            "any_spoof_rule": rules[RULE_COLUMNS].sum(axis=1).values > 0,
        },
        index=X.index,
    )


def psi(reference, current, bins: int = 10) -> float:
    """Population Stability Index on log1p-scaled values (<0.1 stable, 0.1-0.25 moderate, >0.25 major)."""
    reference = np.log1p(np.asarray(reference, float))
    current = np.log1p(np.asarray(current, float))
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.clip(np.histogram(reference, edges)[0] / len(reference), 1e-6, None)
    cur = np.clip(np.histogram(current, edges)[0] / len(current), 1e-6, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


# ---------------------------------------------------------------------------
# Evasion: attacker-controlled changes to their own traffic (used in notebook 05 and the simulator)
# ---------------------------------------------------------------------------
def perturb(frame: pd.DataFrame, pad_bytes, slowdown) -> pd.DataFrame:
    """Pad each flow's payload by `pad_bytes` (spread over extra packets) and stretch its duration by `slowdown`.
    `pad_bytes` and `slowdown` can be scalars or arrays (one value per row)."""
    out = frame.copy()
    pad = np.broadcast_to(np.asarray(pad_bytes, dtype=float), (len(out),))
    extra_pkts = np.ceil(pad / 1460).astype(int)
    out["src_bytes"] = out["src_bytes"] + pad
    out["src_pkts"] = out["src_pkts"] + extra_pkts
    out["src_ip_bytes"] = out["src_ip_bytes"] + pad + 40 * extra_pkts
    out["duration"] = out["duration"] * np.broadcast_to(np.asarray(slowdown, dtype=float), (len(out),))
    return out
