import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from demo_lib import FAMILY_LABELS, results_csv, style_page

style_page()
st.title("Model results")
st.caption("Why we test on unseen devices, and what the final system achieves.")

tab1, tab2, tab3, tab4 = st.tabs(["Random split vs. unseen devices", "Fingerprint ablation", "Final system confusion matrix", "External dataset"])

with tab1:
    folds = results_csv("results/status_report/metrics/model_comparison_folds.csv")
    if folds is None:
        st.info("Run notebook 05 to create results/status_report/metrics/.")
    else:
        names = {"majority": "Majority", "logreg": "Logistic Regression", "random_forest": "Random Forest", "hist_gb": "Gradient Boosting"}
        f = folds[folds["model"].isin(names)].assign(model=lambda d: d["model"].map(names), split=lambda d: d["split"].map({"random": "Random 5-fold", "host_grouped": "Unseen devices (host-grouped)"}))
        agg = f.groupby(["model", "split"])["macro_f1"].agg(["mean", "std"]).reset_index()
        fig = px.bar(agg, x="model", y="mean", color="split", barmode="group", error_y="std", text=agg["mean"].round(2),
                     color_discrete_map={"Random 5-fold": "#9aa4b1", "Unseen devices (host-grouped)": "#2a78d6"},
                     category_orders={"model": list(names.values())})
        fig.update_layout(height=400, yaxis_title="10-class macro-F1", xaxis_title=None, legend_title=None, yaxis=dict(range=[0, 1.1]))
        st.plotly_chart(fig, key="split_compare")
        st.markdown("Every model loses **0.26–0.31 macro-F1** when tested on devices it never saw. A random split lets the model "
                    "recognise machines it already knows, which is not what an IDS faces in a real network.")

with tab2:
    ab = results_csv("results/status_report/metrics/fingerprint_ablation.csv", header=[0, 1], index_col=[0, 1])
    if ab is not None:
        m = ab[("macro_f1", "mean")].rename("macro_f1").reset_index()
        m.columns = ["features", "split", "macro_f1"]
        m["features"] = m["features"].map({"with_ips": "+ IP addresses", "all_40": "Raw ports", "no_ports": "No ports", "engineered": "Bucketed ports + ratios"})
        m["split"] = m["split"].map({"random": "Random 5-fold", "host_grouped": "Unseen devices"})
        fig = px.bar(m, x="macro_f1", y="features", color="split", barmode="group", orientation="h", text=m["macro_f1"].round(2),
                     color_discrete_map={"Random 5-fold": "#9aa4b1", "Unseen devices": "#2a78d6"})
        fig.update_layout(height=380, xaxis=dict(range=[0, 1.1], title="10-class macro-F1 (gradient boosting)"), yaxis_title=None, legend_title=None)
        st.plotly_chart(fig, key="ablation")
        st.markdown("Adding IP addresses looks perfect on a random split (0.99) and collapses on new devices (0.17): the model was "
                    "memorising hosts. Dropping ports costs almost nothing, so the final model uses **no IPs and no raw ports**.")

with tab3:
    cm = results_csv("results/spoofing/metrics/final_system_confusion.csv", index_col=0)
    if cm is not None:
        norm = cm.div(cm.sum(axis=1), axis=0)
        labels = [FAMILY_LABELS.get(c, c) for c in cm.columns]
        fig = go.Figure(go.Heatmap(z=norm.values, x=labels, y=labels, colorscale="Blues", zmin=0, zmax=1,
                                   text=[[f"{v:.0%}" if v >= 0.02 else "" for v in r] for r in norm.values], texttemplate="%{text}",
                                   hovertemplate="True %{y}<br>Predicted %{x}<br>%{z:.1%}<extra></extra>"))
        fig.update_layout(height=520, xaxis_title="Predicted", yaxis_title="True", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, key="cm")
        st.markdown("Final system (hybrid Random Forest + spoofing detectors, calibrated threshold), every flow predicted by the "
                    "model that never saw its device. Families learned from only one attacker machine (backdoor, DoS) are still "
                    "often given the wrong attack label, which is the main item for the final phase.")

with tab4:
    ext = results_csv("results/status_report/metrics/generalization_external.csv")
    if ext is not None:
        show = ext[["setting", "precision", "recall", "fpr", "roc_auc"]].rename(columns={"setting": "Setting", "precision": "Precision", "recall": "Recall", "fpr": "False-alarm rate", "roc_auc": "ROC-AUC"})
        st.dataframe(show.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "False-alarm rate": "{:.1%}", "ROC-AUC": "{:.3f}"}), hide_index=True)
        st.markdown("Trained on TON_IoT and tested on **UNSW-NB15** (a different lab, tool and attack mix), the model does worse "
                    "than chance (ROC-AUC 0.47). Adding just **5% labelled UNSW data** brings it to 0.97. Lesson: a new network needs a "
                    "small labelled sample before deployment, which is why the drift monitor exists.")
