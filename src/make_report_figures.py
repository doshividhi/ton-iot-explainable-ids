"""Redraw Figures 1-3 used in the Interim Progress Report.

Usage (from the repo root):
    python src/make_report_figures.py data/train_test_network.csv results/report_figures
"""
import sys, pandas as pd, numpy as np, matplotlib, matplotlib.ticker
matplotlib.use("Agg"); import matplotlib.pyplot as plt
df = pd.read_csv(sys.argv[1]); out=sys.argv[2]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.edgecolor":"#999","axes.titlesize":10.5,"axes.titleweight":"bold","axes.spines.top":False,"axes.spines.right":False})
BLUE="#2a78d6"; ORANGE="#eb6834"; INK="#333"
# Fig1
c = df.type.value_counts().sort_values()
fig,ax=plt.subplots(figsize=(6.5,2.6))
cols=[ORANGE if k=="mitm" else BLUE for k in c.index]
ax.barh(c.index,c.values,color=cols,height=0.62)
for i,(k,v) in enumerate(c.items()):
    ax.text(v+600,i,f"{v:,} ({v/len(df):.1%})",va="center",fontsize=8,color=INK)
ax.set_xlim(0,62000); ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}")); ax.set_xlabel("Number of flows"); ax.grid(axis="x",color="#e6e6e6"); ax.set_axisbelow(True)
ax.set_title("Records per traffic category (n = 211,043)",loc="left")
fig.tight_layout(); fig.savefig(f"{out}/fig1_classes.png",dpi=220); plt.close()
# Fig2 heatmap conn_state
order=["normal","backdoor","ddos","dos","injection","mitm","password","ransomware","scanning","xss"]
states=["SF","S0","S1","S3","SH","REJ","OTH","RSTR","RSTRH","SHR","RSTO"]
ct=pd.crosstab(df.type,df.conn_state,normalize="index").reindex(order)
other=1-ct[states].sum(axis=1); ct=ct[states].assign(other=other)
fig,ax=plt.subplots(figsize=(6.5,3.0))
im=ax.imshow(ct.values,cmap="Blues",vmin=0,vmax=1,aspect="auto")
ax.set_xticks(range(ct.shape[1]),ct.columns,fontsize=8); ax.set_yticks(range(len(order)),order,fontsize=8)
for i in range(ct.shape[0]):
    for j in range(ct.shape[1]):
        v=ct.values[i,j]
        if v>=0.05: ax.text(j,i,f"{v:.0%}",ha="center",va="center",fontsize=7,color="white" if v>0.55 else INK)
ax.set_xlabel("Zeek connection state (conn_state)"); ax.spines[:].set_visible(False)
cb=fig.colorbar(im,ax=ax,fraction=0.03,pad=0.02); cb.ax.tick_params(labelsize=7); cb.set_label("Share of category's flows",fontsize=7.5)
ax.set_title("Connection-state mix within each category (row %)",loc="left")
fig.tight_layout(); fig.savefig(f"{out}/fig2_connstate.png",dpi=220); plt.close()
# Fig3 src_bytes boxplot log
fig,ax=plt.subplots(figsize=(6.5,2.7))
data=[np.log10(df.loc[df.type==k,"src_bytes"]+1) for k in order]
bp=ax.boxplot(data,vert=False,tick_labels=order,widths=0.55,patch_artist=True,showfliers=True,
  flierprops=dict(marker="o",markersize=1.5,markerfacecolor="#9bbbe6",markeredgecolor="none",alpha=.4),
  medianprops=dict(color=INK,linewidth=1.3),boxprops=dict(facecolor="#cfe0f6",edgecolor=BLUE),whiskerprops=dict(color=BLUE),capprops=dict(color=BLUE))
ax.invert_yaxis(); ax.set_xticks(range(0,11),["0","10","100","1K","10K","100K","1M","10M","100M","1B","10B"])
ax.set_xlabel("Source payload bytes per flow (log scale)"); ax.grid(axis="x",color="#e6e6e6"); ax.set_axisbelow(True)
ax.tick_params(axis="y",labelsize=8)
ax.set_title("Source payload bytes per flow, by category",loc="left")
fig.tight_layout(); fig.savefig(f"{out}/fig3_srcbytes.png",dpi=220); plt.close()
