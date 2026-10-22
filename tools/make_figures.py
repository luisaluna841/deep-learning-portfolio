"""Gera as figuras do RELATORIO.md a partir dos CSVs em results/.

Uso (na raiz do repositório): python tools/make_figures.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RES = Path(__file__).resolve().parent.parent / "results"
OUT = RES / "figuras"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("figura:", OUT / name)


# Fig. 1: curvas de treino (fase 1 = cabeça, fase 2 = fine-tuning)
hist = json.loads((RES / "train_history.json").read_text(encoding="utf-8"))
ep = range(1, len(hist) + 1)
n_head = sum(h["phase"] != "fine-tuning" for h in hist)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
for key, ax, title in [("loss", axes[0], "Loss"), ("top1", axes[1], "Acurácia top-1")]:
    ax.plot(ep, [h[f"train_{key}"] for h in hist], marker="o", ms=3, label="treino")
    ax.plot(ep, [h[f"val_{key}"] for h in hist], marker="o", ms=3, label="validação")
    ax.axvline(n_head + 0.5, color="gray", ls="--", lw=1)
    ax.set(title=title, xlabel="época")
    ax.legend()
axes[1].text(n_head + 0.7, 0.82, "fine-tuning →", color="gray")
save(fig, "fig1_treino.png")

# Fig. 2: sensibilidade por bloco à quantização INT8
sens = pd.read_csv(RES / "sensibilidade_blocos.csv")
fp32 = 0.900
drop = (fp32 - sens["top1"]) * 100
fig, ax = plt.subplots(figsize=(9, 3))
ax.bar(sens["bloco"].str.replace("features.", "f"), drop, color=["tab:red" if d > 5 else "tab:gray" for d in drop])
ax.set(ylabel="queda de top-1 (p.p.)", title="Quantizando um bloco por vez (INT8, MinMax)")
ax.tick_params(axis="x", rotation=60)
save(fig, "fig2_sensibilidade.png")

# Fig. 3: pruning
pr = pd.read_csv(RES / "pruning.csv")
fig, ax1 = plt.subplots(figsize=(5.5, 3.2))
ax1.plot(pr["pruning"] * 100, pr["top1_subset"] * 100, marker="o", color="tab:blue", label="top-1")
ax1.set(xlabel="pesos zerados (%)", ylabel="top-1 (%)", title="Pruning L1 global sem fine-tuning")
ax2 = ax1.twinx()
ax2.plot(pr["pruning"] * 100, pr["latencia_ms_p50"], marker="s", color="tab:orange", label="latência")
ax2.set(ylabel="latência p50 (ms)", ylim=(0, pr["latencia_ms_p50"].max() * 1.5))
ax2.grid(False)
fig.legend(loc="lower left", bbox_to_anchor=(0.15, 0.2))
save(fig, "fig3_pruning.png")

# Fig. 4: MACs x latência
ar = pd.read_csv(RES / "arquiteturas.csv")
fig, ax = plt.subplots(figsize=(5.5, 3.6))
ax.scatter(ar["GMACs"], ar["latencia_ms_p50"], s=ar["params_M"] * 10 + 20, alpha=0.8)
for _, r in ar.iterrows():
    ax.annotate(r["arquitetura"], (r["GMACs"], r["latencia_ms_p50"]), xytext=(5, 3), textcoords="offset points", fontsize=7)
ax.set(xscale="log", yscale="log", xlabel="GMACs por imagem", ylabel="latência p50 (ms, 1 thread)",
       title="Custo teórico × latência medida")
save(fig, "fig4_macs_latencia.png")
