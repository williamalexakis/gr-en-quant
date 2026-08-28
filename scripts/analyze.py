import re, glob, os
import pandas as pd
import matplotlib.pyplot as plt

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)
ORDER = ["Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
P = {"ppl_q" : r"Mean PPL\(Q\)\s*:\s*([\d.]+)",
     "ppl_base" : r"Mean PPL\(base\)\s*:\s*([\d.]+)",
     "ppl_ratio" : r"Mean PPL\(Q\)/PPL\(base\)\s*:\s*([\d.]+)",
     "kld_mean" : r"Mean\s+KLD:\s*([\-\d.]+)",
     "kld_median" : r"Median\s+KLD:\s*([\-\d.]+)",
     "kld_999" : r"99\.9%\s+KLD:\s*([\-\d.]+)",
     "same_top" : r"Same top p:\s*([\d.]+)"}

LABEL = {"el": "Greek (el)", "en": "English (en)"}
COLOR = {"el": "#1f77b4", "en": "#d1495b"}

# We parse a results dir and make
# a dataframe and CSV out of it
def load(res_dir, out_csv):
    if not os.path.isdir(res_dir):
        return None
    rows = []
    for path in glob.glob(f"{res_dir}/*_*.txt"):
        lang, quant = os.path.basename(path)[:-4].split("_", 1)
        if quant == "Q8_0":
            continue
        txt = open(path).read()
        rec = {"lang": lang, "quant": quant}
        for k, pat in P.items():
            m = re.search(pat, txt)
            rec[k] = float(m.group(1)) if m else None
        rows.append(rec)
    if not rows:
        return None
    df = pd.DataFrame(rows)

    # We should fail loudly if we can't
    # access the quants
    unknown = sorted(set(df["quant"]) - set(ORDER))
    if unknown:
        raise SystemExit(f"{res_dir}: Unrecognized quant name(s) '{unknown}'.")
    df["quant"] = pd.Categorical(df["quant"], categories=ORDER, ordered=True)
    
    df = df.sort_values(["lang", "quant"])
    df.to_csv(out_csv, index=False)
    print(f"\n[{res_dir}]")
    print(df.to_string(index=False))
    return df

def style(ax):
    ax.grid(True, which="both", ls=":", lw=0.6, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Quantization (ascending)")

# Plot a single model
def plot(df, col, ylab, title, fname, logy=False, hline=None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for lang in ["en", "el"]:
        g = df[df["lang"] == lang]
        ax.plot(g["quant"].astype(str), g[col], marker="o", ms=7, lw=2,
                color=COLOR[lang], label=LABEL[lang])

    if logy: ax.set_yscale("log")
    if hline is not None: ax.axhline(hline, ls="--", lw=0.9, color="grey")

    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=13, fontweight="bold")
    style(ax); ax.legend(frameon=False, fontsize=11)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/{fname}", dpi=150); plt.close(fig)

# Plot PPL ratio, mean KLD, and top-1 token
# agreement individual metrics
def single_model_figs(df, tag, suffix):
    plot(df, "ppl_ratio", "PPL(Q) / PPL(Q8\u2080)",
         f"Relative perplexity degradation{suffix}", f"ppl_degradation{tag}.png", hline=1.0)
    plot(df, "kld_mean", "Mean KL divergence from Q8\u2080  (log scale)",
         f"Distributional distortion{suffix}", f"kld{tag}.png", logy=True)
    plot(df, "same_top", "Top-1 agreement with Q8\u2080  (%)",
         f"Top-token survival{suffix}", f"same_top{tag}.png")

# Plot the 1B versus 3B comparison
def plot_compare(d3, d1):
    def series(d, lang):
        g = d[d["lang"] == lang].sort_values("quant")
        return g["quant"].astype(str), g
    pairs = [(d1, "1B", "-", "o"), (d3, "3B", "--", "s")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for d, size, ls, mk in pairs:
        for lang in ["en", "el"]:
            x, g = series(d, lang)
            name = f"{size} {'Greek' if lang=='el' else 'English'}"
            ax1.plot(x, g["kld_mean"], ls=ls, marker=mk, ms=6, color=COLOR[lang], label=name)
            ax2.plot(x, g["same_top"], ls=ls, marker=mk, ms=6, color=COLOR[lang], label=name)
    ax1.set_yscale("log")
    ax1.set_ylabel("Mean KL Divergence From Q8\u2080 (log)")
    ax1.set_title("Distributional Distortion (1B vs 3B)", fontweight="bold")
    ax2.set_ylabel("Top-1 Token Agreement With Q8\u2080 (%)")
    ax2.set_title("Top-Token Survival (1B vs 3B)", fontweight="bold")
    for ax in (ax1, ax2):
        style(ax); ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/size_comparison.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for d, size, ls, mk in pairs:
        for lang in ["en", "el"]:
            x, g = series(d, lang)
            ax.plot(x, g["ppl_ratio"], ls=ls, marker=mk, ms=6, color=COLOR[lang],
                    label=f"{size} {'Greek' if lang=='el' else 'English'}")
    ax.axhline(1.0, ls="--", lw=0.9, color="grey")
    ax.set_ylabel("PPL(Q) / PPL(Q8\u2080)")
    ax.set_title("Relative Perplexity Degradation (1B vs 3B)", fontweight="bold")
    style(ax); ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/ppl_comparison.png", dpi=150); plt.close(fig)

df3 = load("results_3b", "results_3b/summary_3b.csv")
df1 = load("results_1b", "results_1b/summary_1b.csv")

if df3 is not None:
    single_model_figs(df3, "", "")
if df1 is not None:
    single_model_figs(df1, "_1b", " (1B)")
if df3 is not None and df1 is not None:
    plot_compare(df3, df1)

print("\nSuccessfully created summaries and graphs.")
