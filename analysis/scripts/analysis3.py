"""
query_log3.csv — Comprehensive Analysis & Visualization (v3)
5 figures: figs 1, 2, 3, 5, 6
- Fig 4 removed
- Fig 7 removed
- Fig 2: removed hallucination count bar chart (pie only)
- Fig 5: removed response time by satisfaction (overall distribution only)
- Fig 3/5 remain as box-and-whisker
- satisfied NULL/NaN treated as yes
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "analysis" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from collections import Counter
import re
import warnings
warnings.filterwarnings("ignore")

# ── Color palette ──────────────────────────────────────────────────────────────
C_GREEN  = "#2ecc71"
C_RED    = "#e74c3c"
C_BLUE   = "#3498db"
C_ORANGE = "#e67e22"
C_PURPLE = "#9b59b6"
C_TEAL   = "#1abc9c"
C_GRAY   = "#95a5a6"
C_DARK   = "#2c3e50"
C_YELLOW = "#f1c40f"
C_PINK   = "#e91e8c"

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlepad":     12,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "figure.facecolor":  "white",
    "axes.facecolor":    "#f9f9f9",
})

# ── Load & clean ───────────────────────────────────────────────────────────────
df = pd.read_csv(PROJECT_ROOT / "query_log3.csv")

# satisfied: NULL/NaN/empty → yes
def parse_satisfied(val):
    if pd.isna(val):
        return True
    s = str(val).strip().lower()
    if s in ("", "null", "nan"):
        return True
    return s == "yes"

df["satisfied_bool"]     = df["satisfied"].apply(parse_satisfied)
df["hallucinated_clean"] = df["hallucinated"].fillna("no").str.strip().str.lower()
df["hallucinated_bool"]  = df["hallucinated_clean"] == "yes"
df["hallucination_type"] = df["hallucination_type"].fillna("none").str.strip()
df["total_tokens"]       = df["input_tokens"] + df["output_tokens"]
df["query_num"]          = range(1, len(df) + 1)

n = len(df)
print(f"Loaded {n} rows.\n")
print("=== SUMMARY STATS ===")
print(f"Satisfied:    {df['satisfied_bool'].sum()} / {n}  ({df['satisfied_bool'].mean()*100:.1f}%)")
print(f"Hallucinated: {df['hallucinated_bool'].sum()} / {n}  ({df['hallucinated_bool'].mean()*100:.1f}%)")
print(f"Input tokens  — mean: {df['input_tokens'].mean():.0f}, std: {df['input_tokens'].std():.0f}")
print(f"Output tokens — mean: {df['output_tokens'].mean():.0f}, std: {df['output_tokens'].std():.0f}")
print(f"Total tokens  — mean: {df['total_tokens'].mean():.0f}, std: {df['total_tokens'].std():.0f}")
print(f"Response time — mean: {df['response_time_sec'].mean():.2f}s, std: {df['response_time_sec'].std():.2f}s\n")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Satisfaction & Hallucination Pie Charts
# ══════════════════════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(1, 2, figsize=(11, 5))
fig1.suptitle(f"Response Quality Overview — Multi-Agent (n = {n})", fontsize=15, fontweight="bold", y=1.01)

def make_pie(ax, vals, labels, colors, title, n_total):
    wedges, texts, autotexts = ax.pie(
        vals, colors=colors,
        autopct=lambda p: f"{p:.1f}%\n({int(round(p * n_total / 100))})",
        startangle=90, pctdistance=0.65,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for at in autotexts:
        at.set_fontsize(10); at.set_fontweight("bold")
    ax.set_title(title, fontweight="bold")
    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
    ax.legend(handles=patches, fontsize=10, loc="lower center",
              bbox_to_anchor=(0.5, -0.12), framealpha=0.9, edgecolor="#ccc")

sat_counts = df["satisfied_bool"].value_counts()
make_pie(axes[0],
         [sat_counts.get(True, 0), sat_counts.get(False, 0)],
         ["Satisfied", "Unsatisfied"], [C_GREEN, C_RED],
         "Satisfied vs. Unsatisfied Responses", n)

hall_counts = df["hallucinated_bool"].value_counts()
make_pie(axes[1],
         [hall_counts.get(False, 0), hall_counts.get(True, 0)],
         ["Not Hallucinated", "Hallucinated"], [C_TEAL, C_ORANGE],
         "Hallucinated vs. Non-Hallucinated Responses", n)

plt.tight_layout()
fig1.savefig(FIGURES_DIR / "fig1_satisfaction_hallucination_pies.png", dpi=150, bbox_inches="tight")
print("Saved fig1_satisfaction_hallucination_pies.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Hallucination Type Breakdown (donut pie only)
# ══════════════════════════════════════════════════════════════════════════════
hall_df     = df[df["hallucinated_bool"]].copy()
type_counts = hall_df["hallucination_type"].value_counts()

type_label_map  = {"prereq_wrong": "Wrong / Missing Prereq",
                   "course_invented": "Course Invented",
                   "other": "Other", "none": "Unspecified"}
type_colors_map = {"prereq_wrong": C_RED, "course_invented": C_ORANGE,
                   "other": C_PURPLE, "none": C_GRAY}

labels_ordered = [k for k in ["prereq_wrong", "course_invented", "other", "none"]
                  if k in type_counts.index]
vals_ordered   = [type_counts[k] for k in labels_ordered]
colors_ordered = [type_colors_map[k] for k in labels_ordered]
nice_labels    = [type_label_map[k] for k in labels_ordered]

fig2, ax2 = plt.subplots(figsize=(7, 6))
fig2.suptitle(f"Hallucination Type Analysis — Multi-Agent ({len(hall_df)} hallucinated responses)",
              fontsize=15, fontweight="bold")

wedges3, _, autotexts3 = ax2.pie(
    vals_ordered, colors=colors_ordered,
    autopct=lambda p: f"{p:.0f}%\n(n={int(round(p*len(hall_df)/100))})",
    startangle=90, pctdistance=0.72,
    wedgeprops={"edgecolor": "white", "linewidth": 2, "width": 0.55},
)
for at in autotexts3:
    at.set_fontsize(9); at.set_fontweight("bold")
ax2.set_title("")
ax2.add_patch(plt.Circle((0, 0), 0.45, fc="white"))
ax2.text(0, 0, f"{len(hall_df)}\ntotal", ha="center", va="center",
         fontsize=12, fontweight="bold", color=C_DARK)

patches2 = [mpatches.Patch(color=c, label=l) for c, l in zip(colors_ordered, nice_labels)]
ax2.legend(handles=patches2, fontsize=10, loc="lower center",
           bbox_to_anchor=(0.5, -0.08), framealpha=0.9, edgecolor="#ccc")

plt.tight_layout()
fig2.savefig(FIGURES_DIR / "fig2_hallucination_types.png", dpi=150, bbox_inches="tight")
print("Saved fig2_hallucination_types.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Token Usage Bar Chart with Error Bars (styled after query_analysis_0518)
# ══════════════════════════════════════════════════════════════════════════════
token_labels  = ["Input Tokens", "Output Tokens", "Total Tokens"]
token_series  = [df["input_tokens"], df["output_tokens"], df["total_tokens"]]
token_colors3 = ["#5dade2", "#58d68d", "#f0b27a"]

means3 = [s.mean() for s in token_series]
stds3  = [s.std()  for s in token_series]
vars3  = [s.var()  for s in token_series]

fig3, ax3 = plt.subplots(figsize=(10, 6))
fig3.suptitle(f"Token Usage Distribution — Multi-Agent (n = {n})", fontsize=15, fontweight="bold")

x3   = np.arange(len(token_labels))
bars3 = ax3.bar(x3, means3, width=0.5, color=token_colors3, edgecolor="white",
                linewidth=1.2, alpha=0.88, zorder=3)

# Error bars ±1 std
ax3.errorbar(x3, means3, yerr=stds3, fmt="none", color="#333",
             capsize=7, linewidth=1.8, zorder=4)

# Annotate each bar above the error cap
for i, (bar, mean, std, var) in enumerate(zip(bars3, means3, stds3, vars3)):
    ax3.text(
        bar.get_x() + bar.get_width() / 2,
        mean + std + max(means3) * 0.12,
        f"μ = {mean:,.0f}\nσ = {std:,.0f}\nσ² = {var:,.0f}",
        ha="center", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#ccc", alpha=0.9)
    )

ax3.set_xticks(x3)
ax3.set_xticklabels(token_labels, fontsize=11)
ax3.set_ylabel("Token Count", fontsize=11)
ax3.set_title("")
ax3.yaxis.grid(True, linestyle="--", alpha=0.5)
ax3.set_axisbelow(True)
ax3.set_ylim(0, ax3.get_ylim()[1] * 1.45)

patches3 = [mpatches.Patch(color=c, label=l)
            for c, l in zip(token_colors3, token_labels)]
ax3.legend(handles=patches3, fontsize=9, loc="lower right")

plt.tight_layout()
fig3.savefig(FIGURES_DIR / "fig3_token_bars.png", dpi=150, bbox_inches="tight")
print("Saved fig3_token_bars.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Response Time Box-and-Whisker with jitter (styled after query_analysis_0518)
# ══════════════════════════════════════════════════════════════════════════════
rt = df["response_time_sec"]

fig5, ax5 = plt.subplots(figsize=(8, 6))
fig5.suptitle("Response Time Analysis — Multi-Agent", fontsize=15, fontweight="bold")

bp5 = ax5.boxplot(rt, patch_artist=True, widths=0.4,
                  medianprops=dict(color=C_RED, linewidth=2.5),
                  boxprops=dict(facecolor="#aed6f1", color="#2980b9", linewidth=1.5),
                  whiskerprops=dict(color="#2980b9", linewidth=1.5, linestyle="--"),
                  capprops=dict(color="#2980b9", linewidth=1.5),
                  flierprops=dict(marker="", markersize=0))

# Jittered individual points
np.random.seed(42)
jitter = np.random.uniform(-0.07, 0.07, size=len(rt))
ax5.scatter(1 + jitter, rt, alpha=0.5, color="#2471a3", s=35, zorder=3,
            label="Individual queries")

q1_rt, med_rt, q3_rt = rt.quantile([0.25, 0.5, 0.75])
ax5.text(
    1.28, rt.max() * 0.99,
    f"n = {n}\n"
    f"Mean   = {rt.mean():.1f}s\n"
    f"Median = {med_rt:.1f}s\n"
    f"Std    = {rt.std():.1f}s\n"
    f"Var    = {rt.var():.1f}\n"
    f"Max    = {rt.max():.1f}s\n"
    f"Min    = {rt.min():.1f}s\n"
    f"Outliers (>80s): {(rt > 80).sum()}",
    fontsize=8.5, va="top",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#aaa", alpha=0.9)
)

ax5.set_xticks([1])
ax5.set_xticklabels(["All Queries"], fontsize=10)
ax5.set_ylabel("Response Time (seconds)", fontsize=11)
ax5.set_title("")
ax5.yaxis.grid(True, linestyle="--", alpha=0.5)
ax5.set_axisbelow(True)

ax5.legend(handles=[
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2471a3",
               markersize=7, label="Individual queries"),
], fontsize=8, loc="upper left")

plt.tight_layout()
fig5.savefig(FIGURES_DIR / "fig5_response_time_boxplot.png", dpi=150, bbox_inches="tight")
print("Saved fig5_response_time_boxplot.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Course Recommendation Frequency (updated with cybersecurity query)
# ══════════════════════════════════════════════════════════════════════════════
course_map = {
    "01:198:210": "Data Mgmt for Data Science\n(01:198:210)",
    "01:198:213": "Software Methodology\n(01:198:213)",
    "01:198:214": "Systems Programming\n(01:198:214)",
    "01:198:314": "Principles of Prog. Languages\n(01:198:314)",
    "01:198:323": "Numerical Analysis\n(01:198:323)",
    "01:198:336": "Principles of Info & Data Mgmt\n(01:198:336)",
    "01:198:352": "Internet Technology\n(01:198:352)",
    "01:198:415": "Compilers\n(01:198:415)",
    "01:198:416": "Operating Systems Design\n(01:198:416)",
    "01:198:417": "Distributed Systems\n(01:198:417)",
    "01:198:419": "Computer Security\n(01:198:419)",
    "01:198:428": "Intro to Computer Graphics\n(01:198:428)",
    "01:198:431": "Software Engineering\n(01:198:431)",
    "01:198:440": "Intro to AI\n(01:198:440)",
    "01:198:442": "Topics in CS (442)\n(01:198:442)",
    "01:198:443": "Topics in CS (443)\n(01:198:443)",
    "01:198:452": "Formal Languages & Automata\n(01:198:452)",
}

raw_responses = {
 1: ["01:198:210"],
 2: ["01:198:210","01:198:336","01:198:442"],
 3: ["01:198:336","01:198:214","01:198:210","01:198:443"],
 4: ["01:198:336","01:198:210","01:198:442"],
 5: ["01:198:213","01:198:214","01:198:336","01:198:210","01:198:443"],
 6: ["01:198:442"],
 7: ["01:198:336","01:198:314","01:198:214","01:198:210","01:198:443"],
 8: ["01:198:213","01:198:336","01:198:214","01:198:443"],
 9: ["01:198:213","01:198:314","01:198:336","01:198:214","01:198:442"],
10: ["01:198:214","01:198:314","01:198:442"],
11: ["01:198:210","01:198:336","01:198:442"],
12: ["01:198:428","01:198:442","01:198:443"],
13: ["01:198:336","01:198:210","01:198:323","01:198:214","01:198:213"],
14: ["01:198:336","01:198:452","01:198:314","01:198:210","01:198:442"],
15: ["01:198:210","01:198:442","01:198:443"],
16: ["01:198:214","01:198:314"],
17: ["01:198:336","01:198:214","01:198:213","01:198:443"],
18: ["01:198:213","01:198:214","01:198:336"],
19: ["01:198:213","01:198:214","01:198:314","01:198:443","01:198:442"],
20: ["01:198:314","01:198:442","01:198:443"],
21: ["01:198:323","01:198:336","01:198:314","01:198:214","01:198:443"],
22: ["01:198:214","01:198:336","01:198:314","01:198:210","01:198:443"],
23: ["01:198:213","01:198:314","01:198:214","01:198:336","01:198:443"],
24: ["01:198:214","01:198:443","01:198:442","01:198:336"],
25: ["01:198:210","01:198:336","01:198:213","01:198:442"],
26: ["01:198:428","01:198:443"],
27: ["01:198:443"],
28: ["01:198:214","01:198:336","01:198:213","01:198:210"],
29: ["01:198:210","01:198:336","01:198:214","01:198:443"],
30: ["01:198:213","01:198:314","01:198:214","01:198:443"],
31: ["01:198:314","01:198:336","01:198:214","01:198:443","01:198:442"],
32: ["01:198:443"],
33: ["01:198:210","01:198:336","01:198:214","01:198:314","01:198:443"],
34: ["01:198:214","01:198:210","01:198:443"],
35: ["01:198:213","01:198:336","01:198:210","01:198:442"],
36: ["01:198:214","01:198:314"],
37: ["01:198:210","01:198:336","01:198:214","01:198:442"],
38: ["01:198:314","01:198:214","01:198:442"],
39: ["01:198:210","01:198:336","01:198:443"],
40: ["01:198:314","01:198:452","01:198:214","01:198:443"],
41: ["01:198:314","01:198:323","01:198:443","01:198:442"],
42: ["01:198:336","01:198:314","01:198:214","01:198:443"],
43: ["01:198:214","01:198:314","01:198:210","01:198:443"],
44: ["01:198:452","01:198:314","01:198:213","01:198:443"],
45: ["01:198:214","01:198:442","01:198:443"],
46: ["01:198:210","01:198:442"],
47: ["01:198:214","01:198:314","01:198:443"],
48: ["01:198:336","01:198:210"],
49: ["01:198:314","01:198:336","01:198:214","01:198:443"],
50: ["01:198:214","01:198:443","01:198:336","01:198:213","01:198:442"],
}

course_freq = Counter()
for codes in raw_responses.values():
    course_freq.update(codes)

course_freq_df = pd.DataFrame(
    [(course_map.get(k, k), v) for k, v in course_freq.most_common()],
    columns=["Course", "Frequency"]
)

total_q = len(raw_responses)
palette = [C_RED, C_ORANGE, C_BLUE, C_GREEN, C_PURPLE,
           C_TEAL, C_PINK, C_YELLOW, C_GRAY, C_DARK, "#f39c12", "#16a085"]
bar_colors = [palette[i % len(palette)] for i in range(len(course_freq_df))]

fig6, ax6 = plt.subplots(figsize=(13, 7))
bars6 = ax6.barh(course_freq_df["Course"], course_freq_df["Frequency"],
                 color=bar_colors, edgecolor="white", height=0.65)
ax6.set_xlabel("Number of Queries Recommending This Course")
ax6.set_title(
    "Course Recommendation Frequency\n"
    "(Agent Framework Convergence — n = 50 queries)",
    fontweight="bold", fontsize=13)
ax6.invert_yaxis()

for bar, val in zip(bars6, course_freq_df["Frequency"]):
    pct = val / total_q * 100
    ax6.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
             f"{val}  ({pct:.0f}%)", va="center", fontweight="bold", fontsize=9)

ax6.set_xlim(0, course_freq_df["Frequency"].max() + 7)

plt.tight_layout()
fig6.savefig(FIGURES_DIR / "fig6_course_convergence.png", dpi=150, bbox_inches="tight")
print("Saved fig6_course_convergence.png")

# ══════════════════════════════════════════════════════════════════════════════
# Print final numeric summaries
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== TOKEN STATS ===")
for col in ["input_tokens", "output_tokens", "total_tokens"]:
    s = df[col]
    print(f"{col:20s}  mean={s.mean():8.1f}  var={s.var():10.1f}  "
          f"std={s.std():7.1f}  min={s.min()}  max={s.max()}")

print("\n=== RESPONSE TIME ===")
rt = df["response_time_sec"]
print(f"mean={rt.mean():.2f}s  std={rt.std():.2f}s  "
      f"min={rt.min():.2f}s  max={rt.max():.2f}s")
print(f"Outliers (>80s): {(rt>80).sum()} — "
      f"{df[rt>80]['query'].str[:55].tolist()}")

print("\n=== COURSE FREQUENCY (all) ===")
for code, cnt in course_freq.most_common():
    name = course_map.get(code, code).replace("\n", " ")
    print(f"  {code}  {cnt:3d}x  ({cnt/total_q*100:.0f}%)   {name}")

print("\nAll 5 figures saved.")
