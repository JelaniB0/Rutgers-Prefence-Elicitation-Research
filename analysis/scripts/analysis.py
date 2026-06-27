import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Load data
df = pd.read_csv("query_log.csv", header=0)
df.columns = [
    "session_id", "timestamp", "response_time_sec", "query", "response",
    "plan_steps", "agents_invoked", "sources_and_tools",
    "input_tokens", "output_tokens", "satisfied", "feedback"
]

df["response_time_sec"] = pd.to_numeric(df["response_time_sec"], errors="coerce")
df["input_tokens"]      = pd.to_numeric(df["input_tokens"],      errors="coerce")
df["output_tokens"]     = pd.to_numeric(df["output_tokens"],     errors="coerce")

# Filter: 05/18 multi-agent only (non-crashed, plan_steps present) 
df_0518 = df[df["session_id"].str.startswith("2026-05-18")].copy()

df_multi = df_0518[
    df_0518["response_time_sec"] > 0
].copy()

total_tokens = df_multi["input_tokens"] + df_multi["output_tokens"]

# Console output 
print(f"05/18 rows total:       {len(df_0518)}")
print(f"05/18 multi-agent rows: {len(df_multi)}")

rt = df_multi["response_time_sec"].dropna()
print(f"\n── Response Time (seconds) ──────────────────")
print(f"  n        : {len(rt)}")
print(f"  Mean     : {rt.mean():.2f}s")
print(f"  Median   : {rt.median():.2f}s")
print(f"  Std Dev  : {rt.std():.2f}s")
print(f"  Variance : {rt.var():.2f}")
print(f"  Min      : {rt.min():.2f}s")
print(f"  Max      : {rt.max():.2f}s")

print(f"\n── Input Tokens ─────────────────────────────")
print(f"  Mean     : {df_multi['input_tokens'].mean():.2f}")
print(f"  Median   : {df_multi['input_tokens'].median():.2f}")
print(f"  Std Dev  : {df_multi['input_tokens'].std():.2f}")
print(f"  Variance : {df_multi['input_tokens'].var():.2f}")

print(f"\n── Output Tokens ────────────────────────────")
print(f"  Mean     : {df_multi['output_tokens'].mean():.2f}")
print(f"  Median   : {df_multi['output_tokens'].median():.2f}")
print(f"  Std Dev  : {df_multi['output_tokens'].std():.2f}")
print(f"  Variance : {df_multi['output_tokens'].var():.2f}")

print(f"\n── Total Tokens (input + output) ────────────")
print(f"  Mean     : {total_tokens.mean():.2f}")
print(f"  Median   : {total_tokens.median():.2f}")
print(f"  Std Dev  : {total_tokens.std():.2f}")
print(f"  Variance : {total_tokens.var():.2f}")

# Figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Multi-Agent Framework Analysis (First Iteration)",
    fontsize=14, fontweight="bold", y=1.01
)

# Plot 1: Box & Whisker — Response Time 
ax1 = axes[0]
bp = ax1.boxplot(
    rt,
    patch_artist=True,
    widths=0.4,
    medianprops=dict(color="#e74c3c", linewidth=2.5),
    boxprops=dict(facecolor="#aed6f1", color="#2980b9", linewidth=1.5),
    whiskerprops=dict(color="#2980b9", linewidth=1.5, linestyle="--"),
    capprops=dict(color="#2980b9", linewidth=1.5),
    flierprops=dict(marker="o", color="#e74c3c", markersize=7, alpha=0.75,
                    markerfacecolor="#e74c3c")
)

# Jittered individual points
np.random.seed(42)
jitter = np.random.uniform(-0.07, 0.07, size=len(rt))
ax1.scatter(1 + jitter, rt, alpha=0.5, color="#2471a3", s=35, zorder=3,
            label="Individual queries")

# Stats annotation
ax1.text(
    1.30, rt.max() * 0.99,
    f"n = {len(rt)}\n"
    f"Mean   = {rt.mean():.1f}s\n"
    f"Median = {rt.median():.1f}s\n"
    f"Std    = {rt.std():.1f}s\n"
    f"Max    = {rt.max():.1f}s\n"
    f"Min    = {rt.min():.1f}s",
    fontsize=8.5, va="top",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#aaa", alpha=0.9)
)

ax1.set_xticks([1])
ax1.set_xticklabels(["Multi-Agent Queries"], fontsize=10)
ax1.set_ylabel("Response Time (seconds)", fontsize=11)
ax1.set_title("Response Time Distribution", fontsize=12, fontweight="bold")
ax1.yaxis.grid(True, linestyle="--", alpha=0.5)
ax1.set_axisbelow(True)
ax1.legend(fontsize=8, loc="upper left")

# Plot 2: Token Usage Bar Chart 
ax2 = axes[1]

labels  = ["Input Tokens", "Output Tokens", "Total Tokens"]
means   = [df_multi["input_tokens"].mean(),
           df_multi["output_tokens"].mean(),
           total_tokens.mean()]
stds    = [df_multi["input_tokens"].std(),
           df_multi["output_tokens"].std(),
           total_tokens.std()]
vars_   = [df_multi["input_tokens"].var(),
           df_multi["output_tokens"].var(),
           total_tokens.var()]
colors  = ["#5dade2", "#58d68d", "#f0b27a"]

x = np.arange(len(labels))
bars = ax2.bar(x, means, width=0.5, color=colors, edgecolor="white",
               linewidth=1.2, alpha=0.88, zorder=3)

# Error bars ±1 std
ax2.errorbar(x, means, yerr=stds, fmt="none", color="#333",
             capsize=7, linewidth=1.8, zorder=4)

# Annotate each bar, place label just above the error bar cap, centered
for i, (bar, mean, std, var) in enumerate(zip(bars, means, stds, vars_)):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        mean + std + max(means) * 0.12,
        f"μ = {mean:,.0f}\nσ = {std:,.0f}\nσ² = {var:,.0f}",
        ha="center", va="bottom", fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#ccc", alpha=0.9)
    )

ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=10)
ax2.set_ylabel("Token Count", fontsize=11)
ax2.set_title("Token Usage", fontsize=12, fontweight="bold")
ax2.yaxis.grid(True, linestyle="--", alpha=0.5)
ax2.set_axisbelow(True)

# Extra y headroom so annotations don't hit the top edge
current_top = ax2.get_ylim()[1]
ax2.set_ylim(0, current_top * 1.45)

patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
ax2.legend(handles=patches, fontsize=8, loc="lower right")

plt.tight_layout()
out = "query_analysis_0518.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nPlot saved -> {out}")