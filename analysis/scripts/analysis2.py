import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Load 
rows = []
with open("query_log2.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) >= 12:
            rows.append(row[:12])

df = pd.DataFrame(rows[1:], columns=[
    "session_id", "timestamp", "response_time_sec", "query", "response",
    "plan_steps", "agents_invoked", "sources_and_tools",
    "input_tokens", "output_tokens", "satisfied", "feedback"
])

df["response_time_sec"] = pd.to_numeric(df["response_time_sec"], errors="coerce")
df["input_tokens"]      = pd.to_numeric(df["input_tokens"],      errors="coerce")
df["output_tokens"]     = pd.to_numeric(df["output_tokens"],     errors="coerce")

# Filter to exact sessions 
target_sessions = {
    "2026-05-25 18:37:37", "2026-05-25 18:47:14",
    "2026-05-26 14:11:22", "2026-05-26 14:33:38",
    "2026-05-26 14:39:00", "2026-05-26 15:18:28",
    "2026-05-26 17:19:39", "2026-05-26 17:32:43"
}

df_valid = df[
    df["session_id"].isin(target_sessions) &
    (df["response_time_sec"] > 0) &
    df["input_tokens"].notna() &
    df["output_tokens"].notna()
].copy()

df_valid["agent_count"] = df_valid["agents_invoked"].apply(
    lambda x: len(str(x).split("|")) if pd.notna(x) and x != "" else 0
)

total_tokens = df_valid["input_tokens"] + df_valid["output_tokens"]
rt = df_valid["response_time_sec"].dropna()

# Console output 
print(f"Total rows loaded: {len(df)}  |  Filtered valid rows: {len(df_valid)}")

print(f"\n── Response Time (seconds) ──────────────────")
print(f"  n={len(rt)}  Mean={rt.mean():.2f}s  Median={rt.median():.2f}s  Std={rt.std():.2f}s  Min={rt.min():.2f}s  Max={rt.max():.2f}s")

print(f"\n── Token Usage ──────────────────────────────")
for label, series in [("Input", df_valid["input_tokens"]), ("Output", df_valid["output_tokens"]), ("Total", total_tokens)]:
    print(f"  {label:6s}  Mean={series.mean():,.0f}  Std={series.std():,.0f}")

grouped = df_valid.groupby("agent_count").agg(
    n=("input_tokens", "count"),
    mean_input=("input_tokens", "mean"),
    std_input=("input_tokens", "std"),
    mean_output=("output_tokens", "mean"),
    std_output=("output_tokens", "std"),
    mean_rt=("response_time_sec", "mean"),
).reset_index()
grouped["mean_total"] = grouped["mean_input"] + grouped["mean_output"]
grouped["std_total"]  = np.sqrt(grouped["std_input"]**2 + grouped["std_output"]**2)

print(f"\n── Agent Count Distribution ─────────────────")
print(grouped.to_string(index=False))

agent_counts = sorted(df_valid["agent_count"].unique())
rt_groups    = [df_valid[df_valid["agent_count"] == n]["response_time_sec"].dropna().values for n in agent_counts]
agent_labels = [f"{int(n)} agent{'s' if n != 1 else ''}\n(n={int(c)})"
                for n, c in zip(grouped["agent_count"], grouped["n"])]
x            = np.arange(len(grouped))
blue_shades  = [(*plt.cm.Blues(a)[:3],) for a in np.linspace(0.4, 0.95, len(agent_counts))]

# Figure 1: Response Time Distribution + Token Usage Overview
fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))
fig1.suptitle("Multi-Agent Framework Analysis (Second Iteration)", fontsize=14, fontweight="bold", y=1.01)

# Plot 1a: Box & Whisker: Response Time
ax = axes1[0]
ax.boxplot(rt, patch_artist=True, widths=0.4,
           medianprops=dict(color="#e74c3c", linewidth=2.5),
           boxprops=dict(facecolor="#aed6f1", color="#2980b9", linewidth=1.5),
           whiskerprops=dict(color="#2980b9", linewidth=1.5, linestyle="--"),
           capprops=dict(color="#2980b9", linewidth=1.5),
           flierprops=dict(marker="", markersize=0))  # hide fliers — drawn manually below
np.random.seed(42)
ax.scatter(1 + np.random.uniform(-0.07, 0.07, size=len(rt)), rt,
           alpha=0.5, color="#2471a3", s=35, zorder=3, label="Individual queries")
ax.text(1.30, rt.max() * 0.99,
        f"n = {len(rt)}\nMean   = {rt.mean():.1f}s\nMedian = {rt.median():.1f}s\n"
        f"Std    = {rt.std():.1f}s\nMax    = {rt.max():.1f}s\nMin    = {rt.min():.1f}s",
        fontsize=8.5, va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#aaa", alpha=0.9))
ax.set_xticks([1]); ax.set_xticklabels(["Multi-Agent Queries"], fontsize=10)
ax.set_ylabel("Response Time (seconds)", fontsize=11)
ax.set_title("Response Time Distribution", fontsize=12, fontweight="bold")
ax.yaxis.grid(True, linestyle="--", alpha=0.5); ax.set_axisbelow(True)
ax.legend(fontsize=8, loc="upper left")

# Plot 1b: Token Usage Bar Chart
ax = axes1[1]
tok_labels = ["Input Tokens", "Output Tokens", "Total Tokens"]
means  = [df_valid["input_tokens"].mean(), df_valid["output_tokens"].mean(), total_tokens.mean()]
stds   = [df_valid["input_tokens"].std(),  df_valid["output_tokens"].std(),  total_tokens.std()]
vars_  = [df_valid["input_tokens"].var(),  df_valid["output_tokens"].var(),  total_tokens.var()]
colors = ["#5dade2", "#58d68d", "#f0b27a"]
xi = np.arange(len(tok_labels))
bars = ax.bar(xi, means, width=0.5, color=colors, edgecolor="white", linewidth=1.2, alpha=0.88, zorder=3)
ax.errorbar(xi, means, yerr=stds, fmt="none", color="#333", capsize=7, linewidth=1.8, zorder=4)
for bar, mean, std, var in zip(bars, means, stds, vars_):
    ax.text(bar.get_x() + bar.get_width()/2, mean + std + max(means)*0.12,
            f"μ = {mean:,.0f}\nσ = {std:,.0f}\nσ² = {var:,.0f}",
            ha="center", va="bottom", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc", alpha=0.9))
ax.set_xticks(xi); ax.set_xticklabels(tok_labels, fontsize=10)
ax.set_ylabel("Token Count", fontsize=11)
ax.set_title("Token Usage", fontsize=12, fontweight="bold")
ax.yaxis.grid(True, linestyle="--", alpha=0.5); ax.set_axisbelow(True)
ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
ax.legend(handles=[mpatches.Patch(color=c, label=l) for c, l in zip(colors, tok_labels)],
          fontsize=8, loc="lower right")

plt.tight_layout()
plt.savefig("query_analysis_2.png", dpi=150, bbox_inches="tight")
print("\nSaved → query_analysis_2.png")

# Figure 2: Response Time by Agent Count
fig2, ax = plt.subplots(figsize=(11, 6))

bp = ax.boxplot(rt_groups, patch_artist=True, widths=0.45,
                medianprops=dict(color="none", linewidth=0),
                whiskerprops=dict(color="#555", linewidth=1.4, linestyle="--"),
                capprops=dict(color="#555", linewidth=1.4),
                flierprops=dict(marker="", markersize=0))  # hide fliers, drawn manually below
for patch, color in zip(bp["boxes"], blue_shades):
    patch.set_facecolor(color); patch.set_edgecolor("#2471a3"); patch.set_linewidth(1.4)

np.random.seed(42)
for i, group in enumerate(rt_groups, start=1):
    ax.scatter(i + np.random.uniform(-0.12, 0.12, size=len(group)),
               group, alpha=0.55, color="#27ae60", s=30, zorder=3)

for i, (group, n) in enumerate(zip(rt_groups, grouped["n"]), start=1):
    mean = np.mean(group) if len(group) > 0 else 0
    ax.text(i, -18, f"n={n}\nμ={mean:.1f}s", ha="center", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc", alpha=0.85))

rt_means = [np.mean(g) for g in rt_groups if len(g) > 0]
xs = [i + 1 for i, g in enumerate(rt_groups) if len(g) > 0]
ax.plot(xs, rt_means, color="#e74c3c", linewidth=1.8, linestyle="-",
        marker="D", markersize=6, zorder=5)

ax.set_xticks(range(1, len(agent_counts) + 1))
ax.set_xticklabels(agent_labels, fontsize=10)
ax.set_xlabel("Number of Agents Invoked", fontsize=11)
ax.set_ylabel("Response Time (seconds)", fontsize=11)
ax.set_title("Response Time by Number of Agents Invoked (Second Iteration)", fontsize=13, fontweight="bold")
ax.yaxis.grid(True, linestyle="--", alpha=0.5); ax.set_axisbelow(True)
ax.set_ylim(-30, df_valid["response_time_sec"].max() * 1.1)
ax.legend(handles=[
    plt.Line2D([0], [0], color="#e74c3c", linewidth=1.8, marker="D", markersize=6, label="Mean response time"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#27ae60", markersize=7, alpha=0.7, label="Individual query"),
    mpatches.Patch(color="none", label="n = queries in group"),
    mpatches.Patch(color="none", label="μ = mean response time (s)"),
], fontsize=8.5, loc="upper left", framealpha=0.9)

plt.tight_layout()
plt.savefig("runtime_by_agents_2.png", dpi=150, bbox_inches="tight")
print("Saved -> runtime_by_agents_2.png")

# Figure 3: Token Usage by Agent Count
fig3, axes3 = plt.subplots(1, 2, figsize=(15, 6))
fig3.suptitle("Token Usage by Number of Agents Invoked (Second Iteration)", fontsize=14, fontweight="bold", y=1.01)

width        = 0.35
input_color  = "#5dade2"
output_color = "#58d68d"
total_color  = "#f0b27a"

# Plot 3a: Grouped bar, Input vs Output
ax = axes3[0]
bars_in  = ax.bar(x - width/2, grouped["mean_input"],  width, color=input_color,
                  edgecolor="white", linewidth=1.2, alpha=0.88, label="Input Tokens",  zorder=3)
bars_out = ax.bar(x + width/2, grouped["mean_output"], width, color=output_color,
                  edgecolor="white", linewidth=1.2, alpha=0.88, label="Output Tokens", zorder=3)
ax.errorbar(x - width/2, grouped["mean_input"],  yerr=grouped["std_input"],
            fmt="none", color="#333", capsize=5, linewidth=1.5, zorder=4)
ax.errorbar(x + width/2, grouped["mean_output"], yerr=grouped["std_output"],
            fmt="none", color="#333", capsize=5, linewidth=1.5, zorder=4)
for bar, mean, std in zip(bars_in, grouped["mean_input"], grouped["std_input"]):
    ax.text(bar.get_x() + bar.get_width()/2, mean + std + 200, f"{mean:,.0f}",
            ha="center", va="bottom", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#ccc", alpha=0.85))
for bar, mean, std in zip(bars_out, grouped["mean_output"], grouped["std_output"]):
    ax.text(bar.get_x() + bar.get_width()/2, mean + std + 200, f"{mean:,.0f}",
            ha="center", va="bottom", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#ccc", alpha=0.85))
ax.set_xticks(x); ax.set_xticklabels(agent_labels, fontsize=9)
ax.set_xlabel("Number of Agents Invoked", fontsize=11)
ax.set_ylabel("Mean Token Count", fontsize=11)
ax.set_title("Avg Input vs Output Tokens\nby Agent Count", fontsize=12, fontweight="bold")
ax.yaxis.grid(True, linestyle="--", alpha=0.5); ax.set_axisbelow(True)
ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
ax.legend(fontsize=9)

# Plot 3b: Total token trend line
ax = axes3[1]
ax.fill_between(x, grouped["mean_total"] - grouped["std_total"],
                grouped["mean_total"] + grouped["std_total"],
                alpha=0.2, color=total_color, label="±1 Std Dev band")
ax.plot(x, grouped["mean_total"], color=total_color, linewidth=2.2,
        marker="o", markersize=8, markerfacecolor="#d35400", zorder=5, label="Mean Total Tokens")
ax.plot(x, grouped["mean_input"],  color=input_color,  linewidth=1.6,
        marker="s", markersize=6, linestyle="--", label="Mean Input Tokens")
ax.plot(x, grouped["mean_output"], color=output_color, linewidth=1.6,
        marker="^", markersize=6, linestyle="--", label="Mean Output Tokens")
for xi_val, (total, std) in enumerate(zip(grouped["mean_total"], grouped["std_total"])):
    ax.text(xi_val, total + std + 200, f"{total:,.0f}", ha="center", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#ccc", alpha=0.9))
ax.set_xticks(x); ax.set_xticklabels(agent_labels, fontsize=9)
ax.set_xlabel("Number of Agents Invoked", fontsize=11)
ax.set_ylabel("Mean Token Count", fontsize=11)
ax.set_title("Total Token Usage Trend\nby Agent Count", fontsize=12, fontweight="bold")
ax.yaxis.grid(True, linestyle="--", alpha=0.5); ax.set_axisbelow(True)
ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
ax.legend(fontsize=8.5, loc="upper left")

plt.tight_layout()
plt.savefig("tokens_by_agents_2.png", dpi=150, bbox_inches="tight")
print("Saved -> tokens_by_agents_2.png")