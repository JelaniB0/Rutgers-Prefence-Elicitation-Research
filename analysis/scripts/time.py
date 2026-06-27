import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Load 
df = pd.read_csv("query_log.csv", header=0)
df.columns = [
    "session_id", "timestamp", "response_time_sec", "query", "response",
    "plan_steps", "agents_invoked", "sources_and_tools",
    "input_tokens", "output_tokens", "satisfied", "feedback"
]

df["response_time_sec"] = pd.to_numeric(df["response_time_sec"], errors="coerce")
df["input_tokens"]      = pd.to_numeric(df["input_tokens"],      errors="coerce")
df["output_tokens"]     = pd.to_numeric(df["output_tokens"],     errors="coerce")

# Filter: 05/18, valid response time 
df_0518 = df[
    df["session_id"].str.startswith("2026-05-18") &
    (df["response_time_sec"] > 0)
].copy()

# Count agents invoked per row 
df_0518["agent_count"] = df_0518["agents_invoked"].apply(
    lambda x: len(str(x).split("|")) if pd.notna(x) and x != "" else 0
)

# Console summary 
print("Agent count distribution:")
print(df_0518.groupby("agent_count")["response_time_sec"].describe().round(2))

# Build box plot groups
agent_counts = sorted(df_0518["agent_count"].unique())
groups = [df_0518[df_0518["agent_count"] == n]["response_time_sec"].dropna().values
          for n in agent_counts]
labels = [f"{n} agent{'s' if n != 1 else ''}" for n in agent_counts]
counts = [len(g) for g in groups]

# Color map — more agents = darker
colors = [(*plt.cm.Blues(a)[:3],) for a in np.linspace(0.4, 0.95, len(agent_counts))]

#  Figure 1: Response Time by Agent Count 
fig, ax = plt.subplots(figsize=(11, 6))

bp = ax.boxplot(
    groups,
    patch_artist=True,
    widths=0.45,
    medianprops=dict(color="none", linewidth=0),
    whiskerprops=dict(color="#555", linewidth=1.4, linestyle="--"),
    capprops=dict(color="#555", linewidth=1.4),
    flierprops=dict(marker="o", markersize=6, alpha=0.7,
                    markerfacecolor="#27ae60",
                    markeredgecolor="#27ae60")
)

for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_edgecolor("#2471a3")
    patch.set_linewidth(1.4)

# Jittered scatter overlay
np.random.seed(42)
for i, (group, color) in enumerate(zip(groups, colors), start=1):
    jitter = np.random.uniform(-0.12, 0.12, size=len(group))
    ax.scatter(i + jitter, group, alpha=0.55, color="#27ae60", s=30, zorder=3)

# Annotate n and mean under each box
for i, (group, label, n) in enumerate(zip(groups, labels, counts), start=1):
    mean = np.mean(group) if len(group) > 0 else 0
    ax.text(i, -18, f"n={n}\nμ={mean:.1f}s",
            ha="center", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#ccc", alpha=0.85))

# Trend line (mean per group)
means = [np.mean(g) for g in groups if len(g) > 0]
xs = [i + 1 for i, g in enumerate(groups) if len(g) > 0]
ax.plot(xs, means, color="#e74c3c", linewidth=1.8, linestyle="-",
        marker="D", markersize=6, zorder=5, label="Mean response time")

ax.set_xticks(range(1, len(agent_counts) + 1))
ax.set_xticklabels(labels, fontsize=10)
ax.set_xlabel("Number of Agents Invoked", fontsize=11)
ax.set_ylabel("Response Time (seconds)", fontsize=11)
ax.set_title("Response Time by Number of Agents Invoked (First Iteration)",
             fontsize=13, fontweight="bold")
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
ax.set_ylim(-30, df_0518["response_time_sec"].max() * 1.1)

# Legend
mean_line = plt.Line2D([0], [0], color="#e74c3c", linewidth=1.8,
                       linestyle="-", marker="D", markersize=6, label="Mean response time")
green_dot = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#27ae60",
                       markersize=7, alpha=0.7, label="Individual query (incl. outliers)")
n_patch   = mpatches.Patch(color="none", label="n = number of queries in group")
mu_patch  = mpatches.Patch(color="none", label="μ = mean response time (seconds)")

ax.legend(handles=[mean_line, green_dot, n_patch, mu_patch],
          fontsize=8.5, loc="upper left", framealpha=0.9)

plt.tight_layout()
out = "runtime_by_agents.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nPlot saved → {out}")

# Figure 2: Token Usage by Agent Count 
# Build per-agent-count token stats (only rows with valid token data)
df_tokens = df_0518[
    df_0518["input_tokens"].notna() &
    df_0518["output_tokens"].notna()
].copy()

grouped = df_tokens.groupby("agent_count").agg(
    n=("input_tokens", "count"),
    mean_input=("input_tokens", "mean"),
    std_input=("input_tokens", "std"),
    mean_output=("output_tokens", "mean"),
    std_output=("output_tokens", "std"),
).reset_index()
grouped["mean_total"] = grouped["mean_input"] + grouped["mean_output"]
grouped["std_total"]  = np.sqrt(grouped["std_input"].fillna(0)**2 +
                                grouped["std_output"].fillna(0)**2)

# x-axis labels reuse the same agent_counts order, aligned to grouped
token_agent_labels = [
    f"{int(row.agent_count)} agent{'s' if row.agent_count != 1 else ''}\n(n={int(row.n)})"
    for row in grouped.itertuples()
]
x = np.arange(len(grouped))

input_color  = "#5dade2"
output_color = "#58d68d"
total_color  = "#f0b27a"
width        = 0.35

fig3, axes3 = plt.subplots(1, 2, figsize=(15, 6))
fig3.suptitle("Token Usage by Number of Agents Invoked  (First Iteration)",
              fontsize=14, fontweight="bold", y=1.01)

# Plot 2a: Grouped bar, Input vs Output
ax = axes3[0]
bars_in  = ax.bar(x - width/2, grouped["mean_input"],  width, color=input_color,
                  edgecolor="white", linewidth=1.2, alpha=0.88, label="Input Tokens",  zorder=3)
bars_out = ax.bar(x + width/2, grouped["mean_output"], width, color=output_color,
                  edgecolor="white", linewidth=1.2, alpha=0.88, label="Output Tokens", zorder=3)
ax.errorbar(x - width/2, grouped["mean_input"],  yerr=grouped["std_input"].fillna(0),
            fmt="none", color="#333", capsize=5, linewidth=1.5, zorder=4)
ax.errorbar(x + width/2, grouped["mean_output"], yerr=grouped["std_output"].fillna(0),
            fmt="none", color="#333", capsize=5, linewidth=1.5, zorder=4)
for bar, mean, std in zip(bars_in,  grouped["mean_input"],  grouped["std_input"].fillna(0)):
    ax.text(bar.get_x() + bar.get_width()/2, mean + std + 200, f"{mean:,.0f}",
            ha="center", va="bottom", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#ccc", alpha=0.85))
for bar, mean, std in zip(bars_out, grouped["mean_output"], grouped["std_output"].fillna(0)):
    ax.text(bar.get_x() + bar.get_width()/2, mean + std + 200, f"{mean:,.0f}",
            ha="center", va="bottom", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#ccc", alpha=0.85))
ax.set_xticks(x)
ax.set_xticklabels(token_agent_labels, fontsize=9)
ax.set_xlabel("Number of Agents Invoked", fontsize=11)
ax.set_ylabel("Mean Token Count", fontsize=11)
ax.set_title("Avg Input vs Output Tokens\nby Agent Count", fontsize=12, fontweight="bold")
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
ax.legend(fontsize=9)

# Plot 2b: Total token trend line
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
    ax.text(xi_val, total + std + 200, f"{total:,.0f}",
            ha="center", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#ccc", alpha=0.9))
ax.set_xticks(x)
ax.set_xticklabels(token_agent_labels, fontsize=9)
ax.set_xlabel("Number of Agents Invoked", fontsize=11)
ax.set_ylabel("Mean Token Count", fontsize=11)
ax.set_title("Total Token Usage Trend\nby Agent Count", fontsize=12, fontweight="bold")
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
ax.legend(fontsize=8.5, loc="upper left")

plt.tight_layout()
out2 = "tokens_by_agents.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Plot saved -> {out2}")