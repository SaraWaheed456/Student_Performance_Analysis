
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load Cleaned Dataset ─────────────────────────────────────
df = pd.read_csv("data/cleaned_students.csv")

# Set a consistent, professional style for all plots
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 130       # Higher resolution output
plt.rcParams["font.family"] = "sans-serif"


# ── PLOT 1: Histogram of Math Score with KDE ─────────────────
# KDE (Kernel Density Estimate) = smooth curve showing distribution shape
fig, ax = plt.subplots(figsize=(8, 5))

sns.histplot(
    data=df,
    x="math_score",
    kde=True,            # Overlay the KDE curve on the histogram
    bins=20,             # Divide scores into 20 bins
    color="steelblue",
    edgecolor="white",
    linewidth=0.6,
    ax=ax
)

# Add vertical lines for mean and median so they're easy to compare
ax.axvline(df["math_score"].mean(),   color="red",    linestyle="--", lw=2, label=f"Mean   = {df['math_score'].mean():.1f}")
ax.axvline(df["math_score"].median(), color="orange", linestyle="-.", lw=2, label=f"Median = {df['math_score'].median():.1f}")

ax.set_title("Distribution of Math Scores (with KDE)", fontsize=13, fontweight="bold")
ax.set_xlabel("Math Score", fontsize=11)
ax.set_ylabel("Count of Students", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("task5_plot1_histogram.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot 1 saved: task5_plot1_histogram.png")


# ── PLOT 2: Boxplot of Math Score by Gender ───────────────────
# Boxplot shows: median (center line), IQR (box), whiskers, and outliers (dots)
fig, ax = plt.subplots(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="gender",
    y="math_score",
    palette={"male": "skyblue", "female": "lightcoral"},  # gender-appropriate colors
    width=0.5,
    linewidth=1.5,
    ax=ax
)

# Overlay actual data points as a strip to see data density
sns.stripplot(
    data=df,
    x="gender",
    y="math_score",
    color="black",
    alpha=0.2,    # 20% opacity so overlapping points are visible
    size=3,
    jitter=True,  # Spread points horizontally so they don't overlap
    ax=ax
)

ax.set_title("Math Score Distribution by Gender", fontsize=13, fontweight="bold")
ax.set_xlabel("Gender", fontsize=11)
ax.set_ylabel("Math Score", fontsize=11)
ax.grid(True, axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("task5_plot2_boxplot.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot 2 saved: task5_plot2_boxplot.png")


# ── PLOT 3: Scatter Plot (Reading vs Writing, colored by Gender) ──
fig, ax = plt.subplots(figsize=(8, 6))

# Plot each gender separately so we get a proper legend
colors = {"male": "royalblue", "female": "crimson"}
for gender, group in df.groupby("gender"):
    ax.scatter(
        group["reading_score"],
        group["writing_score"],
        label=gender.capitalize(),
        alpha=0.5,            # Transparency to handle overlap
        s=40,                 # Marker size
        color=colors[gender],
        edgecolors="white",
        linewidth=0.4
    )

# Correlation annotation
r = df["reading_score"].corr(df["writing_score"])
ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes,
        fontsize=11, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

ax.set_title("Reading Score vs Writing Score (by Gender)", fontsize=13, fontweight="bold")
ax.set_xlabel("Reading Score", fontsize=11)
ax.set_ylabel("Writing Score", fontsize=11)
ax.legend(title="Gender", fontsize=10)
ax.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("task5_plot3_scatter.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot 3 saved: task5_plot3_scatter.png")


# ── PLOT 4: Bar Chart — Avg Math Score by Parental Education ──
fig, ax = plt.subplots(figsize=(9, 5))

# Compute average math score for each parental education level,
# then sort from highest to lowest for a cleaner presentation
avg_by_edu = (
    df.groupby("parental_education")["math_score"]
    .mean()
    .sort_values(ascending=False)
    .round(2)
)

bars = ax.bar(
    avg_by_edu.index,
    avg_by_edu.values,
    color=sns.color_palette("Blues_d", len(avg_by_edu)),
    edgecolor="black",
    linewidth=0.7
)

# Add value labels on top of each bar
for bar, val in zip(bars, avg_by_edu.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,  # Horizontal center of bar
        bar.get_height() + 0.5,             # Just above the bar top
        f"{val:.1f}",
        ha="center", va="bottom", fontsize=9, fontweight="bold"
    )

ax.set_title("Average Math Score by Parental Education Level", fontsize=13, fontweight="bold")
ax.set_xlabel("Parental Education Level", fontsize=11)
ax.set_ylabel("Average Math Score", fontsize=11)
ax.set_xticklabels(avg_by_edu.index, rotation=30, ha="right", fontsize=9)
ax.set_ylim(0, avg_by_edu.max() + 10)   # Extra headroom for labels
ax.grid(True, axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("task5_plot4_barchart.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot 4 saved: task5_plot4_barchart.png")


# ── PLOT 5: Combined 2x2 Figure ──────────────────────────────
# This is the "best_visualization.png" the assignment asks for.
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Student Performance Analysis — Combined Dashboard",
    fontsize=15, fontweight="bold", y=1.01
)

# --- Subplot (0,0): Histogram with KDE ---
sns.histplot(data=df, x="math_score", kde=True, bins=20,
             color="steelblue", edgecolor="white", ax=axes[0, 0])
axes[0, 0].axvline(df["math_score"].mean(),   color="red",    linestyle="--", lw=1.5,
                   label=f"Mean={df['math_score'].mean():.1f}")
axes[0, 0].axvline(df["math_score"].median(), color="orange", linestyle="-.", lw=1.5,
                   label=f"Median={df['math_score'].median():.1f}")
axes[0, 0].set_title("Math Score Distribution (KDE)")
axes[0, 0].set_xlabel("Math Score")
axes[0, 0].set_ylabel("Count")
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(True, linestyle="--", alpha=0.5)

# --- Subplot (0,1): Boxplot by Gender ---
sns.boxplot(data=df, x="gender", y="math_score",
            palette={"male": "skyblue", "female": "lightcoral"},
            ax=axes[0, 1])
axes[0, 1].set_title("Math Score by Gender")
axes[0, 1].set_xlabel("Gender")
axes[0, 1].set_ylabel("Math Score")
axes[0, 1].grid(True, axis="y", linestyle="--", alpha=0.5)

# --- Subplot (1,0): Scatter Reading vs Writing ---
for gender, group in df.groupby("gender"):
    axes[1, 0].scatter(
        group["reading_score"], group["writing_score"],
        label=gender.capitalize(), alpha=0.4, s=25,
        color=colors[gender], edgecolors="white", linewidth=0.3
    )
axes[1, 0].set_title("Reading vs Writing (by Gender)")
axes[1, 0].set_xlabel("Reading Score")
axes[1, 0].set_ylabel("Writing Score")
axes[1, 0].legend(title="Gender", fontsize=8)
axes[1, 0].grid(True, linestyle="--", alpha=0.5)

# --- Subplot (1,1): Bar Chart by Parental Education ---
axes[1, 1].bar(
    range(len(avg_by_edu)),
    avg_by_edu.values,
    color=sns.color_palette("Blues_d", len(avg_by_edu)),
    edgecolor="black", linewidth=0.6
)
axes[1, 1].set_xticks(range(len(avg_by_edu)))
axes[1, 1].set_xticklabels(avg_by_edu.index, rotation=35, ha="right", fontsize=7.5)
axes[1, 1].set_title("Avg Math Score by Parental Education")
axes[1, 1].set_xlabel("Parental Education Level")
axes[1, 1].set_ylabel("Average Math Score")
axes[1, 1].grid(True, axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("best_visualization.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n✓ Final combined figure saved as: best_visualization.png")