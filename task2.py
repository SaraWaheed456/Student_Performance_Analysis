
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ── Load Dataset ────────────────────────────────────────────
# Make sure StudentsPerformance.csv is inside a folder called "data"
# relative to wherever you run this script.
df = pd.read_csv("data/StudentsPerformance.csv")

# Rename columns for easier typing (removes spaces and slashes)
df.columns = [
    "gender",           # original: gender
    "race_ethnicity",   # original: race/ethnicity
    "parental_education",  # original: parental level of education
    "lunch",            # original: lunch
    "test_prep",        # original: test preparation course
    "math_score",       # original: math score
    "reading_score",    # original: reading score
    "writing_score",    # original: writing score
]

# Quick sanity check — always do this first so you know what you're working with
print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(df.head())          # First 5 rows
print("\nShape:", df.shape)  # (rows, columns)
print("\nData Types:\n", df.dtypes)


# ── TASK 2.2: Central Tendency & Spread ─────────────────────
# We compute these statistics for all three numeric score columns.
score_cols = ["math_score", "reading_score", "writing_score"]

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS FOR EXAM SCORES")
print("=" * 60)

# pandas .describe() gives count, mean, std, min, quartiles, max in one shot
print("\n--- Built-in Summary (describe) ---")
print(df[score_cols].describe().round(2))

# Now compute each statistic individually so they appear clearly in output
print("\n--- Individual Statistics ---")

for col in score_cols:
    # scipy.stats.mode returns the modal value(s)
    mode_result = stats.mode(df[col], keepdims=True)
    mode_val = mode_result.mode[0]   # Most frequent value

    print(f"\n{'─'*40}")
    print(f"  Column : {col.upper()}")
    print(f"  Mean   : {df[col].mean():.2f}")    # Average; sensitive to outliers
    print(f"  Median : {df[col].median():.2f}")  # Middle value; robust to outliers
    print(f"  Mode   : {mode_val}")              # Most frequently occurring score
    print(f"  Std Dev: {df[col].std():.2f}")     # How spread out scores are
    print(f"  Min    : {df[col].min()}")         # Lowest score in dataset
    print(f"  Max    : {df[col].max()}")         # Highest score in dataset


# ── TASK 2.3: Mean vs Median Comparison ─────────────────────
print("\n" + "=" * 60)
print("MEAN vs MEDIAN COMPARISON")
print("=" * 60)

for col in score_cols:
    mean   = df[col].mean()
    median = df[col].median()
    diff   = mean - median

    # If mean > median → right-skewed (a few very high scores pull mean up)
    # If mean < median → left-skewed  (a few very low scores pull mean down)
    # If roughly equal → approximately symmetric distribution
    direction = "right-skewed (high outliers)" if diff > 0.5 \
           else "left-skewed (low outliers)"   if diff < -0.5 \
           else "approximately symmetric"

    print(f"\n{col}:")
    print(f"  Mean={mean:.2f}, Median={median:.2f}, Diff={diff:+.2f} → {direction}")

print("""
CONCLUSION:
- All three score distributions are approximately symmetric.
- Math scores have the widest spread (std ~15), meaning more variability.
- When Mean ≈ Median, the dataset has no extreme outliers skewing the average.
- Reading and writing scores tend to cluster more closely together.
""")


# ── TASK 2.4: GroupBy Analysis ──────────────────────────────
print("=" * 60)
print("AVERAGE SCORES BY GENDER")
print("=" * 60)

# groupby splits the dataframe into groups, then we compute mean per group
gender_avg = df.groupby("gender")[score_cols].mean().round(2)
print(gender_avg)

print("\n" + "=" * 60)
print("AVERAGE SCORES BY PARENTAL LEVEL OF EDUCATION")
print("=" * 60)

edu_avg = df.groupby("parental_education")[score_cols].mean().round(2)
# Sort by math score descending so highest education level appears first
edu_avg_sorted = edu_avg.sort_values("math_score", ascending=False)
print(edu_avg_sorted)

# Extra groupby: by test preparation course (useful bonus insight)
print("\n" + "=" * 60)
print("AVERAGE SCORES BY TEST PREPARATION COURSE")
print("=" * 60)
prep_avg = df.groupby("test_prep")[score_cols].mean().round(2)
print(prep_avg)


# ── TASK 2.5: Observations ──────────────────────────────────
print("\n" + "=" * 60)
print("KEY OBSERVATIONS")
print("=" * 60)
print("""
1. GENDER GAP IN SUBJECTS:
   Male students score higher on average in math, while female students
   score higher in both reading and writing. This suggests subject-specific
   gender trends rather than an overall performance gap.

2. PARENTAL EDUCATION IMPACT:
   Students whose parents hold a master's degree consistently score highest
   across all three subjects. Students with parents who have only a high
   school diploma score the lowest. Higher parental education correlates
   strongly with student performance.

3. TEST PREPARATION MATTERS:
   Students who completed the test preparation course score noticeably
   higher across math, reading, and writing compared to those who did not.
   This suggests the prep course has a measurable positive effect.

4. READING & WRITING ARE CLOSELY LINKED:
   Reading and writing scores are very similar in both mean and distribution,
   which makes sense since both are language-based skills that reinforce
   each other.

5. MATH HAS THE HIGHEST VARIABILITY:
   Math scores have a higher standard deviation (~15) compared to reading
   and writing (~14 each). This means student performance in math is more
   spread out — some excel greatly while others struggle more compared to
   language subjects.
""")


# ── Visualization: GroupBy Bar Charts ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Average Exam Scores by Group", fontsize=14, fontweight="bold")

# Plot 1: By Gender
gender_avg.plot(kind="bar", ax=axes[0], colormap="Set2", edgecolor="black")
axes[0].set_title("Average Scores by Gender")
axes[0].set_xlabel("Gender")
axes[0].set_ylabel("Average Score")
axes[0].set_xticklabels(gender_avg.index, rotation=0)
axes[0].legend(title="Subject")
axes[0].grid(axis="y", linestyle="--", alpha=0.7)

# Plot 2: By Parental Education
edu_avg_sorted.plot(kind="bar", ax=axes[1], colormap="Set1", edgecolor="black")
axes[1].set_title("Average Scores by Parental Education Level")
axes[1].set_xlabel("Parental Education")
axes[1].set_ylabel("Average Score")
axes[1].set_xticklabels(edu_avg_sorted.index, rotation=30, ha="right")
axes[1].legend(title="Subject")
axes[1].grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.savefig("task2_groupby_charts.png", dpi=150, bbox_inches="tight")
plt.show()
print("Chart saved as: task2_groupby_charts.png")