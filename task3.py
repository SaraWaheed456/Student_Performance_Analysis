
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── STEP 1: Create the Messy Dataset ─────────────────────────
# We first load the clean original data, then intentionally
# inject problems into it to simulate real-world messy data.

df_clean = pd.read_csv("data/StudentsPerformance.csv")

# Rename for convenience
df_clean.columns = [
    "gender", "race_ethnicity", "parental_education",
    "lunch", "test_prep", "math_score", "reading_score", "writing_score"
]

# Work on a copy so we don't modify the original
df_messy = df_clean.copy()

np.random.seed(42)  # Ensures reproducibility — same random results every run

# Problem 1: Inject missing values into numeric columns (~5% of rows)
for col in ["math_score", "reading_score", "writing_score"]:
    missing_idx = np.random.choice(df_messy.index, size=25, replace=False)
    df_messy.loc[missing_idx, col] = np.nan  # Replace with NaN

# Problem 2: Inject missing values into categorical columns
cat_missing_idx = np.random.choice(df_messy.index, size=15, replace=False)
df_messy.loc[cat_missing_idx, "gender"] = np.nan

# Problem 3: Add duplicate rows (copy 20 random rows and append them)
dup_rows = df_messy.sample(20, random_state=42)
df_messy = pd.concat([df_messy, dup_rows], ignore_index=True)

# Problem 4: Inject outliers — scores below 0 or above 100 are impossible
outlier_idx = np.random.choice(df_messy.index, size=10, replace=False)
df_messy.loc[outlier_idx, "math_score"] = np.random.choice(
    [-5, -10, 105, 110, 115], size=10
)

# Problem 5: Add inconsistent string casing in categorical columns
gender_typo_idx = np.random.choice(df_messy.index, size=30, replace=False)
df_messy.loc[gender_typo_idx, "gender"] = df_messy.loc[
    gender_typo_idx, "gender"
].str.upper()  # "MALE", "FEMALE" instead of "male", "female"

# Save the messy dataset
df_messy.to_csv("data/messy_students.csv", index=False)
print("✓ messy_students.csv created with the following injected problems:")
print(f"  - Missing values in score columns (~25 each)")
print(f"  - Missing values in 'gender' column (~15)")
print(f"  - 20 duplicate rows appended")
print(f"  - 10 outliers in math_score (values outside 0–100)")
print(f"  - Inconsistent casing in 'gender' column")


# ── STEP 2: Load the Messy Dataset & Show All Problems ───────
print("\n" + "=" * 60)
print("LOADING & DIAGNOSING THE MESSY DATASET")
print("=" * 60)

df = pd.read_csv("data/messy_students.csv")

# 2a. Basic info — dtypes and non-null counts per column
print("\n--- df.info() ---")
df.info()

# 2b. Missing value count per column
print("\n--- Missing Values (per column) ---")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_report = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
print(missing_report[missing_report["Missing Count"] > 0])  # Only show affected cols

# 2c. Duplicate rows
n_dupes = df.duplicated().sum()
print(f"\n--- Duplicate Rows ---")
print(f"  Total duplicates found: {n_dupes}")

# 2d. Outliers: scores must be in [0, 100]
print("\n--- Outliers in Score Columns (valid range: 0–100) ---")
score_cols = ["math_score", "reading_score", "writing_score"]
for col in score_cols:
    outliers = df[(df[col] < 0) | (df[col] > 100)]
    print(f"  {col}: {len(outliers)} outlier(s)")
    if len(outliers) > 0:
        print(f"    Values: {outliers[col].tolist()}")

# 2e. Value counts for categorical columns — spot inconsistencies
print("\n--- Gender Column Unique Values (inconsistent casing?) ---")
print(df["gender"].value_counts())


# ── STEP 3: Clean the Data Step by Step ─────────────────────
print("\n" + "=" * 60)
print("CLEANING THE DATA")
print("=" * 60)

# Store original shape for "before vs after" comparison
original_shape = df.shape

# --- Clean Step 1: Fix inconsistent string casing ---
# Lowercase everything, then strip whitespace, so "MALE" → "male"
df["gender"] = df["gender"].str.lower().str.strip()
df["test_prep"] = df["test_prep"].str.lower().str.strip()
df["lunch"] = df["lunch"].str.lower().str.strip()
df["parental_education"] = df["parental_education"].str.lower().str.strip()
df["race_ethnicity"] = df["race_ethnicity"].str.lower().str.strip()
print("✓ Step 1: Standardized string casing across all categorical columns")

# --- Clean Step 2: Handle outliers BEFORE filling missing values ---
# Rationale: replace impossible scores with NaN first, so they can be
# treated as missing values in the next step rather than corrupting the mean.
for col in score_cols:
    bad_mask = (df[col] < 0) | (df[col] > 100)  # Boolean mask of invalid scores
    n_bad = bad_mask.sum()
    df.loc[bad_mask, col] = np.nan  # Replace outlier with NaN
    print(f"✓ Step 2: Replaced {n_bad} outlier(s) in '{col}' with NaN")

# --- Clean Step 3: Fill missing numeric values with column median ---
# Why median, not mean?
# → Median is robust to outliers; even after outlier removal, it's safer.
for col in score_cols:
    n_missing = df[col].isnull().sum()
    median_val = df[col].median()  # Compute median of non-null values
    df[col] = df[col].fillna(median_val)
    print(f"✓ Step 3: Filled {n_missing} missing value(s) in '{col}' with median ({median_val:.1f})")

# --- Clean Step 4: Fill missing categorical values with mode ---
# Mode = the most frequently occurring category in that column
n_missing_gender = df["gender"].isnull().sum()
mode_gender = df["gender"].mode()[0]  # mode() returns a Series; [0] = top mode
df["gender"] = df["gender"].fillna(mode_gender)
print(f"✓ Step 4: Filled {n_missing_gender} missing 'gender' value(s) with mode ('{mode_gender}')")

# --- Clean Step 5: Remove duplicate rows ---
n_before_dedup = len(df)
df = df.drop_duplicates()  # Keeps first occurrence of each duplicate
n_removed = n_before_dedup - len(df)
print(f"✓ Step 5: Removed {n_removed} duplicate rows ({n_before_dedup} → {len(df)} rows)")

# --- Clean Step 6: Reset index after row removal ---
df = df.reset_index(drop=True)  # drop=True prevents old index from becoming a column
print("✓ Step 6: Reset DataFrame index")


# ── STEP 4: Save & Show Before vs After Summary ─────────────
df.to_csv("data/cleaned_students.csv", index=False)
print("\n✓ Cleaned dataset saved as: data/cleaned_students.csv")

print("\n" + "=" * 60)
print("BEFORE vs AFTER SUMMARY")
print("=" * 60)

df_verify = pd.read_csv("data/cleaned_students.csv")  # Load from disk to confirm save

print(f"\n  Rows  : {original_shape[0]:>6}  →  {len(df_verify):>6}")
print(f"  Cols  : {original_shape[1]:>6}  →  {len(df_verify.columns):>6}")
print(f"\n  Missing values after cleaning:")
print(df_verify.isnull().sum())
print(f"\n  Duplicates after cleaning: {df_verify.duplicated().sum()}")
print(f"\n  Outliers in math_score after cleaning:")
outlier_check = df_verify[(df_verify["math_score"] < 0) | (df_verify["math_score"] > 100)]
print(f"    {len(outlier_check)} outlier(s) found (should be 0)")

print("\n--- Final Descriptive Summary of Cleaned Data ---")
print(df_verify[score_cols].describe().round(2))


# ── Visualization: Missing Values Heatmap ────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Data Cleaning: Before vs After", fontsize=13, fontweight="bold")

# Left: Missing value counts in messy data
messy_reload = pd.read_csv("data/messy_students.csv")
missing_before = messy_reload[score_cols + ["gender"]].isnull().sum()
axes[0].bar(missing_before.index, missing_before.values, color="salmon", edgecolor="black")
axes[0].set_title("Missing Values — BEFORE Cleaning")
axes[0].set_ylabel("Count of Missing Values")
axes[0].set_xticklabels(missing_before.index, rotation=15)
axes[0].grid(axis="y", linestyle="--", alpha=0.6)

# Right: Missing value counts in cleaned data
missing_after = df_verify[score_cols + ["gender"]].isnull().sum()
axes[1].bar(missing_after.index, missing_after.values, color="lightgreen", edgecolor="black")
axes[1].set_title("Missing Values — AFTER Cleaning")
axes[1].set_ylabel("Count of Missing Values")
axes[1].set_xticklabels(missing_after.index, rotation=15)
axes[1].grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("task3_cleaning_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Chart saved as: task3_cleaning_comparison.png")