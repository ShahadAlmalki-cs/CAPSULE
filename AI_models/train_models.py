"""
CAPSULE — Comprehensive Bilingual Model Training Pipeline
Trains ML classifiers + BiLSTM on DataSet/capsule_dataset.csv (Arabic + English).
Produces dataset analysis, visualizations, training results, and an HTML report.

Run:
    python AI_models/train_models.py
"""

import os, re, json, warnings, base64, io, time, sys
from datetime import datetime
from collections import Counter, OrderedDict

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, matthews_corrcoef,
    classification_report, confusion_matrix,
    roc_auc_score, log_loss,
)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset as TorchDataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODEL_DIR   = os.path.join(BASE_DIR, "trained_models")
REPORT_PATH = os.path.join(MODEL_DIR, "training_report.html")

for _cand in [
    os.path.join(PROJECT_DIR, "DataSet", "capsule_dataset.csv"),
    os.path.join(PROJECT_DIR, "DataSet", "capsule_dataset.csv.csv"),
    os.path.join(PROJECT_DIR, "capsule_dataset.csv"),
]:
    if os.path.isfile(_cand):
        DATASET_PATH = _cand
        break
else:
    raise FileNotFoundError("capsule_dataset.csv not found. Expected in DataSet/ folder.")

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Plotting defaults
# ---------------------------------------------------------------------------
PALETTE = sns.color_palette("Set2", 16)
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.unicode_minus": False, "font.size": 10,
})

SEP  = "=" * 76
SSEP = "-" * 76

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64

def safe(val):
    return "" if pd.isna(val) else str(val).strip()

def pct(n, total):
    return f"{n/total*100:.1f}%" if total > 0 else "0.0%"

def elapsed(t0):
    return f"{time.time()-t0:.2f}s"

start_time = time.time()
charts     = {}

# ===========================================================================
#  SECTION 1 — LOAD DATASET
# ===========================================================================
print(f"\n{SEP}")
print("  CAPSULE AI — Bilingual Model Training Pipeline")
print(f"  Date   : {datetime.now():%Y-%m-%d %H:%M:%S}")
print(f"  Python : {sys.version.split()[0]}")
print(f"  PyTorch: {'Yes (' + torch.__version__ + ')' if HAS_TORCH else 'Not installed — BiLSTM skipped'}")
print(SEP)

print(f"\n{SEP}")
print("  SECTION 1 — Loading Dataset")
print(SEP)

df = pd.read_csv(DATASET_PATH)
EN_COLS = [c for c in df.columns if c.endswith("_en")]
AR_COLS = [c for c in df.columns if c.endswith("_ar")]

print(f"  Path    : {DATASET_PATH}")
print(f"  Records : {len(df):,}")
print(f"  Columns : {len(df.columns)}  ({len(EN_COLS)} English, {len(AR_COLS)} Arabic)")
print(f"  EN cols : {', '.join(EN_COLS)}")
print(f"  AR cols : {', '.join(AR_COLS)}")

# ===========================================================================
#  SECTION 2 — DATASET STATISTICAL ANALYSIS
# ===========================================================================
print(f"\n{SEP}")
print("  SECTION 2 — Dataset Statistical Analysis")
print(SEP)

# 2.1 Basic stats
print("\n  2.1  Basic Statistics")
print(f"       {'Metric':<40} {'Value':>8}")
print("       " + SSEP[:50])
stats = [
    ("Total records",                       len(df)),
    ("Total columns",                       len(df.columns)),
    ("English-specific columns",            len(EN_COLS)),
    ("Arabic-specific columns",             len(AR_COLS)),
    ("Unique drug names (EN)",              df["drug_name_en"].nunique()),
    ("Unique drug names (AR)",              df["drug_name_ar"].nunique()),
    ("Unique generic names (EN)",           df["generic_name_en"].nunique()),
    ("Unique drug classes (EN)",            df["drug_class_en"].nunique()),
    ("Unique dosage forms (EN)",            df["dosage_form_en"].nunique()),
    ("Unique manufacturers (EN)",           df["manufacturer_en"].nunique()),
    ("Unique colors (EN)",                  df["color_en"].nunique()),
    ("Unique shapes (EN)",                  df["shape_en"].nunique()),
    ("Unique strengths",                    df["strength"].nunique() if "strength" in df.columns else 0),
    ("Unique imprint codes",                df["imprint_code"].nunique() if "imprint_code" in df.columns else 0),
]
for name, val in stats:
    print(f"       {name:<40} {val:>8,}")

# 2.2 RX/OTC
print("\n  2.2  Prescription vs OTC Status")
rx_otc = df["rx_otc_status_en"].value_counts()
print(f"       {'Status (EN)':<25} {'Status (AR)':<25} {'Count':>6} {'%':>8}")
print("       " + SSEP[:67])
for s_en, cnt in rx_otc.items():
    s_ar = df[df["rx_otc_status_en"] == s_en]["rx_otc_status_ar"].iloc[0]
    print(f"       {s_en:<25} {safe(s_ar):<25} {cnt:>6} {pct(cnt, len(df)):>8}")

# 2.3 Missing data
print("\n  2.3  Missing Data Analysis")
total_missing = df.isna().sum().sum()
if total_missing == 0:
    print("       No missing values detected.")
else:
    print(f"       {'Column':<32} {'Missing':>8} {'%':>8}")
    print("       " + SSEP[:51])
    for col in df.columns:
        n = df[col].isna().sum()
        if n > 0:
            print(f"       {col:<32} {n:>8} {pct(n, len(df)):>8}")
    print(f"       {'TOTAL':<32} {total_missing:>8}")

# 2.4 Duplicate detection
print("\n  2.4  Duplicate Detection")
n_dup_all  = df.duplicated().sum()
n_dup_name = df["drug_name_en"].duplicated().sum()
n_dup_id   = df["drug_id"].duplicated().sum() if "drug_id" in df.columns else 0
print(f"       Full duplicate rows     : {n_dup_all}")
print(f"       Duplicate drug names EN : {n_dup_name}")
print(f"       Duplicate drug IDs      : {n_dup_id}")

# 2.5 Drug class distribution
print("\n  2.5  Drug Class Distribution (English | Arabic)")
class_counts = df["drug_class_en"].value_counts()
print(f"       {'Drug Class (EN)':<45} {'(AR)':<32} {'Count':>5} {'%':>7}")
print("       " + SSEP[:92])
for cls_en, cnt in class_counts.items():
    ar_val = df[df["drug_class_en"] == cls_en]["drug_class_ar"].iloc[0]
    print(f"       {cls_en:<45} {safe(ar_val):<32} {cnt:>5} {pct(cnt, len(df)):>7}")
print(f"       {'TOTAL':<83} {len(df):>5}")

# 2.6 Dosage form
print("\n  2.6  Dosage Form Distribution")
form_counts = df["dosage_form_en"].value_counts()
print(f"       {'Form (EN)':<30} {'(AR)':<25} {'Count':>6} {'%':>7}")
print("       " + SSEP[:71])
for f_en, cnt in form_counts.items():
    ar_val = df[df["dosage_form_en"] == f_en]["dosage_form_ar"].iloc[0]
    print(f"       {f_en:<30} {safe(ar_val):<25} {cnt:>6} {pct(cnt, len(df)):>7}")

# 2.7 Color
print("\n  2.7  Color Distribution")
color_counts = df["color_en"].value_counts()
print(f"       {'Color (EN)':<22} {'(AR)':<18} {'Count':>6} {'%':>7}")
print("       " + SSEP[:56])
for c_en, cnt in color_counts.items():
    ar_val = df[df["color_en"] == c_en]["color_ar"].iloc[0]
    print(f"       {c_en:<22} {safe(ar_val):<18} {cnt:>6} {pct(cnt, len(df)):>7}")

# 2.8 Shape
print("\n  2.8  Shape Distribution")
shape_counts = df["shape_en"].value_counts()
print(f"       {'Shape (EN)':<22} {'(AR)':<18} {'Count':>6} {'%':>7}")
print("       " + SSEP[:56])
for s_en, cnt in shape_counts.items():
    ar_val = df[df["shape_en"] == s_en]["shape_ar"].iloc[0]
    print(f"       {s_en:<22} {safe(ar_val):<18} {cnt:>6} {pct(cnt, len(df)):>7}")

# 2.9 Manufacturer top 15
print("\n  2.9  Top 15 Manufacturers")
mfr_counts = df["manufacturer_en"].value_counts().head(15)
print(f"       {'Manufacturer (EN)':<32} {'(AR)':<25} {'Count':>6}")
print("       " + SSEP[:66])
for mfr_en, cnt in mfr_counts.items():
    ar_val = df[df["manufacturer_en"] == mfr_en]["manufacturer_ar"].iloc[0]
    print(f"       {mfr_en:<32} {safe(ar_val):<25} {cnt:>6}")

# 2.10 Strength
if "strength" in df.columns:
    print("\n  2.10 Top 15 Strength Values")
    str_counts = df["strength"].value_counts().head(15)
    print(f"       {'Strength':<28} {'Count':>6}")
    print("       " + SSEP[:37])
    for s, cnt in str_counts.items():
        print(f"       {safe(s):<28} {cnt:>6}")

# 2.11 Arabic column coverage
print("\n  2.11 Arabic Column Coverage")
print(f"       {'Column':<32} {'Filled':>6} {'Empty':>6} {'Coverage':>10}")
print("       " + SSEP[:58])
ar_coverage = {}
for col in AR_COLS:
    filled = df[col].notna().sum()
    empty  = len(df) - filled
    ar_coverage[col] = filled / len(df) * 100
    print(f"       {col:<32} {filled:>6} {empty:>6} {pct(filled, len(df)):>10}")

# 2.12 Source dataset
if "source_dataset" in df.columns:
    print("\n  2.12 Source Dataset Distribution")
    src_vc = df["source_dataset"].value_counts()
    for src, cnt in src_vc.items():
        print(f"       {safe(src):<45} {cnt:>6} {pct(cnt, len(df)):>8}")

# 2.13 Class imbalance
print("\n  2.13 Class Imbalance Analysis")
imbalance_ratio = class_counts.max() / class_counts.min()
gini = 1 - sum((c / len(df)) ** 2 for c in class_counts.values)
print(f"       Max class  : {class_counts.max()} ({class_counts.idxmax()})")
print(f"       Min class  : {class_counts.min()} ({class_counts.idxmin()})")
print(f"       Ratio      : {imbalance_ratio:.2f}:1")
print(f"       Gini index : {gini:.4f}  (0=perfectly balanced, 1=totally imbalanced)")

# 2.14 Data types
print("\n  2.14 Data Types Summary")
print(f"       Object columns   : {(df.dtypes == 'object').sum()}")
print(f"       Numeric columns  : {df.select_dtypes(include=[np.number]).shape[1]}")
print(f"       Memory usage     : {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

# ===========================================================================
#  SECTION 3 — DATASET VISUALIZATIONS
# ===========================================================================
print(f"\n{SEP}")
print("  SECTION 3 — Dataset Visualizations")
print(SEP)

# 3.1 Drug class distribution
fig, ax = plt.subplots(figsize=(12, max(5, len(class_counts) * 0.42)))
bars = ax.barh(class_counts.index[::-1], class_counts.values[::-1],
               color=PALETTE[0], edgecolor="white", linewidth=0.5)
ax.set_xlabel("Number of Records")
ax.set_title("Drug Class Distribution (English)", fontsize=13, fontweight="bold")
for bar in bars:
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            str(int(bar.get_width())), va="center", fontsize=8)
fig.tight_layout()
charts["drug_class_dist"] = fig_to_b64(fig)
print("  [OK] Drug class distribution")

# 3.2 Dosage form
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(form_counts.index, form_counts.values, color=PALETTE[1], edgecolor="white")
ax.set_ylabel("Count")
ax.set_title("Dosage Form Distribution", fontsize=13, fontweight="bold")
plt.xticks(rotation=35, ha="right", fontsize=9)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            str(int(bar.get_height())), ha="center", fontsize=8)
fig.tight_layout()
charts["dosage_form_dist"] = fig_to_b64(fig)
print("  [OK] Dosage form distribution")

# 3.3 Manufacturer top 10
mfr10 = df["manufacturer_en"].value_counts().head(10)
fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.barh(mfr10.index[::-1], mfr10.values[::-1], color=PALETTE[2], edgecolor="white")
ax.set_xlabel("Number of Drugs")
ax.set_title("Top 10 Manufacturers", fontsize=13, fontweight="bold")
for bar in bars:
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
            str(int(bar.get_width())), va="center", fontsize=8)
fig.tight_layout()
charts["manufacturer_dist"] = fig_to_b64(fig)
print("  [OK] Manufacturer distribution")

# 3.4 Color distribution
COLOR_HEX = {
    "white": "#e8e8e8", "red": "#e74c3c", "blue": "#3498db", "green": "#27ae60",
    "yellow": "#f1c40f", "pink": "#e91e8a", "orange": "#e67e22", "brown": "#8b4513",
    "black": "#2c3e50", "beige": "#f5deb3", "purple": "#9b59b6", "tan": "#d2b48c",
    "peach": "#ffdab9", "light grey": "#c0c0c0", "dark grey": "#808080",
    "dark green": "#1a5e1a", "cream": "#fffdd0", "light blue": "#add8e6",
}
fig, ax = plt.subplots(figsize=(11, 5))
c_colors = [COLOR_HEX.get(c.lower(), PALETTE[3]) for c in color_counts.index]
bars = ax.bar(color_counts.index, color_counts.values, color=c_colors, edgecolor="grey", linewidth=0.5)
ax.set_ylabel("Count")
ax.set_title("Color Distribution", fontsize=13, fontweight="bold")
plt.xticks(rotation=40, ha="right", fontsize=9)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            str(int(bar.get_height())), ha="center", fontsize=8)
fig.tight_layout()
charts["color_dist"] = fig_to_b64(fig)
print("  [OK] Color distribution")

# 3.5 Shape distribution
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(shape_counts.index, shape_counts.values, color=PALETTE[4], edgecolor="white")
ax.set_ylabel("Count")
ax.set_title("Shape Distribution", fontsize=13, fontweight="bold")
plt.xticks(rotation=35, ha="right", fontsize=9)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            str(int(bar.get_height())), ha="center", fontsize=8)
fig.tight_layout()
charts["shape_dist"] = fig_to_b64(fig)
print("  [OK] Shape distribution")

# 3.6 RX/OTC pie
fig, ax = plt.subplots(figsize=(6, 5))
wedges, texts, autotexts = ax.pie(
    rx_otc.values, labels=rx_otc.index, autopct="%1.1f%%",
    colors=[PALETTE[5], PALETTE[6]], startangle=90,
    textprops={"fontsize": 11})
for at in autotexts:
    at.set_fontweight("bold")
ax.set_title("Prescription vs OTC Status", fontsize=13, fontweight="bold")
fig.tight_layout()
charts["rx_otc_pie"] = fig_to_b64(fig)
print("  [OK] RX/OTC pie chart")

# 3.7 Strength top 15
if "strength" in df.columns:
    str_top = df["strength"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(str_top.index[::-1].astype(str), str_top.values[::-1],
                   color=PALETTE[7], edgecolor="white")
    ax.set_xlabel("Count")
    ax.set_title("Top 15 Strength Values", fontsize=13, fontweight="bold")
    for bar in bars:
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                str(int(bar.get_width())), va="center", fontsize=8)
    fig.tight_layout()
    charts["strength_dist"] = fig_to_b64(fig)
    print("  [OK] Strength distribution")

# 3.8 Arabic column coverage
ar_col_labels = [c.replace("_ar", "").replace("_", " ").title() for c in AR_COLS]
ar_col_vals   = [df[c].notna().sum() / len(df) * 100 for c in AR_COLS]
fig, ax = plt.subplots(figsize=(10, 5))
bar_colors = [PALETTE[8] if v >= 90 else (PALETTE[9] if v >= 70 else PALETTE[10]) for v in ar_col_vals]
bars = ax.barh(ar_col_labels[::-1], ar_col_vals[::-1], color=bar_colors[::-1], edgecolor="white")
ax.set_xlabel("Coverage %")
ax.set_xlim(0, 115)
ax.axvline(100, color="#ccc", linestyle="--", linewidth=0.8)
ax.set_title("Arabic Column Coverage (%)", fontsize=13, fontweight="bold")
for bar in bars:
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.0f}%", va="center", fontsize=8)
fig.tight_layout()
charts["arabic_coverage"] = fig_to_b64(fig)
print("  [OK] Arabic column coverage")

# 3.9 Class imbalance bar (class size distribution)
fig, ax = plt.subplots(figsize=(12, max(5, len(class_counts) * 0.42)))
colors_imb = [PALETTE[0] if v >= 10 else (PALETTE[1] if v >= 5 else PALETTE[3])
              for v in class_counts.values[::-1]]
bars = ax.barh(class_counts.index[::-1], class_counts.values[::-1],
               color=colors_imb, edgecolor="white")
ax.set_xlabel("Number of Samples")
ax.set_title("Class Size Distribution (Imbalance View)", fontsize=13, fontweight="bold")
ax.axvline(class_counts.mean(), color="red", linestyle="--", linewidth=1.5,
           label=f"Mean={class_counts.mean():.1f}")
ax.legend(fontsize=9)
for bar in bars:
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
            str(int(bar.get_width())), va="center", fontsize=8)
fig.tight_layout()
charts["class_imbalance"] = fig_to_b64(fig)
print("  [OK] Class imbalance chart")

# ===========================================================================
#  SECTION 4 — FEATURE ENGINEERING (BILINGUAL)
# ===========================================================================
print(f"\n{SEP}")
print("  SECTION 4 — Feature Engineering (Bilingual)")
print(SEP)

def build_text_en(row):
    parts = [
        safe(row.get("drug_name_en")),
        f"generic {safe(row.get('generic_name_en'))}",
        f"form {safe(row.get('dosage_form_en'))}",
        f"class {safe(row.get('drug_class_en'))}",
        f"indication {safe(row.get('indications_en'))}",
        f"color {safe(row.get('color_en'))}",
        f"shape {safe(row.get('shape_en'))}",
        safe(row.get("salt_composition")),
        safe(row.get("side_effects_en")),
        safe(row.get("contraindications_en")),
    ]
    return " ".join(p for p in parts if p).lower()

def build_text_ar(row):
    parts = [
        safe(row.get("drug_name_ar")),
        safe(row.get("generic_name_ar")),
        safe(row.get("dosage_form_ar")),
        safe(row.get("drug_class_ar")),
        safe(row.get("indications_ar")),
        safe(row.get("color_ar")),
        safe(row.get("shape_ar")),
        safe(row.get("side_effects_ar")),
        safe(row.get("contraindications_ar")),
    ]
    return " ".join(p for p in parts if p)

df["text_en"]       = df.apply(build_text_en, axis=1)
df["text_ar"]       = df.apply(build_text_ar, axis=1)
df["text_combined"] = df["text_en"] + " " + df["text_ar"]

en_lens = df["text_en"].str.len()
ar_lens = df["text_ar"].str.len()
cb_lens = df["text_combined"].str.len()

print(f"\n  {'Feature':<12} {'Records':>8} {'Mean':>8} {'Min':>6} {'Max':>6} {'Median':>8} {'Std':>8}")
print("  " + SSEP[:58])
for name, lens in [("English", en_lens), ("Arabic", ar_lens), ("Combined", cb_lens)]:
    print(f"  {name:<12} {len(df):>8,} {lens.mean():>8.0f} {lens.min():>6} "
          f"{lens.max():>6} {lens.median():>8.0f} {lens.std():>8.0f}")

# Text length distribution chart
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, (name, lens, c) in zip(axes, [
    ("English",  en_lens, PALETTE[0]),
    ("Arabic",   ar_lens, PALETTE[1]),
    ("Combined", cb_lens, PALETTE[2]),
]):
    ax.hist(lens, bins=20, color=c, edgecolor="white", alpha=0.85)
    ax.axvline(lens.mean(),   color="red",    linestyle="--", linewidth=1.5, label=f"Mean={lens.mean():.0f}")
    ax.axvline(lens.median(), color="orange", linestyle=":",  linewidth=1.5, label=f"Median={lens.median():.0f}")
    ax.set_xlabel("Character Length")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{name} Text Length")
    ax.legend(fontsize=8)
fig.suptitle("Text Feature Length Distributions (EN / AR / Combined)",
             fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
charts["text_length_dist"] = fig_to_b64(fig)
print("  [OK] Text length distribution chart")

# Word frequency analysis
print("\n  Top 20 English Terms (excluding feature keywords):")
_stop = {"form", "class", "color", "shape", "generic", "indication", "the", "and", "for"}
en_wf = Counter(w for t in df["text_en"] for w in t.split() if len(w) > 3 and w not in _stop)
for w, c in en_wf.most_common(20):
    bar_str = "#" * min(c // 2, 30)
    print(f"    {w:<28} {c:>5}  {bar_str}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
en_top = en_wf.most_common(15)
axes[0].barh([w for w, _ in reversed(en_top)], [c for _, c in reversed(en_top)],
             color=PALETTE[0], edgecolor="white")
axes[0].set_xlabel("Frequency")
axes[0].set_title("Top 15 English Terms", fontsize=11, fontweight="bold")

ar_wf   = Counter(w for t in df["text_ar"] for w in t.split() if len(w) > 2)
ar_top  = ar_wf.most_common(15)
axes[1].barh(range(len(ar_top)), [c for _, c in reversed(ar_top)],
             color=PALETTE[1], edgecolor="white")
axes[1].set_yticks(range(len(ar_top)))
axes[1].set_yticklabels([w for w, _ in reversed(ar_top)], fontsize=9)
axes[1].set_xlabel("Frequency")
axes[1].set_title("Top 15 Arabic Terms", fontsize=11, fontweight="bold")
fig.suptitle("Top Word Frequencies (English & Arabic)", fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
charts["word_freq"] = fig_to_b64(fig)
print("  [OK] Word frequency chart")

# ===========================================================================
#  SECTION 5 — CLASSIFICATION TARGET PREPARATION
# ===========================================================================
print(f"\n{SEP}")
print("  SECTION 5 — Preparing Classification Target")
print(SEP)

LABEL_COL   = "drug_class_en"
MIN_SAMPLES = 2

label_vc    = df[LABEL_COL].value_counts()
valid_lbls  = label_vc[label_vc >= MIN_SAMPLES].index.tolist()
dropped     = label_vc[label_vc < MIN_SAMPLES]

ml_df = df[df[LABEL_COL].isin(valid_lbls)].copy()
ml_df = ml_df[ml_df["text_en"].str.strip() != ""].copy()

le          = LabelEncoder()
y           = le.fit_transform(ml_df[LABEL_COL])
class_names = list(le.classes_)
n_classes   = len(class_names)

min_test  = max(n_classes, int(len(ml_df) * 0.15))
TEST_SIZE = min(0.40, max(0.20, min_test / len(ml_df)))

print(f"  Target column      : {LABEL_COL}")
print(f"  Min samples/class  : {MIN_SAMPLES}")
print(f"  Valid classes      : {len(valid_lbls)}")
print(f"  Dropped classes    : {len(dropped)}")
if len(dropped):
    for lbl, cnt in dropped.items():
        print(f"    - {lbl} ({cnt} sample{'s' if cnt != 1 else ''})")
print(f"  Training samples   : {len(ml_df):,}")
print(f"  Test split         : {TEST_SIZE*100:.0f}%")
print(f"\n  {'#':>4} {'Class Name':<50} {'Samples':>8}")
print("  " + SSEP[:65])
for i, cn in enumerate(class_names):
    cnt = int((y == i).sum())
    bar = "|" * min(cnt, 40)
    print(f"  {i+1:>4}. {cn:<50} {cnt:>8}  {bar}")

joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))
print(f"\n  [SAVED] label_encoder.pkl")

# ===========================================================================
#  SECTION 6 — TF-IDF VECTORIZERS
# ===========================================================================
print(f"\n{SEP}")
print("  SECTION 6 — Training TF-IDF Vectorizers")
print(SEP)

TFIDF_CFGS = {"English": "text_en", "Arabic": "text_ar", "Combined": "text_combined"}
tfidf_models   = {}
tfidf_matrices = {}

print(f"  {'Config':<12} {'Vocab':>8} {'Matrix Shape':>16} {'Time':>8}")
print("  " + SSEP[:48])
for name, col in TFIDF_CFGS.items():
    t0    = time.time()
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=10_000, min_df=1, sublinear_tf=True)
    tfidf.fit(df[col].fillna("").tolist())
    X_ml  = tfidf.transform(ml_df[col].fillna(""))
    tfidf_models[name]   = tfidf
    tfidf_matrices[name] = X_ml
    print(f"  {name:<12} {len(tfidf.vocabulary_):>8,} {str(X_ml.shape):>16} {elapsed(t0):>8}")

joblib.dump(tfidf_models["Combined"], os.path.join(MODEL_DIR, "tfidf_retrieval.pkl"))
print(f"\n  [SAVED] tfidf_retrieval.pkl  (Combined — primary for inference)")

# ===========================================================================
#  SECTION 7 — ML CLASSIFIER TRAINING
# ===========================================================================
print(f"\n{SEP}")
print("  SECTION 7 — Training ML Classifiers  (3 Languages x 6 Models = 18 Total)")
print(SEP)

min_class_size = min(Counter(y).values())
N_CV = max(2, min(5, min_class_size))
_SVM_CV = max(2, min(N_CV, min_class_size))

def make_classifiers():
    return OrderedDict([
        ("Naive Bayes",         MultinomialNB(alpha=0.1)),
        ("Logistic Regression", LogisticRegression(max_iter=2000, C=1.0, random_state=42)),
        ("Linear SVM",          CalibratedClassifierCV(LinearSVC(max_iter=3000, random_state=42),
                                                        cv=_SVM_CV)),
        ("Random Forest",       RandomForestClassifier(n_estimators=200, max_depth=20,
                                                       random_state=42, n_jobs=-1)),
        ("Gradient Boosting",   GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                                            random_state=42)),
        ("KNN",                 KNeighborsClassifier(n_neighbors=min(5, min_class_size),
                                                     n_jobs=-1)),
    ])

all_results    = {}
confusion_data = {}
cv_results     = {}
best_acc       = 0.0
best_key       = None
best_model_obj = None

langs       = ["English", "Arabic", "Combined"]
models_list = list(make_classifiers().keys())

for lang in langs:
    print(f"\n  ── {lang} {'─'*55}")
    X = tfidf_matrices[lang]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y)
    print(f"  Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}  |  Features: {X_train.shape[1]:,}")
    print(f"\n  {'Model':<22} {'Acc':>7} {'F1':>7} {'Prec':>7} {'Rec':>7} "
          f"{'Bal Acc':>8} {'MCC':>7} {'AUC-ROC':>8} {'LogLoss':>8} {'CV':>8} {'Time':>7}")
    print("  " + SSEP[:100])

    for model_name, clf in make_classifiers().items():
        t0 = time.time()
        try:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
        except Exception as e:
            print(f"  {model_name:<22} [SKIP] {e}")
            continue
        elapsed_ = time.time() - t0

        acc     = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        f1      = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        prec    = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec     = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        mcc     = matthews_corrcoef(y_test, y_pred)

        # AUC-ROC and Log Loss (require probability estimates)
        try:
            proba = clf.predict_proba(X_test)
            # Expand to full n_classes columns if classifier only saw a subset
            if hasattr(clf, "classes_") and len(clf.classes_) < n_classes:
                full_p = np.zeros((len(proba), n_classes))
                for _ci, _cc in enumerate(clf.classes_):
                    full_p[:, int(_cc)] = proba[:, _ci]
                proba = full_p
            roc_auc   = roc_auc_score(y_test, proba, multi_class="ovr",
                                       average="weighted", labels=list(range(n_classes)))
            clf_logloss = log_loss(y_test, proba, labels=list(range(n_classes)))
        except Exception:
            roc_auc     = np.nan
            clf_logloss = np.nan

        all_labels   = sorted(set(y_test) | set(y_pred))
        report_names = [class_names[i] for i in all_labels]

        try:
            cv_sc = cross_val_score(
                clf, X, y,
                cv=StratifiedKFold(n_splits=N_CV, shuffle=True, random_state=42),
                scoring="accuracy")
            cv_mean, cv_std = cv_sc.mean(), cv_sc.std()
        except Exception:
            cv_sc = np.array([acc])
            cv_mean, cv_std = acc, 0.0

        key = (lang, model_name)
        all_results[key] = {
            "accuracy":          acc,
            "balanced_accuracy": bal_acc,
            "f1":                f1,
            "precision":         prec,
            "recall":            rec,
            "mcc":               mcc,
            "roc_auc":           roc_auc,
            "log_loss":          clf_logloss,
            "time":              elapsed_,
            "report": classification_report(y_test, y_pred, labels=all_labels,
                                            target_names=report_names, zero_division=0),
        }
        confusion_data[key] = (confusion_matrix(y_test, y_pred, labels=all_labels), report_names)
        cv_results[key]     = {"mean": cv_mean, "std": cv_std, "scores": cv_sc.tolist()}

        star = ""
        if acc > best_acc:
            best_acc, best_key, best_model_obj = acc, key, clf
            star = " <-- BEST"

        auc_str = f"{roc_auc:.3f}" if not np.isnan(roc_auc) else "  N/A "
        ll_str  = f"{clf_logloss:.3f}" if not np.isnan(clf_logloss) else "  N/A "
        print(f"  {model_name:<22} {acc*100:>6.1f}% {f1*100:>6.1f}% {prec*100:>6.1f}% "
              f"{rec*100:>6.1f}% {bal_acc*100:>7.1f}% {mcc:>7.3f} "
              f"{auc_str:>8} {ll_str:>8} {cv_mean*100:>6.1f}%  {elapsed_:>6.2f}s{star}")

print(f"\n  Overall Best: {best_key[1]} ({best_key[0]}) -> Accuracy = {best_acc*100:.1f}%")

joblib.dump(best_model_obj, os.path.join(MODEL_DIR, "best_ml_model.pkl"))
print(f"  [SAVED] best_ml_model.pkl")

# ===========================================================================
#  SECTION 8 — TRAINING RESULT VISUALIZATIONS
# ===========================================================================
print(f"\n{SEP}")
print("  SECTION 8 — Training Result Visualizations")
print(SEP)

bilstm_results = {}   # will be populated in Section 9; needed here for charts 8.12/8.13

metric_names = ["accuracy", "f1", "precision", "recall"]

# 8.1 Model comparison (4 metrics x 3 langs)
fig, axes = plt.subplots(1, 4, figsize=(24, 5), sharey=True)
for idx, metric in enumerate(metric_names):
    ax = axes[idx]
    x  = np.arange(len(models_list))
    w  = 0.25
    for i, lang in enumerate(langs):
        vals = [all_results.get((lang, m), {}).get(metric, 0) * 100 for m in models_list]
        ax.bar(x + i * w, vals, w, label=lang, color=PALETTE[i])
    ax.set_xticks(x + w)
    ax.set_xticklabels(models_list, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("%" if idx == 0 else "")
    ax.set_title(metric.capitalize(), fontsize=11, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=7)
fig.suptitle("ML Model Comparison Across Languages (EN / AR / Combined)",
             fontsize=14, fontweight="bold", y=1.03)
fig.tight_layout()
charts["model_comparison"] = fig_to_b64(fig)
print("  [OK] Model comparison")

# 8.2 Best ML confusion matrix
best_cm, best_cm_labels = confusion_data[best_key]
n_cm = len(best_cm_labels)
fig, ax = plt.subplots(figsize=(max(8, n_cm * 0.6), max(7, n_cm * 0.55)))
sns.heatmap(best_cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=best_cm_labels, yticklabels=best_cm_labels, ax=ax,
            linewidths=0.5, linecolor="white")
ax.set_xlabel("Predicted", fontsize=11)
ax.set_ylabel("Actual", fontsize=11)
ax.set_title(f"Confusion Matrix — {best_key[1]} ({best_key[0]})",
             fontsize=12, fontweight="bold")
plt.xticks(rotation=40, ha="right", fontsize=7)
plt.yticks(fontsize=7)
fig.tight_layout()
charts["confusion_matrix"] = fig_to_b64(fig)
print("  [OK] Confusion matrix (best model)")

# 8.3 Accuracy by language
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(langs))
w = 0.13
for i, m_name in enumerate(models_list):
    vals = [all_results.get((lang, m_name), {}).get("accuracy", 0) * 100 for lang in langs]
    ax.bar(x + i * w, vals, w, label=m_name, color=PALETTE[i + 3])
ax.set_xticks(x + w * (len(models_list) - 1) / 2)
ax.set_xticklabels(langs, fontsize=11)
ax.set_ylabel("Accuracy %")
ax.set_title("Accuracy by Language Configuration", fontsize=13, fontweight="bold")
ax.legend(fontsize=8, ncol=2)
ax.set_ylim(0, 110)
fig.tight_layout()
charts["accuracy_by_lang"] = fig_to_b64(fig)
print("  [OK] Accuracy by language")

# 8.4 Cross-validation box plots
fig, ax = plt.subplots(figsize=(14, 5))
cv_data_list, cv_lbl_list, cv_color_list = [], [], []
for lang_i, lang in enumerate(langs):
    for m_name in models_list:
        key = (lang, m_name)
        if key in cv_results:
            cv_data_list.append([s * 100 for s in cv_results[key]["scores"]])
            cv_lbl_list.append(f"{m_name[:10]}\n({lang[:3]})")
            cv_color_list.append(PALETTE[lang_i])
bp = ax.boxplot(cv_data_list, labels=cv_lbl_list, patch_artist=True)
for patch, color in zip(bp["boxes"], cv_color_list):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel("Accuracy %")
ax.set_title(f"Cross-Validation Score Distribution ({N_CV}-Fold Stratified)",
             fontsize=13, fontweight="bold")
plt.xticks(fontsize=7, rotation=45, ha="right")
fig.tight_layout()
charts["cv_comparison"] = fig_to_b64(fig)
print("  [OK] Cross-validation comparison")

# 8.5 All results heatmap
hm_data, hm_labels = [], []
for lang in langs:
    for m_name in models_list:
        r = all_results.get((lang, m_name), {})
        hm_data.append([r.get(m, 0) * 100 for m in metric_names])
        hm_labels.append(f"{m_name} ({lang[:3]})")
fig, ax = plt.subplots(figsize=(8, max(6, len(hm_labels) * 0.42)))
sns.heatmap(np.array(hm_data), annot=True, fmt=".1f", cmap="YlGnBu",
            xticklabels=[m.capitalize() for m in metric_names],
            yticklabels=hm_labels, ax=ax, linewidths=0.5, linecolor="white",
            vmin=0, vmax=100)
ax.set_title("All Model Results — Metric Heatmap (%)", fontsize=13, fontweight="bold")
plt.yticks(fontsize=8)
fig.tight_layout()
charts["results_heatmap"] = fig_to_b64(fig)
print("  [OK] Results heatmap")

# 8.6 Confusion matrices for best model type across all languages
best_model_name = best_key[1]
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
for ax, lang in zip(axes, langs):
    key = (lang, best_model_name)
    if key in confusion_data:
        cm_data, cm_labels = confusion_data[key]
        sns.heatmap(cm_data, annot=True, fmt="d", cmap="Blues",
                    xticklabels=cm_labels, yticklabels=cm_labels, ax=ax,
                    linewidths=0.3, linecolor="white")
        acc_val = all_results[key]["accuracy"] * 100
        ax.set_title(f"{lang}  (Acc = {acc_val:.1f}%)", fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("Actual", fontsize=8)
        ax.tick_params(labelsize=6)
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
fig.suptitle(f"Confusion Matrices — {best_model_name} Across Languages",
             fontsize=13, fontweight="bold", y=1.03)
fig.tight_layout()
charts["cm_all_langs"] = fig_to_b64(fig)
print("  [OK] Confusion matrices across languages")

# 8.7 Feature importance (Random Forest — Combined)
_rf_tmp = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
X_comb  = tfidf_matrices["Combined"]
X_tr, X_te, y_tr, y_te = train_test_split(X_comb, y, test_size=TEST_SIZE, random_state=42, stratify=y)
_rf_tmp.fit(X_tr, y_tr)
feat_names  = tfidf_models["Combined"].get_feature_names_out()
importances = _rf_tmp.feature_importances_
top_idx     = np.argsort(importances)[-25:]
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(range(len(top_idx)), importances[top_idx], color=PALETTE[0], edgecolor="white")
ax.set_yticks(range(len(top_idx)))
ax.set_yticklabels(feat_names[top_idx], fontsize=8)
ax.set_xlabel("Feature Importance")
ax.set_title("Top 25 Most Important Features  (Random Forest, Combined EN+AR)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
charts["feature_importance"] = fig_to_b64(fig)
print("  [OK] Feature importance chart")

# 8.8 Per-class accuracy (best model)
best_cm_full, best_lbls_full = confusion_data[best_key]
row_sums = best_cm_full.sum(axis=1)
row_sums = np.where(row_sums == 0, 1, row_sums)
per_class_acc = best_cm_full.diagonal() / row_sums
pc_colors = [PALETTE[0] if v >= 0.8 else (PALETTE[1] if v >= 0.5 else PALETTE[3])
             for v in per_class_acc]
fig, ax = plt.subplots(figsize=(12, max(5, len(best_lbls_full) * 0.45)))
ax.barh(best_lbls_full[::-1], (per_class_acc * 100)[::-1],
        color=pc_colors[::-1], edgecolor="white")
ax.set_xlabel("Accuracy %")
ax.set_xlim(0, 115)
ax.axvline(80, color="red", linestyle="--", linewidth=0.8, label="80% threshold")
ax.legend(fontsize=9)
ax.set_title(f"Per-Class Accuracy — {best_key[1]} ({best_key[0]})",
             fontsize=12, fontweight="bold")
for i, v in enumerate(per_class_acc[::-1]):
    ax.text(v * 100 + 1, i, f"{v*100:.0f}%", va="center", fontsize=8)
fig.tight_layout()
charts["per_class_acc"] = fig_to_b64(fig)
print("  [OK] Per-class accuracy chart")

# ---------------------------------------------------------------------------
#  8.9  Performance Metrics — AUC-ROC comparison (3 languages)
# ---------------------------------------------------------------------------
_perf_metrics_names = ["accuracy", "f1", "precision", "recall", "roc_auc"]
fig, axes = plt.subplots(1, 5, figsize=(28, 5), sharey=True)
for idx, m in enumerate(_perf_metrics_names):
    ax = axes[idx]
    x  = np.arange(len(models_list))
    w  = 0.25
    for li, lang in enumerate(langs):
        vals = [all_results.get((lang, mn), {}).get(m, np.nan) for mn in models_list]
        vals = [v * 100 if not np.isnan(v) else 0 for v in vals]
        ax.bar(x + li * w, vals, w, label=lang, color=PALETTE[li])
    ax.set_xticks(x + w)
    ax.set_xticklabels(models_list, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("%" if idx == 0 else "")
    ax.set_title(m.replace("_", " ").title(), fontsize=11, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=7)
fig.suptitle("Performance Metrics Comparison — Accuracy / F1 / Precision / Recall / AUC-ROC",
             fontsize=13, fontweight="bold", y=1.03)
fig.tight_layout()
charts["perf_metrics_compare"] = fig_to_b64(fig)
print("  [OK] Performance metrics comparison chart (5 metrics)")

# ---------------------------------------------------------------------------
#  8.10  AUC-ROC grouped bar (by language)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(models_list))
w = 0.25
for li, lang in enumerate(langs):
    vals = [all_results.get((lang, mn), {}).get("roc_auc", np.nan) for mn in models_list]
    clean = [v if not np.isnan(v) else 0 for v in vals]
    bars = ax.bar(x + li * w, clean, w, label=lang, color=PALETTE[li], edgecolor="white")
ax.set_xticks(x + w)
ax.set_xticklabels(models_list, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("AUC-ROC Score")
ax.set_ylim(0, 1.1)
ax.axhline(0.5, color="#ccc", linestyle="--", linewidth=0.8, label="Random baseline (0.5)")
ax.set_title("AUC-ROC Comparison Across Languages", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
fig.tight_layout()
charts["auc_roc_chart"] = fig_to_b64(fig)
print("  [OK] AUC-ROC comparison chart")

# ---------------------------------------------------------------------------
#  8.11  Log Loss comparison (lower = better)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5))
for li, lang in enumerate(langs):
    vals = [all_results.get((lang, mn), {}).get("log_loss", np.nan) for mn in models_list]
    clean = [v if not np.isnan(v) else 0 for v in vals]
    ax.bar(x + li * w, clean, w, label=lang, color=PALETTE[li + 3], edgecolor="white")
ax.set_xticks(x + w)
ax.set_xticklabels(models_list, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Log Loss  (lower is better)")
ax.set_title("Log Loss Comparison Across Languages", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
fig.tight_layout()
charts["log_loss_chart"] = fig_to_b64(fig)
print("  [OK] Log Loss comparison chart")

# ---------------------------------------------------------------------------
#  8.12  Comprehensive 6-metric heatmap (all models + all metrics)
# ---------------------------------------------------------------------------
_full_metrics  = ["accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"]
_full_labels   = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC", "Log Loss"]
hm6_data, hm6_labels = [], []
for lang in langs:
    for m_name in models_list:
        r = all_results.get((lang, m_name), {})
        row = []
        for m in _full_metrics:
            v = r.get(m, np.nan)
            if m == "log_loss":
                # Invert log loss so higher = better on same scale
                row.append(round(1 / (1 + v), 4) * 100 if not np.isnan(v) else 0)
            else:
                row.append((v * 100) if not np.isnan(v) else 0)
        hm6_data.append(row)
        hm6_labels.append(f"{m_name} ({lang[:3]})")
if bilstm_results:
    _br = bilstm_results
    ll_inv = round(1 / (1 + _br.get("log_loss", 0)), 4) * 100 if not np.isnan(_br.get("log_loss", 0)) else 0
    hm6_data.append([
        _br.get("accuracy", 0)*100, _br.get("precision", 0)*100, _br.get("recall", 0)*100,
        _br.get("f1", 0)*100, (_br.get("roc_auc", 0))*100 if not np.isnan(_br.get("roc_auc", 0)) else 0,
        ll_inv,
    ])
    hm6_labels.append("BiLSTM (Com)")

fig, ax = plt.subplots(figsize=(10, max(7, len(hm6_labels) * 0.44)))
sns.heatmap(np.array(hm6_data, dtype=float), annot=True, fmt=".1f", cmap="YlGnBu",
            xticklabels=_full_labels, yticklabels=hm6_labels,
            ax=ax, linewidths=0.5, linecolor="white", vmin=0, vmax=100)
ax.set_title("All 6 Performance Metrics — Heatmap\n(Log Loss shown as inverted score: higher = better)",
             fontsize=12, fontweight="bold")
plt.yticks(fontsize=8)
fig.tight_layout()
charts["perf_heatmap_6"] = fig_to_b64(fig)
print("  [OK] 6-metric comprehensive heatmap")

# ---------------------------------------------------------------------------
#  8.13  Column chart — top 6 models by accuracy (5 metrics excl. Log Loss)
# ---------------------------------------------------------------------------
_col_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
_col_labels  = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]

# Pick top 6 models by accuracy (best language per model name)
_top_keys = sorted(all_results.keys(),
                   key=lambda k: all_results[k].get("accuracy", 0), reverse=True)[:6]

fig, ax = plt.subplots(figsize=(12, 7))
n_models  = len(_top_keys)
n_metrics = len(_col_metrics)
bar_width = 0.13
x = np.arange(n_metrics)

for ci, key in enumerate(_top_keys):
    r = all_results[key]
    vals = [r.get(m, 0) if not np.isnan(r.get(m, 0)) else 0 for m in _col_metrics]
    offset = (ci - n_models / 2) * bar_width + bar_width / 2
    bars = ax.bar(x + offset, vals, bar_width, label=f"{key[1][:14]} ({key[0][:3]})",
                  color=PALETTE[ci], edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val*100:.0f}%", ha="center", va="bottom", fontsize=7, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(_col_labels, fontsize=11)
ax.set_ylim(0, 1.12)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"], fontsize=9)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("Performance Comparison — Top 6 Models", fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)
fig.tight_layout()
charts["radar_chart"] = fig_to_b64(fig)
print("  [OK] Column chart (top 6 models)")

# ===========================================================================
#  SECTION 9 — BiLSTM DEEP LEARNING
# ===========================================================================
print(f"\n{SEP}")
print("  SECTION 9 — BiLSTM Deep Learning Training")
print(SEP)

bilstm_results = {}

if not HAS_TORCH:
    print("  [SKIP] PyTorch not installed. BiLSTM training skipped.")
else:
    MAX_VOCAB = 8000
    MAX_LEN   = 128
    EMBED_DIM = 128
    HIDDEN    = 256
    N_LAYERS  = 2
    DROPOUT   = 0.3
    BATCH     = 32
    EPOCHS    = 25
    LR        = 1e-3

    all_tokens  = [w for t in ml_df["text_combined"].tolist() for w in t.lower().split()]
    vocab_cnt   = Counter(all_tokens)
    vocab_words = ["<PAD>", "<UNK>"] + [w for w, _ in vocab_cnt.most_common(MAX_VOCAB - 2)]
    w2i         = {w: i for i, w in enumerate(vocab_words)}

    print(f"  Vocabulary  : {len(w2i):,} tokens  (corpus unique: {len(vocab_cnt):,})")
    print(f"  Seq length  : {MAX_LEN}  |  Embed: {EMBED_DIM}  |  Hidden: {HIDDEN} (x2 bidirectional)")
    print(f"  LSTM layers : {N_LAYERS}  |  Dropout: {DROPOUT}  |  Batch: {BATCH}")
    print(f"  Epochs      : {EPOCHS}   |  LR: {LR}  |  Weight decay: 1e-4")

    def encode_text(text, max_len=MAX_LEN):
        ids = [w2i.get(w, 1) for w in text.lower().split()][:max_len]
        return ids + [0] * (max_len - len(ids))

    class DrugDataset(TorchDataset):
        def __init__(self, texts, labels):
            self.X = [torch.tensor(encode_text(t), dtype=torch.long) for t in texts]
            self.y = torch.tensor(labels, dtype=torch.long)
        def __len__(self):
            return len(self.y)
        def __getitem__(self, i):
            return self.X[i], self.y[i]

    class BiLSTMClassifier(nn.Module):
        def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=N_LAYERS,
                                batch_first=True, bidirectional=True, dropout=DROPOUT)
            self.attention = nn.Linear(hidden_dim * 2, 1)
            self.fc = nn.Sequential(
                nn.Dropout(DROPOUT),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
                nn.Linear(hidden_dim, num_classes),
            )
        def forward(self, x):
            emb = self.embedding(x)
            out, _ = self.lstm(emb)
            attn = torch.softmax(self.attention(out), dim=1)
            return self.fc((out * attn).sum(dim=1))

    tr_texts, te_texts, tr_labels, te_labels = train_test_split(
        ml_df["text_combined"].fillna("").tolist(), y.tolist(),
        test_size=TEST_SIZE, random_state=42, stratify=y)

    train_dl = DataLoader(DrugDataset(tr_texts, tr_labels), batch_size=BATCH, shuffle=True)
    test_dl  = DataLoader(DrugDataset(te_texts, te_labels), batch_size=BATCH)

    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_bilstm = BiLSTMClassifier(len(w2i), EMBED_DIM, HIDDEN, n_classes).to(device)
    total_params = sum(p.numel() for p in model_bilstm.parameters())
    print(f"\n  Device      : {device}")
    print(f"  Parameters  : {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_bilstm.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    best_val_acc = 0.0
    best_state   = None
    epoch_hist   = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

    print(f"\n  {'Ep':>4} {'Tr Loss':>9} {'Tr Acc':>8} {'Va Loss':>9} {'Va Acc':>8} {'LR':>10}")
    print("  " + SSEP[:54])

    bilstm_t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model_bilstm.train()
        tr_c = tr_n = 0
        tr_loss_s = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model_bilstm(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model_bilstm.parameters(), 1.0)
            optimizer.step()
            tr_c += (logits.argmax(1) == yb).sum().item()
            tr_n += xb.size(0)
            tr_loss_s += loss.item() * xb.size(0)
        scheduler.step()

        model_bilstm.eval()
        va_c = va_n = 0
        va_loss_s = 0.0
        with torch.no_grad():
            for xb, yb in test_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model_bilstm(xb)
                va_loss_s += criterion(logits, yb).item() * xb.size(0)
                va_c += (logits.argmax(1) == yb).sum().item()
                va_n += xb.size(0)

        tr_acc  = tr_c / tr_n * 100
        va_acc  = va_c / va_n * 100
        tr_loss = tr_loss_s / tr_n
        va_loss = va_loss_s / va_n
        cur_lr  = optimizer.param_groups[0]["lr"]

        epoch_hist["train_acc"].append(tr_acc)
        epoch_hist["val_acc"].append(va_acc)
        epoch_hist["train_loss"].append(tr_loss)
        epoch_hist["val_loss"].append(va_loss)

        marker = ""
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state   = {k: v.clone() for k, v in model_bilstm.state_dict().items()}
            marker = " *"
        print(f"  {epoch:>4} {tr_loss:>9.4f} {tr_acc:>7.2f}% {va_loss:>9.4f} "
              f"{va_acc:>7.2f}%{marker} {cur_lr:>10.6f}")

    bilstm_time = time.time() - bilstm_t0
    print(f"\n  Best Val Accuracy : {best_val_acc:.2f}%")
    print(f"  Training Time     : {bilstm_time:.1f}s")

    if best_state:
        model_bilstm.load_state_dict(best_state)

    model_bilstm.eval()
    preds_all, true_all = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            preds_all.extend(model_bilstm(xb.to(device)).argmax(1).cpu().tolist())
            true_all.extend(yb.tolist())

    bilstm_acc  = accuracy_score(true_all, preds_all)
    bilstm_bac  = balanced_accuracy_score(true_all, preds_all)
    bilstm_f1   = f1_score(true_all, preds_all, average="weighted", zero_division=0)
    bilstm_prec = precision_score(true_all, preds_all, average="weighted", zero_division=0)
    bilstm_rec  = recall_score(true_all, preds_all, average="weighted", zero_division=0)
    bilstm_mcc  = matthews_corrcoef(true_all, preds_all)
    bilstm_lbls = sorted(set(true_all) | set(preds_all))
    bilstm_lbl_names = [class_names[i] for i in bilstm_lbls]
    bilstm_report = classification_report(true_all, preds_all, labels=bilstm_lbls,
                                          target_names=bilstm_lbl_names, zero_division=0)
    bilstm_cm = confusion_matrix(true_all, preds_all, labels=bilstm_lbls)

    # BiLSTM AUC-ROC and Log Loss — collect softmax probabilities
    model_bilstm.eval()
    bilstm_proba_list = []
    with torch.no_grad():
        for xb, _ in test_dl:
            logits = model_bilstm(xb.to(device))
            bilstm_proba_list.append(torch.softmax(logits, dim=1).cpu().numpy())
    bilstm_proba = np.vstack(bilstm_proba_list)
    try:
        bilstm_roc_auc   = roc_auc_score(true_all, bilstm_proba, multi_class="ovr",
                                          average="weighted", labels=list(range(n_classes)))
        bilstm_clf_logloss = log_loss(true_all, bilstm_proba, labels=list(range(n_classes)))
    except Exception:
        bilstm_roc_auc     = np.nan
        bilstm_clf_logloss = np.nan

    bilstm_results = {
        "accuracy": bilstm_acc, "balanced_accuracy": bilstm_bac,
        "f1": bilstm_f1, "precision": bilstm_prec,
        "recall": bilstm_rec, "mcc": bilstm_mcc,
        "roc_auc": bilstm_roc_auc, "log_loss": bilstm_clf_logloss,
        "report": bilstm_report, "best_val_acc": best_val_acc,
        "epochs": EPOCHS, "params": total_params, "train_time": bilstm_time,
    }

    print(f"\n  Final Test Results:")
    print(f"    Accuracy           : {bilstm_acc*100:.2f}%")
    print(f"    Balanced Accuracy  : {bilstm_bac*100:.2f}%")
    print(f"    F1 (Weighted)      : {bilstm_f1*100:.2f}%")
    print(f"    Precision          : {bilstm_prec*100:.2f}%")
    print(f"    Recall             : {bilstm_rec*100:.2f}%")
    print(f"    MCC                : {bilstm_mcc:.4f}")
    print(f"    AUC-ROC            : {bilstm_roc_auc:.4f}" if not np.isnan(bilstm_roc_auc) else "    AUC-ROC            : N/A")
    print(f"    Log Loss           : {bilstm_clf_logloss:.4f}" if not np.isnan(bilstm_clf_logloss) else "    Log Loss           : N/A")
    print(f"\n  Classification Report:")
    for line in bilstm_report.strip().split("\n"):
        print(f"    {line}")

    # Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ep_x = range(1, EPOCHS + 1)
    ax1.plot(ep_x, epoch_hist["train_loss"], "o-", label="Train Loss", color=PALETTE[0], markersize=4)
    ax1.plot(ep_x, epoch_hist["val_loss"],   "s-", label="Val Loss",   color=PALETTE[1], markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("BiLSTM Loss Curve", fontsize=11, fontweight="bold")
    ax1.legend()
    ax2.plot(ep_x, epoch_hist["train_acc"], "o-", label="Train Acc", color=PALETTE[2], markersize=4)
    ax2.plot(ep_x, epoch_hist["val_acc"],   "s-", label="Val Acc",   color=PALETTE[3], markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy %")
    ax2.set_title("BiLSTM Accuracy Curve", fontsize=11, fontweight="bold")
    ax2.legend()
    fig.suptitle("BiLSTM Training Progress (Combined EN+AR)", fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    charts["bilstm_curves"] = fig_to_b64(fig)
    print("  [OK] BiLSTM training curves")

    # BiLSTM confusion matrix
    n_bcm = len(bilstm_lbl_names)
    fig, ax = plt.subplots(figsize=(max(8, n_bcm * 0.6), max(7, n_bcm * 0.55)))
    sns.heatmap(bilstm_cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=bilstm_lbl_names, yticklabels=bilstm_lbl_names, ax=ax,
                linewidths=0.5, linecolor="white")
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title("Confusion Matrix — BiLSTM (Combined EN+AR)", fontsize=12, fontweight="bold")
    plt.xticks(rotation=40, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    fig.tight_layout()
    charts["bilstm_confusion"] = fig_to_b64(fig)
    print("  [OK] BiLSTM confusion matrix")

    torch.save(model_bilstm.state_dict(), os.path.join(MODEL_DIR, "bilstm_best.pt"))
    joblib.dump(w2i, os.path.join(MODEL_DIR, "vocab.pkl"))
    print(f"  [SAVED] bilstm_best.pt  |  vocab.pkl")

# ===========================================================================
#  SECTION 10 — ENSEMBLE CONFIG + CLEAN DATASET
# ===========================================================================
print(f"\n{SEP}")
print("  SECTION 10 — Saving Ensemble Configuration & Clean Dataset")
print(SEP)

ensemble_cfg = {
    "num_text_classes": n_classes,
    "max_seq_len":      MAX_LEN if HAS_TORCH else 128,
    "alpha":            0.5,
    "embed_dim":        128,
    "hidden_dim":       256,
    "n_layers":         2,
    "dropout":          0.3,
    "tfidf_weight":     0.35,
    "ml_weight":        0.30,
    "bilstm_weight":    0.35,
    "classes":          class_names,
    "best_ml_model":    best_key[1] if best_key else "",
    "best_ml_lang":     best_key[0] if best_key else "",
    "best_ml_accuracy": round(best_acc, 4),
    "dataset":          "capsule_dataset.csv",
    "trained_date":     datetime.now().isoformat(),
}
if HAS_TORCH and bilstm_results:
    ensemble_cfg.update({
        "vocab_size":      len(w2i),
        "bilstm_accuracy": round(bilstm_results["accuracy"], 4),
    })

with open(os.path.join(MODEL_DIR, "ensemble_config.json"), "w", encoding="utf-8") as f:
    json.dump(ensemble_cfg, f, indent=2, ensure_ascii=False)
print("  [SAVED] ensemble_config.json")

capsule_out = df.rename(columns={
    "drug_name_en":    "drug_name",
    "generic_name_en": "generic_name",
    "drug_class_en":   "category",
    "dosage_form_en":  "dosage_form",
    "color_en":        "color",
    "shape_en":        "shape",
    "indications_en":  "indication",
    "manufacturer_en": "manufacturer",
    "text_combined":   "text_description",
})
capsule_out["description"] = capsule_out["text_description"]
keep = ["drug_name","generic_name","category","dosage_form","color","shape",
        "indication","manufacturer","strength","text_description","description",
        "drug_name_ar","generic_name_ar","drug_class_ar","dosage_form_ar",
        "color_ar","shape_ar","indications_ar","rx_otc_status_en",
        "side_effects_en","side_effects_ar","contraindications_en","contraindications_ar"]
keep = [c for c in keep if c in capsule_out.columns]
capsule_out[keep].to_csv(os.path.join(MODEL_DIR, "capsule_dataset_clean.csv"), index=False)
print(f"  [SAVED] capsule_dataset_clean.csv  ({len(capsule_out):,} records, {len(keep)} cols)")

# ===========================================================================
#  SECTION 11 — HTML TRAINING REPORT
# ===========================================================================
print(f"\n{SEP}")
print("  SECTION 11 — Generating HTML Training Report")
print(SEP)

total_elapsed = time.time() - start_time

def metric_cell(val, is_pct=True):
    v = val * 100 if is_pct else val
    if is_pct:
        cls = "cell-hi" if v >= 80 else ("cell-mid" if v >= 60 else "cell-lo")
        return f'<td class="{cls}">{v:.2f}%</td>'
    else:
        cls = "cell-hi" if v >= 0.6 else ("cell-mid" if v >= 0.3 else "cell-lo")
        return f'<td class="{cls}">{v:.3f}</td>'

# --- Build results table rows ---
results_rows_html = ""
rank = 0
for lang in langs:
    for m_name in models_list:
        key = (lang, m_name)
        r   = all_results.get(key)
        if not r:
            continue
        rank += 1
        is_best = (key == best_key)
        row_cls = ' class="best-row"' if is_best else ""
        badge   = ' <span class="badge badge-best">BEST</span>' if is_best else ""
        cv      = cv_results.get(key, {})
        cv_str  = f"{cv.get('mean',0)*100:.1f}% &plusmn;{cv.get('std',0)*100:.1f}"
        results_rows_html += (
            f'<tr{row_cls}><td>{rank}</td><td>{m_name}{badge}</td><td>{lang}</td>'
            + metric_cell(r["accuracy"])
            + metric_cell(r["f1"])
            + metric_cell(r["precision"])
            + metric_cell(r["recall"])
            + metric_cell(r["balanced_accuracy"])
            + metric_cell(r["mcc"], is_pct=False)
            + f"<td>{cv_str}</td><td>{r['time']:.2f}s</td></tr>\n"
        )

if bilstm_results:
    rank += 1
    results_rows_html += (
        '<tr class="bilstm-row">'
        f'<td>{rank}</td>'
        '<td>BiLSTM (Deep Learning) <span class="badge badge-dl">DL</span></td>'
        '<td>Combined EN+AR</td>'
        + metric_cell(bilstm_results["accuracy"])
        + metric_cell(bilstm_results["f1"])
        + metric_cell(bilstm_results["precision"])
        + metric_cell(bilstm_results["recall"])
        + metric_cell(bilstm_results["balanced_accuracy"])
        + metric_cell(bilstm_results["mcc"], is_pct=False)
        + f"<td>—</td><td>{bilstm_results.get('train_time',0):.1f}s</td></tr>\n"
    )

# --- Performance Metrics table rows (6 metrics) ---
def logloss_cell(v):
    if np.isnan(v):
        return "<td>N/A</td>"
    cls = "cell-hi" if v <= 0.5 else ("cell-mid" if v <= 2.0 else "cell-lo")
    return f'<td class="{cls}">{v:.4f}</td>'

def roc_cell(v):
    if np.isnan(v):
        return "<td>N/A</td>"
    cls = "cell-hi" if v >= 0.9 else ("cell-mid" if v >= 0.7 else "cell-lo")
    return f'<td class="{cls}">{v:.4f}</td>'

perf_table_rows_html = ""
_perf_rank = 0
for lang in langs:
    for m_name in models_list:
        key = (lang, m_name)
        r   = all_results.get(key)
        if not r:
            continue
        _perf_rank += 1
        is_best = (key == best_key)
        row_cls = ' class="best-row"' if is_best else ""
        badge   = ' <span class="badge badge-best">BEST</span>' if is_best else ""
        perf_table_rows_html += (
            f'<tr{row_cls}><td>{_perf_rank}</td><td>{m_name}{badge}</td><td>{lang}</td>'
            + metric_cell(r["accuracy"])
            + metric_cell(r["precision"])
            + metric_cell(r["recall"])
            + metric_cell(r["f1"])
            + roc_cell(r.get("roc_auc", np.nan))
            + logloss_cell(r.get("log_loss", np.nan))
            + "</tr>\n"
        )
if bilstm_results:
    _perf_rank += 1
    _br = bilstm_results
    perf_table_rows_html += (
        '<tr class="bilstm-row">'
        f'<td>{_perf_rank}</td>'
        '<td>BiLSTM (Deep Learning) <span class="badge badge-dl">DL</span></td>'
        '<td>Combined EN+AR</td>'
        + metric_cell(_br.get("accuracy", 0))
        + metric_cell(_br.get("precision", 0))
        + metric_cell(_br.get("recall", 0))
        + metric_cell(_br.get("f1", 0))
        + roc_cell(_br.get("roc_auc", np.nan))
        + logloss_cell(_br.get("log_loss", np.nan))
        + "</tr>\n"
    )

# --- Drug class table ---
class_table_html = ""
for cls_en, cnt in class_counts.items():
    ar_val = df[df["drug_class_en"] == cls_en]["drug_class_ar"].iloc[0]
    class_table_html += (
        f"<tr><td>{cls_en}</td>"
        f"<td dir='rtl' class='ar-text'>{safe(ar_val)}</td>"
        f"<td>{cnt}</td><td>{pct(cnt, len(df))}</td></tr>\n"
    )

# --- Classification reports ---
reports_html = ""
for lang in langs:
    reports_html += f'<h3>{lang} Models</h3>\n'
    for m_name in models_list:
        key = (lang, m_name)
        r   = all_results.get(key)
        if not r:
            continue
        is_best = (key == best_key)
        badge   = ' <span class="badge badge-best">BEST</span>' if is_best else ""
        reports_html += (
            f'<details class="report-block"><summary>{m_name} ({lang}){badge}</summary>'
            f'<pre>{r["report"]}</pre></details>\n'
        )
if bilstm_results:
    reports_html += (
        '<h3>BiLSTM Deep Learning</h3>'
        '<details class="report-block"><summary>BiLSTM (Combined EN+AR) '
        '<span class="badge badge-dl">DL</span></summary>'
        f'<pre>{bilstm_results["report"]}</pre></details>\n'
    )

# --- CV table ---
cv_table_html = ""
for lang in langs:
    for m_name in models_list:
        key = (lang, m_name)
        cv  = cv_results.get(key, {})
        if not cv:
            continue
        scores_str = ", ".join(f"{s*100:.1f}%" for s in cv.get("scores", []))
        cv_table_html += (
            f"<tr><td>{m_name}</td><td>{lang}</td>"
            + metric_cell(cv.get("mean", 0))
            + f"<td>&plusmn;{cv.get('std',0)*100:.2f}%</td>"
            f"<td class='cv-scores'>{scores_str}</td></tr>\n"
        )

# --- BiLSTM section ---
bilstm_section_html = ""
if bilstm_results:
    bilstm_section_html = f"""
<section id="bilstm">
  <h2>7. BiLSTM Deep Learning Results</h2>
  <div class="stats-grid">
    <div class="stat-card accent"><div class="stat-val">{bilstm_results['accuracy']*100:.1f}%</div><div class="stat-lbl">Accuracy</div></div>
    <div class="stat-card"><div class="stat-val">{bilstm_results['balanced_accuracy']*100:.1f}%</div><div class="stat-lbl">Balanced Acc</div></div>
    <div class="stat-card"><div class="stat-val">{bilstm_results['f1']*100:.1f}%</div><div class="stat-lbl">F1 Score</div></div>
    <div class="stat-card"><div class="stat-val">{bilstm_results['precision']*100:.1f}%</div><div class="stat-lbl">Precision</div></div>
    <div class="stat-card"><div class="stat-val">{bilstm_results['recall']*100:.1f}%</div><div class="stat-lbl">Recall</div></div>
    <div class="stat-card"><div class="stat-val">{bilstm_results['mcc']:.3f}</div><div class="stat-lbl">MCC</div></div>
    <div class="stat-card"><div class="stat-val">{bilstm_results['best_val_acc']:.1f}%</div><div class="stat-lbl">Best Val Acc</div></div>
    <div class="stat-card"><div class="stat-val">{total_params:,}</div><div class="stat-lbl">Parameters</div></div>
    <div class="stat-card"><div class="stat-val">{EPOCHS}</div><div class="stat-lbl">Epochs</div></div>
    <div class="stat-card"><div class="stat-val">{bilstm_results.get('train_time',0):.0f}s</div><div class="stat-lbl">Train Time</div></div>
  </div>
  <h3>Model Architecture</h3>
  <table style="max-width:520px">
    <tr><td><strong>Architecture</strong></td><td>BiLSTM with Soft Attention</td></tr>
    <tr><td><strong>Embedding</strong></td><td>{EMBED_DIM} dimensions</td></tr>
    <tr><td><strong>LSTM Hidden</strong></td><td>{HIDDEN} units &times; {N_LAYERS} layers (bidirectional = {HIDDEN*2})</td></tr>
    <tr><td><strong>Attention</strong></td><td>Single-head soft attention over all time steps</td></tr>
    <tr><td><strong>Dropout</strong></td><td>{DROPOUT}</td></tr>
    <tr><td><strong>Optimizer</strong></td><td>Adam (lr={LR}, weight_decay=1e-4)</td></tr>
    <tr><td><strong>Scheduler</strong></td><td>StepLR (step=7, gamma=0.5)</td></tr>
    <tr><td><strong>Vocabulary</strong></td><td>{len(w2i):,} tokens</td></tr>
    <tr><td><strong>Max Sequence</strong></td><td>{MAX_LEN} tokens</td></tr>
    <tr><td><strong>Input</strong></td><td>Combined English + Arabic text</td></tr>
  </table>
  <h3>Training Curves</h3>
  <img src="data:image/png;base64,{charts.get('bilstm_curves','')}" alt="BiLSTM Training Curves">
  <h3>Confusion Matrix</h3>
  <img src="data:image/png;base64,{charts.get('bilstm_confusion','')}" alt="BiLSTM Confusion Matrix">
</section>"""

# --- Dataset stats for overview ---
overview_stats_html = (
    f"<tr><td>Total Records</td><td>{len(df):,}</td></tr>"
    f"<tr><td>Total Columns</td><td>{len(df.columns)}</td></tr>"
    f"<tr><td>English Columns</td><td>{len(EN_COLS)}</td></tr>"
    f"<tr><td>Arabic Columns</td><td>{len(AR_COLS)}</td></tr>"
    f"<tr><td>Unique Drugs (EN)</td><td>{df['drug_name_en'].nunique()}</td></tr>"
    f"<tr><td>Unique Drugs (AR)</td><td>{df['drug_name_ar'].nunique()}</td></tr>"
    f"<tr><td>Unique Generic Names</td><td>{df['generic_name_en'].nunique()}</td></tr>"
    f"<tr><td>Drug Classes</td><td>{df['drug_class_en'].nunique()}</td></tr>"
    f"<tr><td>Dosage Forms</td><td>{df['dosage_form_en'].nunique()}</td></tr>"
    f"<tr><td>Manufacturers</td><td>{df['manufacturer_en'].nunique()}</td></tr>"
    f"<tr><td>Colors</td><td>{df['color_en'].nunique()}</td></tr>"
    f"<tr><td>Shapes</td><td>{df['shape_en'].nunique()}</td></tr>"
    f"<tr><td>Unique Strengths</td><td>{df['strength'].nunique() if 'strength' in df.columns else 0}</td></tr>"
    f"<tr><td>Missing Values</td><td>{total_missing}</td></tr>"
    f"<tr><td>Duplicate Rows</td><td>{n_dup_all}</td></tr>"
    f"<tr><td>Class Imbalance Ratio</td><td>{imbalance_ratio:.2f}:1</td></tr>"
    f"<tr><td>Gini Index (class balance)</td><td>{gini:.4f}</td></tr>"
    f"<tr><td>Memory Usage</td><td>{df.memory_usage(deep=True).sum() / 1024:.1f} KB</td></tr>"
)

# --- Strength chart conditional include ---
strength_chart_html = ""
if "strength_dist" in charts:
    strength_chart_html = (
        f'<div><h3>Strength Distribution (Top 15)</h3>'
        f'<img src="data:image/png;base64,{charts["strength_dist"]}" alt="Strength"></div>'
    )

source_chart_html = ""
if "source_dist" in charts:
    source_chart_html = (
        f'<div><h3>Source Dataset</h3>'
        f'<img src="data:image/png;base64,{charts["source_dist"]}" alt="Source"></div>'
    )

fi_chart_html = ""
if "feature_importance" in charts:
    fi_chart_html = (
        f'<h3>Feature Importance (Random Forest — Combined EN+AR)</h3>'
        f'<img src="data:image/png;base64,{charts["feature_importance"]}" alt="Feature Importance">'
    )

bilstm_summary_cell = (
    f"Yes &mdash; {bilstm_results.get('accuracy',0)*100:.1f}% accuracy, "
    f"{total_params:,} params, {EPOCHS} epochs"
    if bilstm_results else "No (PyTorch not available)"
)

# ===========================================================================
#  ASSEMBLE HTML
# ===========================================================================
HTML = """<!DOCTYPE html>
<html lang="en" dir="ltr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CAPSULE AI &mdash; Training Report</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;background:#f0f2f5;color:#333;line-height:1.6}

header{background:linear-gradient(135deg,#0d1b2a 0%,#1b2838 50%,#1a237e 100%);color:#fff;padding:2.5rem 2rem;text-align:center}
header h1{font-size:2.2rem;margin-bottom:.4rem;letter-spacing:1px}
header .sub{opacity:.8;font-size:.95rem;margin:.2rem 0}
.header-stats{display:flex;justify-content:center;gap:2rem;margin-top:1.4rem;flex-wrap:wrap}
.header-stat .hs-val{font-size:1.9rem;font-weight:700}
.header-stat .hs-lbl{font-size:.78rem;opacity:.7}

nav{background:#1a237e;padding:.6rem 2rem;text-align:center;position:sticky;top:0;z-index:100;box-shadow:0 2px 6px rgba(0,0,0,.2)}
nav a{color:#e8eaf6;text-decoration:none;margin:0 .7rem;font-size:.85rem;padding:.3rem .6rem;border-radius:4px;transition:background .2s}
nav a:hover{background:rgba(255,255,255,.2)}

.container{max-width:1300px;margin:0 auto;padding:1.5rem}
section{background:#fff;border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,.06);padding:2rem;margin-bottom:1.5rem}
h2{color:#1a237e;border-bottom:3px solid #3949ab;padding-bottom:.5rem;margin-bottom:1.2rem;font-size:1.35rem}
h3{color:#283593;margin:1.3rem 0 .7rem;font-size:1.1rem}
h4{color:#37474f;margin:.8rem 0 .5rem}

table{width:100%;border-collapse:collapse;margin:.8rem 0;font-size:.87rem}
th{background:#e8eaf6;color:#1a237e;padding:.65rem .8rem;text-align:left;font-weight:600;position:sticky;top:42px;z-index:9}
td{padding:.55rem .8rem;border-bottom:1px solid #eeeeee}
tr:hover td{background:#f8f9ff}
.best-row td{background:#e8f5e9 !important;font-weight:600}
.best-row:hover td{background:#c8e6c9 !important}
.bilstm-row td{background:#fff3e0 !important}
.bilstm-row:hover td{background:#ffe0b2 !important}
.ar-text{font-family:'Segoe UI','Traditional Arabic','Noto Sans Arabic',sans-serif;font-size:.95rem}
.cv-scores{font-size:.75rem;color:#666}

.cell-hi{color:#1b5e20;font-weight:600;background:#e8f5e9}
.cell-mid{color:#e65100;font-weight:600;background:#fff3e0}
.cell-lo{color:#b71c1c;font-weight:600;background:#ffebee}

.badge{padding:2px 8px;border-radius:4px;font-size:.7rem;font-weight:700;margin-left:6px;vertical-align:middle}
.badge-best{background:#2e7d32;color:#fff}
.badge-dl{background:#e65100;color:#fff}

img{max-width:100%;height:auto;border-radius:8px;margin:.6rem 0;box-shadow:0 1px 6px rgba(0,0,0,.08);cursor:zoom-in}
img.zoomed{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);max-width:95vw;max-height:95vh;z-index:9999;cursor:zoom-out;box-shadow:0 8px 40px rgba(0,0,0,.5)}
.img-overlay{display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,.7);z-index:9998}
.img-overlay.active{display:block}

.stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:.8rem;margin:.8rem 0}
.stat-card{background:#f8f9fa;border-radius:10px;padding:1rem;text-align:center;border:1px solid #e0e0e0;transition:transform .15s,box-shadow .15s}
.stat-card:hover{transform:translateY(-2px);box-shadow:0 4px 12px rgba(0,0,0,.1)}
.stat-card.accent{background:linear-gradient(135deg,#e8eaf6,#c5cae9);border-color:#9fa8da}
.stat-val{font-size:1.5rem;font-weight:700;color:#1a237e}
.stat-lbl{font-size:.76rem;color:#666;margin-top:.3rem}

.chart-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:1.2rem}
.chart-grid img,.chart-full img{width:100%}

pre{background:#f5f5f5;padding:1rem;border-radius:8px;overflow-x:auto;font-size:.78rem;line-height:1.5;border:1px solid #e0e0e0}
details.report-block{margin-bottom:.8rem;border:1px solid #e0e0e0;border-radius:8px;overflow:hidden}
details.report-block summary{padding:.8rem 1rem;cursor:pointer;background:#f8f9fa;font-weight:600;color:#1a237e;user-select:none}
details.report-block summary:hover{background:#e8eaf6}
details.report-block pre{border-radius:0;border:none;border-top:1px solid #e0e0e0}

.sortable th{cursor:pointer;user-select:none}
.sortable th:hover{background:#c5cae9}
.sortable th::after{content:' \\25B8';opacity:.4;font-size:.75rem}
.sortable th.asc::after{content:' \\25B4';opacity:1}
.sortable th.desc::after{content:' \\25BE';opacity:1}

footer{text-align:center;padding:1.5rem;color:#888;font-size:.82rem;background:#fff;margin-top:1rem;border-top:1px solid #eee}

@media print{nav,footer{display:none}section{box-shadow:none;break-inside:avoid}}
</style>
</head>
<body>
<div class="img-overlay" id="overlay" onclick="closeZoom()"></div>

<header>
  <h1>CAPSULE AI &mdash; Model Training Report</h1>
  <p class="sub">Bilingual (English + Arabic) Pharmaceutical Classification Pipeline</p>
  <p class="sub">Dataset: capsule_dataset.csv &nbsp;|&nbsp; {DATE}</p>
  <div class="header-stats">
    <div class="header-stat"><div class="hs-val">{NREC}</div><div class="hs-lbl">Records</div></div>
    <div class="header-stat"><div class="hs-val">{NCLASS}</div><div class="hs-lbl">Drug Classes</div></div>
    <div class="header-stat"><div class="hs-val">{NMODELS}</div><div class="hs-lbl">Models Trained</div></div>
    <div class="header-stat"><div class="hs-val">{BESTACC}</div><div class="hs-lbl">Best Accuracy</div></div>
    <div class="header-stat"><div class="hs-val">{ELAPSED}</div><div class="hs-lbl">Total Time</div></div>
  </div>
</header>

<nav>
  <a href="#overview">Overview</a>
  <a href="#distributions">Distributions</a>
  <a href="#charts">Charts</a>
  <a href="#features">Features</a>
  <a href="#results">Results</a>
  <a href="#perf-metrics">Performance</a>
  <a href="#cv">Cross-Val</a>
  <a href="#bilstm">BiLSTM</a>
  <a href="#reports">Reports</a>
  <a href="#summary">Summary</a>
</nav>

<div class="container">

<section id="overview">
  <h2>1. Dataset Overview</h2>
  <div class="stats-grid">
    <div class="stat-card accent"><div class="stat-val">{NREC}</div><div class="stat-lbl">Total Records</div></div>
    <div class="stat-card"><div class="stat-val">{NUNIQ}</div><div class="stat-lbl">Unique Drugs</div></div>
    <div class="stat-card"><div class="stat-val">{NCLASS}</div><div class="stat-lbl">Drug Classes</div></div>
    <div class="stat-card"><div class="stat-val">{NFORM}</div><div class="stat-lbl">Dosage Forms</div></div>
    <div class="stat-card"><div class="stat-val">{NMFR}</div><div class="stat-lbl">Manufacturers</div></div>
    <div class="stat-card"><div class="stat-val">{NCOL}</div><div class="stat-lbl">Colors</div></div>
    <div class="stat-card"><div class="stat-val">{NSHAPE}</div><div class="stat-lbl">Shapes</div></div>
    <div class="stat-card"><div class="stat-val">{NCOLS}</div><div class="stat-lbl">Columns</div></div>
  </div>
  <h3>Detailed Statistics</h3>
  <table><tr><th>Metric</th><th>Value</th></tr>{OVERVIEW_STATS}</table>
</section>

<section id="distributions">
  <h2>2. Drug Class Distribution (English &amp; Arabic)</h2>
  <table>
    <tr><th>Drug Class (English)</th><th>Drug Class (Arabic)</th><th>Count</th><th>%</th></tr>
    {CLASS_TABLE}
  </table>
  <img src="data:image/png;base64,{IMG_CLASS}" alt="Drug Class Distribution" onclick="zoomImg(this)">
  <h3>Class Imbalance View</h3>
  <img src="data:image/png;base64,{IMG_IMBALANCE}" alt="Class Imbalance" onclick="zoomImg(this)">
</section>

<section id="charts">
  <h2>3. Dataset Visualizations</h2>
  <div class="chart-grid">
    <div><h3>Dosage Forms</h3><img src="data:image/png;base64,{IMG_FORM}" alt="Dosage Forms" onclick="zoomImg(this)"></div>
    <div><h3>Top 10 Manufacturers</h3><img src="data:image/png;base64,{IMG_MFR}" alt="Manufacturers" onclick="zoomImg(this)"></div>
    <div><h3>Color Distribution</h3><img src="data:image/png;base64,{IMG_COLOR}" alt="Colors" onclick="zoomImg(this)"></div>
    <div><h3>Shape Distribution</h3><img src="data:image/png;base64,{IMG_SHAPE}" alt="Shapes" onclick="zoomImg(this)"></div>
    <div><h3>Prescription vs OTC</h3><img src="data:image/png;base64,{IMG_RXOTC}" alt="RX OTC" onclick="zoomImg(this)"></div>
    <div><h3>Arabic Column Coverage</h3><img src="data:image/png;base64,{IMG_ARCOV}" alt="Arabic Coverage" onclick="zoomImg(this)"></div>
    {STRENGTH_CHART}
    {SOURCE_CHART}
  </div>
</section>

<section id="features">
  <h2>4. Feature Engineering (Bilingual)</h2>
  <p>Text features built in three language configurations for training and retrieval:</p>
  <table>
    <tr><th>Configuration</th><th>Source Columns</th><th>Avg Length</th><th>TF-IDF Terms</th></tr>
    <tr><td><strong>English</strong></td>
        <td>drug_name, generic, dosage_form, drug_class, indications, color, shape, salt, side_effects, contraindications</td>
        <td>{EN_AVG} chars</td><td>{EN_VOCAB}</td></tr>
    <tr><td><strong>Arabic</strong></td>
        <td>drug_name_ar, generic_ar, dosage_form_ar, drug_class_ar, indications_ar, color_ar, shape_ar, side_effects_ar, contraindications_ar</td>
        <td>{AR_AVG} chars</td><td>{AR_VOCAB}</td></tr>
    <tr><td><strong>Combined</strong></td>
        <td>All English + All Arabic fields concatenated</td>
        <td>{CB_AVG} chars</td><td>{CB_VOCAB}</td></tr>
  </table>
  <h3>Text Length Distributions</h3>
  <img src="data:image/png;base64,{IMG_TEXTLEN}" alt="Text Lengths" onclick="zoomImg(this)">
  <h3>Top Word Frequencies (English &amp; Arabic)</h3>
  <img src="data:image/png;base64,{IMG_WORDFREQ}" alt="Word Frequencies" onclick="zoomImg(this)">
</section>

<section id="results">
  <h2>5. Model Training Results</h2>
  <p>
    Target: <code>{LABEL_COL}</code> &nbsp;|&nbsp; Classes: <strong>{NCLASS}</strong>
    &nbsp;|&nbsp; Samples: <strong>{NSAMPLES}</strong>
    &nbsp;|&nbsp; Test split: <strong>{TSPLIT}%</strong>
    &nbsp;|&nbsp; CV folds: <strong>{NCVFOLDS}</strong>
  </p>
  <table class="sortable" id="results-table">
    <tr>
      <th>#</th><th>Model</th><th>Language</th>
      <th>Accuracy</th><th>F1</th><th>Precision</th><th>Recall</th>
      <th>Bal. Acc</th><th>MCC</th><th>CV Mean</th><th>Time</th>
    </tr>
    {RESULTS_ROWS}
  </table>
  <h3>Model Comparison Across Languages</h3>
  <img src="data:image/png;base64,{IMG_COMPARE}" alt="Model Comparison" onclick="zoomImg(this)">
  <h3>Accuracy by Language Configuration</h3>
  <img src="data:image/png;base64,{IMG_BYLANG}" alt="Accuracy by Language" onclick="zoomImg(this)">
  <h3>Results Heatmap (All Models)</h3>
  <img src="data:image/png;base64,{IMG_HEATMAP}" alt="Results Heatmap" onclick="zoomImg(this)">
  <h3>Confusion Matrix &mdash; Best Model: {BESTNAME}</h3>
  <img src="data:image/png;base64,{IMG_CM_BEST}" alt="Best CM" onclick="zoomImg(this)">
  <h3>Confusion Matrices &mdash; {BESTNAME} Across All Languages</h3>
  <img src="data:image/png;base64,{IMG_CM_LANGS}" alt="CM All Languages" onclick="zoomImg(this)">
  {FI_CHART}
  <h3>Per-Class Accuracy &mdash; {BESTNAME}</h3>
  <img src="data:image/png;base64,{IMG_PERCLASS}" alt="Per-Class Accuracy" onclick="zoomImg(this)">
</section>

<section id="perf-metrics">
  <h2>6. Performance Metrics — Comprehensive Evaluation</h2>
  <p>
    Six standard evaluation metrics computed on the held-out test set for every trained model.
    <strong>AUC-ROC</strong> measures class-separation ability (OvR, weighted).
    <strong>Log Loss</strong> penalises uncertain probability predictions (lower&nbsp;=&nbsp;better).
  </p>

  <h3>6.1 Metric Definitions</h3>
  <table style="max-width:820px">
    <tr><th>Metric</th><th>Formula / Notes</th><th>Range</th><th>Direction</th></tr>
    <tr><td><strong>Accuracy</strong></td><td>Correct predictions / Total predictions</td><td>0&ndash;1</td><td>Higher is better</td></tr>
    <tr><td><strong>Precision</strong></td><td>TP / (TP + FP) — weighted across classes</td><td>0&ndash;1</td><td>Higher is better</td></tr>
    <tr><td><strong>Recall (Sensitivity)</strong></td><td>TP / (TP + FN) — weighted across classes</td><td>0&ndash;1</td><td>Higher is better</td></tr>
    <tr><td><strong>F1-Score</strong></td><td>2 &times; (Precision &times; Recall) / (Precision + Recall)</td><td>0&ndash;1</td><td>Higher is better</td></tr>
    <tr><td><strong>AUC-ROC</strong></td><td>Area under ROC curve — One-vs-Rest, weighted</td><td>0&ndash;1</td><td>Higher is better (&gt;0.5 = better than random)</td></tr>
    <tr><td><strong>Log Loss</strong></td><td>&minus;&sum; y&sdot;log(p) — uncertainty of probability output</td><td>0&ndash;&infin;</td><td>Lower is better</td></tr>
  </table>

  <h3>6.2 Complete Performance Metrics Table</h3>
  <table class="sortable" id="perf-table">
    <tr>
      <th>#</th><th>Model</th><th>Language</th>
      <th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-Score</th>
      <th>AUC-ROC</th><th>Log Loss &darr;</th>
    </tr>
    {PERF_TABLE_ROWS}
  </table>

  <h3>6.3 All 5 Metrics Comparison (Accuracy / Precision / Recall / F1 / AUC-ROC)</h3>
  <img src="data:image/png;base64,{IMG_PERF_COMPARE}" alt="5-Metric Comparison" onclick="zoomImg(this)">

  <h3>6.4 AUC-ROC by Language Configuration</h3>
  <img src="data:image/png;base64,{IMG_AUC_ROC}" alt="AUC-ROC Chart" onclick="zoomImg(this)">

  <h3>6.5 Log Loss by Language Configuration  (lower = better)</h3>
  <img src="data:image/png;base64,{IMG_LOG_LOSS}" alt="Log Loss Chart" onclick="zoomImg(this)">

  <h3>6.6 Comprehensive 6-Metric Heatmap  (Log Loss shown as inverted score)</h3>
  <img src="data:image/png;base64,{IMG_PERF_HEATMAP6}" alt="6-Metric Heatmap" onclick="zoomImg(this)">

  <h3>6.7 Performance Radar Chart — Top 6 Models</h3>
  <img src="data:image/png;base64,{IMG_RADAR}" alt="Radar Chart" onclick="zoomImg(this)">
</section>

<section id="cv">
  <h2>7. Cross-Validation Results ({NCVFOLDS}-Fold Stratified K-Fold)</h2>
  <table>
    <tr><th>Model</th><th>Language</th><th>CV Mean</th><th>CV Std</th><th>Fold Scores</th></tr>
    {CV_TABLE}
  </table>
  <h3>Cross-Validation Score Distribution</h3>
  <img src="data:image/png;base64,{IMG_CV}" alt="CV Comparison" onclick="zoomImg(this)">
</section>

{BILSTM_SECTION}

<section id="reports">
  <h2>9. Detailed Classification Reports</h2>
  <p>Per-class precision, recall, F1-score, and support for each model. Click a model to expand.</p>
  {REPORTS_HTML}
</section>

<section id="summary">
  <h2>9. Training Summary</h2>
  <table>
    <tr><th>Item</th><th>Details</th></tr>
    <tr><td>Dataset Path</td><td><code>{DPATH}</code></td></tr>
    <tr><td>Records</td><td>{NREC} total, {NSAMPLES} used for training (min {MIN_S} samples/class)</td></tr>
    <tr><td>Languages</td><td>English, Arabic, Combined (EN+AR) &mdash; all three trained independently</td></tr>
    <tr><td>Classification Target</td><td>{LABEL_COL} ({NCLASS} classes)</td></tr>
    <tr><td>Test Split</td><td>{TSPLIT}% ({NTEST} samples)</td></tr>
    <tr><td>Cross-Validation</td><td>{NCVFOLDS}-Fold Stratified K-Fold</td></tr>
    <tr><td>ML Classifiers</td><td>{NMLMODELS} total (6 classifiers &times; 3 language configs)</td></tr>
    <tr><td>Best ML Model</td><td><strong>{BESTNAME}</strong> &mdash; {BESTACC} accuracy</td></tr>
    <tr><td>BiLSTM Model</td><td>{BILSTM_SUMMARY}</td></tr>
    <tr><td>TF-IDF Vectorizers</td><td>3 configs &mdash; EN: {EN_VOCAB} terms, AR: {AR_VOCAB} terms, Combined: {CB_VOCAB} terms</td></tr>
    <tr><td>Output Directory</td><td><code>{MODEL_DIR}</code></td></tr>
    <tr><td>Total Training Time</td><td>{TOTAL_TIME}s</td></tr>
    <tr><td>Report Generated</td><td>{DATE}</td></tr>
  </table>
</section>

</div>

<footer>
  CAPSULE AI &mdash; Bilingual Model Training Report &nbsp;|&nbsp;
  Generated {DATE} &nbsp;|&nbsp; Total time: {TOTAL_TIME}s
</footer>

<script>
function zoomImg(el){
  el.classList.add('zoomed');
  document.getElementById('overlay').classList.add('active');
}
function closeZoom(){
  document.querySelectorAll('img.zoomed').forEach(i=>i.classList.remove('zoomed'));
  document.getElementById('overlay').classList.remove('active');
}
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeZoom();});

(function(){
  const tbl=document.getElementById('results-table');
  if(!tbl)return;
  const ths=tbl.querySelectorAll('th');
  let sortCol=-1,sortAsc=true;
  ths.forEach((th,ci)=>{
    th.addEventListener('click',()=>{
      sortAsc=(ci===sortCol)?!sortAsc:true;
      sortCol=ci;
      ths.forEach(h=>{h.classList.remove('asc','desc');});
      th.classList.add(sortAsc?'asc':'desc');
      const tbody=tbl.tBodies[0];
      const rows=Array.from(tbody.rows);
      rows.sort((a,b)=>{
        const av=a.cells[ci]?a.cells[ci].textContent.trim():'';
        const bv=b.cells[ci]?b.cells[ci].textContent.trim():'';
        const an=parseFloat(av),bn=parseFloat(bv);
        if(!isNaN(an)&&!isNaN(bn))return sortAsc?an-bn:bn-an;
        return sortAsc?av.localeCompare(bv):bv.localeCompare(av);
      });
      rows.forEach(r=>tbody.appendChild(r));
    });
  });
})();
</script>
</body>
</html>"""

# Fill template
HTML = HTML.replace("{DATE}",           datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
HTML = HTML.replace("{NREC}",           f"{len(df):,}")
HTML = HTML.replace("{NCLASS}",         str(n_classes))
HTML = HTML.replace("{NMODELS}",        str(rank))
HTML = HTML.replace("{BESTACC}",        f"{best_acc*100:.1f}%")
HTML = HTML.replace("{ELAPSED}",        f"{total_elapsed:.0f}s")
HTML = HTML.replace("{NUNIQ}",          str(df["drug_name_en"].nunique()))
HTML = HTML.replace("{NFORM}",          str(df["dosage_form_en"].nunique()))
HTML = HTML.replace("{NMFR}",           str(df["manufacturer_en"].nunique()))
HTML = HTML.replace("{NCOL}",           str(df["color_en"].nunique()))
HTML = HTML.replace("{NSHAPE}",         str(df["shape_en"].nunique()))
HTML = HTML.replace("{NCOLS}",          str(len(df.columns)))
HTML = HTML.replace("{OVERVIEW_STATS}", overview_stats_html)
HTML = HTML.replace("{CLASS_TABLE}",    class_table_html)
HTML = HTML.replace("{IMG_CLASS}",      charts.get("drug_class_dist", ""))
HTML = HTML.replace("{IMG_IMBALANCE}",  charts.get("class_imbalance", ""))
HTML = HTML.replace("{IMG_FORM}",       charts.get("dosage_form_dist", ""))
HTML = HTML.replace("{IMG_MFR}",        charts.get("manufacturer_dist", ""))
HTML = HTML.replace("{IMG_COLOR}",      charts.get("color_dist", ""))
HTML = HTML.replace("{IMG_SHAPE}",      charts.get("shape_dist", ""))
HTML = HTML.replace("{IMG_RXOTC}",      charts.get("rx_otc_pie", ""))
HTML = HTML.replace("{IMG_ARCOV}",      charts.get("arabic_coverage", ""))
HTML = HTML.replace("{STRENGTH_CHART}", strength_chart_html)
HTML = HTML.replace("{SOURCE_CHART}",   source_chart_html)
HTML = HTML.replace("{EN_AVG}",         f"{en_lens.mean():.0f}")
HTML = HTML.replace("{AR_AVG}",         f"{ar_lens.mean():.0f}")
HTML = HTML.replace("{CB_AVG}",         f"{cb_lens.mean():.0f}")
HTML = HTML.replace("{EN_VOCAB}",       f"{len(tfidf_models['English'].vocabulary_):,}")
HTML = HTML.replace("{AR_VOCAB}",       f"{len(tfidf_models['Arabic'].vocabulary_):,}")
HTML = HTML.replace("{CB_VOCAB}",       f"{len(tfidf_models['Combined'].vocabulary_):,}")
HTML = HTML.replace("{IMG_TEXTLEN}",    charts.get("text_length_dist", ""))
HTML = HTML.replace("{IMG_WORDFREQ}",   charts.get("word_freq", ""))
HTML = HTML.replace("{LABEL_COL}",      LABEL_COL)
HTML = HTML.replace("{NSAMPLES}",       f"{len(ml_df):,}")
HTML = HTML.replace("{TSPLIT}",         f"{TEST_SIZE*100:.0f}")
HTML = HTML.replace("{NCVFOLDS}",       str(N_CV))
HTML = HTML.replace("{RESULTS_ROWS}",      results_rows_html)
HTML = HTML.replace("{PERF_TABLE_ROWS}",   perf_table_rows_html)
HTML = HTML.replace("{IMG_PERF_COMPARE}",  charts.get("perf_metrics_compare", ""))
HTML = HTML.replace("{IMG_AUC_ROC}",       charts.get("auc_roc_chart", ""))
HTML = HTML.replace("{IMG_LOG_LOSS}",      charts.get("log_loss_chart", ""))
HTML = HTML.replace("{IMG_PERF_HEATMAP6}", charts.get("perf_heatmap_6", ""))
HTML = HTML.replace("{IMG_RADAR}",         charts.get("radar_chart", ""))
HTML = HTML.replace("{IMG_COMPARE}",    charts.get("model_comparison", ""))
HTML = HTML.replace("{IMG_BYLANG}",     charts.get("accuracy_by_lang", ""))
HTML = HTML.replace("{IMG_HEATMAP}",    charts.get("results_heatmap", ""))
HTML = HTML.replace("{IMG_CM_BEST}",    charts.get("confusion_matrix", ""))
HTML = HTML.replace("{IMG_CM_LANGS}",   charts.get("cm_all_langs", ""))
HTML = HTML.replace("{FI_CHART}",       fi_chart_html)
HTML = HTML.replace("{IMG_PERCLASS}",   charts.get("per_class_acc", ""))
HTML = HTML.replace("{CV_TABLE}",       cv_table_html)
HTML = HTML.replace("{IMG_CV}",         charts.get("cv_comparison", ""))
HTML = HTML.replace("{BILSTM_SECTION}", bilstm_section_html)
HTML = HTML.replace("{REPORTS_HTML}",   reports_html)
HTML = HTML.replace("{DPATH}",          DATASET_PATH)
HTML = HTML.replace("{MIN_S}",          str(MIN_SAMPLES))
HTML = HTML.replace("{NTEST}",          str(int(len(ml_df) * TEST_SIZE)))
HTML = HTML.replace("{NMLMODELS}",      str(len(all_results)))
HTML = HTML.replace("{BESTNAME}",       f"{best_key[1]} ({best_key[0]})" if best_key else "")
HTML = HTML.replace("{BILSTM_SUMMARY}", bilstm_summary_cell)
HTML = HTML.replace("{MODEL_DIR}",      MODEL_DIR)
HTML = HTML.replace("{TOTAL_TIME}",     f"{total_elapsed:.1f}")

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(HTML)

report_kb = os.path.getsize(REPORT_PATH) / 1024
print(f"  [SAVED] {REPORT_PATH}")
print(f"  Report size: {report_kb:.1f} KB  ({len(charts)} embedded charts)")

# ===========================================================================
#  SECTION 12 — FINAL SUMMARY
# ===========================================================================
print(f"\n{SEP}")
print("  TRAINING COMPLETE")
print(SEP)
print(f"  Output folder : {MODEL_DIR}")
print()
print(f"  {'File':<44} {'Size':>12}")
print("  " + SSEP[:58])
for fname in sorted(os.listdir(MODEL_DIR)):
    fpath = os.path.join(MODEL_DIR, fname)
    if os.path.isfile(fpath):
        sz = os.path.getsize(fpath)
        sz_str = f"{sz/(1024*1024):.1f} MB" if sz >= 1024*1024 else f"{sz/1024:.1f} KB"
        print(f"  {fname:<44} {sz_str:>12}")

print(f"\n  HTML Report   : {REPORT_PATH}  ({report_kb:.1f} KB)")
print(f"  Best ML Model : {best_key[1]} ({best_key[0]}) -> {best_acc*100:.1f}%")
if bilstm_results:
    print(f"  BiLSTM        : {bilstm_results['accuracy']*100:.1f}% accuracy  "
          f"({total_params:,} params, {EPOCHS} epochs, {bilstm_time:.0f}s)")
print(f"  Charts        : {len(charts)} visualizations embedded in HTML")
print(f"  Total Time    : {total_elapsed:.1f}s")
print()
print(f"  Restart the Flask app to load the newly trained models.")
print(SEP)
