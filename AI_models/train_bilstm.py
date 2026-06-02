"""
CAPSULE — BiLSTM Deep Learning Training (Standalone)
Trains BiLSTM on DataSet/capsule_dataset.csv (bilingual Arabic + English).
Saves all artefacts to AI_models/trained_models/.

Run:
    python AI_models/train_bilstm.py

NOTE: train_models.py already includes BiLSTM as part of the full pipeline.
      Use this script only when you want to retrain BiLSTM independently.
"""

import os, json, warnings, base64, io, time, sys
from datetime import datetime
from collections import Counter

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, matthews_corrcoef,
    classification_report, confusion_matrix,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODEL_DIR   = os.path.join(BASE_DIR, "trained_models")

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

PALETTE = sns.color_palette("Set2", 8)
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})

SEP  = "=" * 68
SSEP = "-" * 68

def safe(val):
    return "" if pd.isna(val) else str(val).strip()

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64

start_time = time.time()
charts     = {}

# ===========================================================================
#  HEADER
# ===========================================================================
print(f"\n{SEP}")
print("  CAPSULE AI — BiLSTM Standalone Training")
print(f"  Date   : {datetime.now():%Y-%m-%d %H:%M:%S}")
print(f"  Python : {sys.version.split()[0]}")
print(f"  PyTorch: {torch.__version__}")
print(f"  Device : {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
print(SEP)

# ===========================================================================
#  STEP 1 — LOAD DATASET
# ===========================================================================
print(f"\n[1/8] Loading dataset ...")
df = pd.read_csv(DATASET_PATH)
print(f"  Path    : {DATASET_PATH}")
print(f"  Records : {len(df):,}")
print(f"  Columns : {len(df.columns)}")

EN_COLS = [c for c in df.columns if c.endswith("_en")]
AR_COLS = [c for c in df.columns if c.endswith("_ar")]
print(f"  EN cols : {len(EN_COLS)}   AR cols : {len(AR_COLS)}")

# ===========================================================================
#  STEP 2 — DATASET ANALYSIS
# ===========================================================================
print(f"\n[2/8] Dataset analysis ...")
print(f"\n  {'Metric':<38} {'Value':>8}")
print("  " + SSEP[:48])
stats = [
    ("Total records",             len(df)),
    ("Unique drug names (EN)",    df["drug_name_en"].nunique()),
    ("Unique drug classes (EN)",  df["drug_class_en"].nunique()),
    ("Unique manufacturers (EN)", df["manufacturer_en"].nunique()),
    ("RX drugs",                  (df["rx_otc_status_en"] == "Prescription").sum()
                                  if "rx_otc_status_en" in df.columns else 0),
    ("OTC drugs",                 (df["rx_otc_status_en"] == "OTC").sum()
                                  if "rx_otc_status_en" in df.columns else 0),
    ("Missing values (total)",    int(df.isna().sum().sum())),
]
for name, val in stats:
    print(f"  {name:<38} {val:>8,}")

# Arabic coverage
print(f"\n  Arabic Column Coverage:")
print(f"  {'Column':<32} {'Filled':>6} {'%':>7}")
print("  " + SSEP[:43])
for col in AR_COLS:
    filled = df[col].notna().sum()
    print(f"  {col:<32} {filled:>6} {filled/len(df)*100:>6.1f}%")

# ===========================================================================
#  STEP 3 — BUILD BILINGUAL TEXT FEATURES
# ===========================================================================
print(f"\n[3/8] Building bilingual text features ...")

def build_text(row):
    parts_en = [
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
    parts_ar = [
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
    en = " ".join(p for p in parts_en if p).lower()
    ar = " ".join(p for p in parts_ar if p)
    return en + " " + ar

df["text_en"]       = df.apply(lambda r: " ".join(
    p for p in [
        safe(r.get("drug_name_en")),
        f"generic {safe(r.get('generic_name_en'))}",
        f"form {safe(r.get('dosage_form_en'))}",
        f"class {safe(r.get('drug_class_en'))}",
        f"indication {safe(r.get('indications_en'))}",
        safe(r.get("salt_composition")),
        safe(r.get("side_effects_en")),
    ] if p).lower(), axis=1)
df["text_ar"]       = df.apply(lambda r: " ".join(
    p for p in [
        safe(r.get("drug_name_ar")), safe(r.get("generic_name_ar")),
        safe(r.get("dosage_form_ar")), safe(r.get("drug_class_ar")),
        safe(r.get("indications_ar")), safe(r.get("side_effects_ar")),
    ] if p), axis=1)
df["text_combined"] = df.apply(build_text, axis=1)

en_lens = df["text_en"].str.len()
ar_lens = df["text_ar"].str.len()
cb_lens = df["text_combined"].str.len()
print(f"  {'Feature':<12} {'Mean':>8} {'Min':>6} {'Max':>6} {'Median':>8}")
print("  " + SSEP[:43])
for name, lens in [("English", en_lens), ("Arabic", ar_lens), ("Combined", cb_lens)]:
    print(f"  {name:<12} {lens.mean():>8.0f} {lens.min():>6} {lens.max():>6} {lens.median():>8.0f}")

# ===========================================================================
#  STEP 4 — PREPARE LABELS
# ===========================================================================
print(f"\n[4/8] Preparing classification labels ...")

LABEL_COL   = "drug_class_en"
MIN_SAMPLES = 2

label_vc   = df[LABEL_COL].value_counts()
valid_lbls = label_vc[label_vc >= MIN_SAMPLES].index.tolist()
ml_df      = df[df[LABEL_COL].isin(valid_lbls)].copy()
ml_df      = ml_df[ml_df["text_combined"].str.strip() != ""].copy()

le_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
if os.path.exists(le_path):
    le = joblib.load(le_path)
    ml_df = ml_df[ml_df[LABEL_COL].isin(le.classes_)].copy()
    y     = le.transform(ml_df[LABEL_COL])
    class_names = list(le.classes_)
    print(f"  Label encoder loaded from {le_path}")
else:
    le          = LabelEncoder()
    y           = le.fit_transform(ml_df[LABEL_COL])
    class_names = list(le.classes_)
    joblib.dump(le, le_path)
    print(f"  Label encoder created and saved to {le_path}")

n_classes = len(class_names)
min_test  = max(n_classes, int(len(ml_df) * 0.15))
TEST_SIZE = min(0.40, max(0.20, min_test / len(ml_df)))

print(f"  Samples    : {len(ml_df):,}")
print(f"  Classes    : {n_classes}")
print(f"  Test split : {TEST_SIZE*100:.0f}%")
print(f"\n  {'#':>4} {'Class Name':<48} {'Count':>6}")
print("  " + SSEP[:61])
for i, cn in enumerate(class_names):
    cnt = int((y == i).sum())
    bar = "|" * min(cnt, 35)
    print(f"  {i+1:>4}. {cn:<48} {cnt:>6}  {bar}")

# ===========================================================================
#  STEP 5 — BUILD VOCABULARY
# ===========================================================================
print(f"\n[5/8] Building vocabulary ...")

MAX_VOCAB   = 8000
MAX_LEN     = 128
all_tokens  = [w for t in ml_df["text_combined"].tolist() for w in t.lower().split()]
vocab_cnt   = Counter(all_tokens)
vocab_words = ["<PAD>", "<UNK>"] + [w for w, _ in vocab_cnt.most_common(MAX_VOCAB - 2)]
w2i         = {w: i for i, w in enumerate(vocab_words)}

print(f"  Corpus unique tokens : {len(vocab_cnt):,}")
print(f"  Vocabulary size      : {len(w2i):,} (max {MAX_VOCAB})")
print(f"  Max sequence length  : {MAX_LEN}")

def encode(text, max_len=MAX_LEN):
    ids = [w2i.get(w, 1) for w in text.lower().split()][:max_len]
    return ids + [0] * (max_len - len(ids))

# ===========================================================================
#  STEP 6 — BUILD BiLSTM MODEL
# ===========================================================================
print(f"\n[6/8] Building BiLSTM model ...")

EMBED_DIM = 128
HIDDEN    = 256
N_LAYERS  = 2
DROPOUT   = 0.3
BATCH     = 32
EPOCHS    = 25
LR        = 1e-3

class DrugDataset(Dataset):
    def __init__(self, texts, labels):
        self.X = [torch.tensor(encode(t), dtype=torch.long) for t in texts]
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
        emb  = self.embedding(x)
        out, _ = self.lstm(emb)
        attn = torch.softmax(self.attention(out), dim=1)
        return self.fc((out * attn).sum(dim=1))

tr_texts, te_texts, tr_labels, te_labels = train_test_split(
    ml_df["text_combined"].fillna("").tolist(), y.tolist(),
    test_size=TEST_SIZE, random_state=42, stratify=y)

train_dl = DataLoader(DrugDataset(tr_texts, tr_labels), batch_size=BATCH, shuffle=True)
test_dl  = DataLoader(DrugDataset(te_texts, te_labels), batch_size=BATCH)

device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model        = BiLSTMClassifier(len(w2i), EMBED_DIM, HIDDEN, n_classes).to(device)
total_params = sum(p.numel() for p in model.parameters())
train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"  Architecture  : BiLSTM x{N_LAYERS} layers  |  Hidden: {HIDDEN}  |  Embed: {EMBED_DIM}")
print(f"  Bidirectional : Yes  (effective hidden dim = {HIDDEN*2})")
print(f"  Attention     : Single-head soft attention")
print(f"  Classifier    : FC({HIDDEN*2}->{HIDDEN}->ReLU->{n_classes})")
print(f"  Dropout       : {DROPOUT}")
print(f"  Total params  : {total_params:,}")
print(f"  Trainable     : {train_params:,}")
print(f"  Device        : {device}")

# ===========================================================================
#  STEP 7 — TRAIN
# ===========================================================================
print(f"\n[7/8] Training BiLSTM ({EPOCHS} epochs) ...")

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=7, gamma=0.5)

best_val_acc = 0.0
best_state   = None
history      = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

print(f"\n  {'Ep':>4} {'Tr Loss':>9} {'Tr Acc':>8} {'Va Loss':>9} {'Va Acc':>8} {'LR':>10}")
print("  " + SSEP[:54])

bilstm_t0 = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    tr_c = tr_n = 0
    tr_loss_s = 0.0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tr_c     += (logits.argmax(1) == yb).sum().item()
        tr_n     += xb.size(0)
        tr_loss_s += loss.item() * xb.size(0)
    scheduler.step()

    model.eval()
    va_c = va_n = 0
    va_loss_s = 0.0
    with torch.no_grad():
        for xb, yb in test_dl:
            xb, yb = xb.to(device), yb.to(device)
            logits  = model(xb)
            va_loss_s += criterion(logits, yb).item() * xb.size(0)
            va_c  += (logits.argmax(1) == yb).sum().item()
            va_n  += xb.size(0)

    tr_acc  = tr_c / tr_n * 100
    va_acc  = va_c / va_n * 100
    tr_loss = tr_loss_s / tr_n
    va_loss = va_loss_s / va_n
    cur_lr  = optimizer.param_groups[0]["lr"]

    history["train_acc"].append(tr_acc)
    history["val_acc"].append(va_acc)
    history["train_loss"].append(tr_loss)
    history["val_loss"].append(va_loss)

    marker = ""
    if va_acc > best_val_acc:
        best_val_acc = va_acc
        best_state   = {k: v.clone() for k, v in model.state_dict().items()}
        marker = " *"
    print(f"  {epoch:>4} {tr_loss:>9.4f} {tr_acc:>7.2f}% {va_loss:>9.4f} "
          f"{va_acc:>7.2f}%{marker} {cur_lr:>10.6f}")

bilstm_time = time.time() - bilstm_t0
print(f"\n  Best Val Accuracy : {best_val_acc:.2f}%")
print(f"  Training Time     : {bilstm_time:.1f}s")

if best_state:
    model.load_state_dict(best_state)

# --- Final evaluation ---
model.eval()
preds_all, true_all = [], []
with torch.no_grad():
    for xb, yb in test_dl:
        preds_all.extend(model(xb.to(device)).argmax(1).cpu().tolist())
        true_all.extend(yb.tolist())

acc  = accuracy_score(true_all, preds_all)
bac  = balanced_accuracy_score(true_all, preds_all)
f1   = f1_score(true_all, preds_all, average="weighted", zero_division=0)
prec = precision_score(true_all, preds_all, average="weighted", zero_division=0)
rec  = recall_score(true_all, preds_all, average="weighted", zero_division=0)
mcc  = matthews_corrcoef(true_all, preds_all)
rep  = classification_report(true_all, preds_all, target_names=class_names, zero_division=0)
cm   = confusion_matrix(true_all, preds_all)

print(f"\n  Final Test Results:")
print(f"  {'Metric':<25} {'Value':>10}")
print("  " + SSEP[:38])
print(f"  {'Accuracy':<25} {acc*100:>9.2f}%")
print(f"  {'Balanced Accuracy':<25} {bac*100:>9.2f}%")
print(f"  {'F1 Score (Weighted)':<25} {f1*100:>9.2f}%")
print(f"  {'Precision':<25} {prec*100:>9.2f}%")
print(f"  {'Recall':<25} {rec*100:>9.2f}%")
print(f"  {'MCC':<25} {mcc:>10.4f}")
print(f"\n  Classification Report:")
print(rep)

# ===========================================================================
#  STEP 8 — SAVE ARTEFACTS + REPORT
# ===========================================================================
print(f"\n[8/8] Saving artefacts ...")

# Models
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "bilstm_best.pt"))
joblib.dump(w2i, os.path.join(MODEL_DIR, "vocab.pkl"))
print(f"  [SAVED] {os.path.join(MODEL_DIR, 'bilstm_best.pt')}")
print(f"  [SAVED] {os.path.join(MODEL_DIR, 'vocab.pkl')}")

# Update ensemble config
config_path = os.path.join(MODEL_DIR, "ensemble_config.json")
cfg = {}
if os.path.exists(config_path):
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
cfg.update({
    "vocab_size":       len(w2i),
    "num_text_classes": n_classes,
    "max_seq_len":      MAX_LEN,
    "embed_dim":        EMBED_DIM,
    "hidden_dim":       HIDDEN,
    "n_layers":         N_LAYERS,
    "dropout":          DROPOUT,
    "tfidf_weight":     0.35,
    "bilstm_weight":    0.35,
    "ml_weight":        0.30,
    "bilstm_accuracy":  round(acc, 4),
    "bilstm_retrained": datetime.now().isoformat(),
})
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)
print(f"  [UPDATED] {config_path}")

# --- Training curves chart ---
ep_x = range(1, EPOCHS + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(ep_x, history["train_loss"], "o-", label="Train Loss", color=PALETTE[0], markersize=4)
ax1.plot(ep_x, history["val_loss"],   "s-", label="Val Loss",   color=PALETTE[1], markersize=4)
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.set_title("Loss Curve", fontsize=11, fontweight="bold")
ax1.legend()
ax2.plot(ep_x, history["train_acc"], "o-", label="Train Acc", color=PALETTE[2], markersize=4)
ax2.plot(ep_x, history["val_acc"],   "s-", label="Val Acc",   color=PALETTE[3], markersize=4)
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy %")
ax2.set_title("Accuracy Curve", fontsize=11, fontweight="bold")
ax2.legend()
fig.suptitle("BiLSTM Training Progress (Combined EN+AR)", fontsize=13, fontweight="bold", y=1.03)
fig.tight_layout()
curves_path = os.path.join(MODEL_DIR, "bilstm_training_curves.png")
fig.savefig(curves_path, dpi=150, bbox_inches="tight")
charts["curves"] = fig_to_b64(fig)
print(f"  [SAVED] {curves_path}")

# --- Confusion matrix chart ---
fig, ax = plt.subplots(figsize=(max(7, n_classes * 0.65), max(6, n_classes * 0.6)))
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
            xticklabels=class_names, yticklabels=class_names, ax=ax,
            linewidths=0.5, linecolor="white")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("BiLSTM Confusion Matrix (Combined EN+AR)", fontsize=12, fontweight="bold")
plt.xticks(rotation=40, ha="right", fontsize=7)
plt.yticks(fontsize=7)
fig.tight_layout()
cm_path = os.path.join(MODEL_DIR, "bilstm_confusion_matrix.png")
fig.savefig(cm_path, dpi=150, bbox_inches="tight")
charts["cm"] = fig_to_b64(fig)
print(f"  [SAVED] {cm_path}")

# --- Per-class accuracy chart ---
row_sums  = np.where(cm.sum(axis=1) == 0, 1, cm.sum(axis=1))
pc_acc    = cm.diagonal() / row_sums
pc_colors = [PALETTE[0] if v >= 0.8 else (PALETTE[1] if v >= 0.5 else PALETTE[3]) for v in pc_acc]
fig, ax = plt.subplots(figsize=(10, max(5, n_classes * 0.45)))
ax.barh(class_names[::-1], (pc_acc * 100)[::-1], color=pc_colors[::-1], edgecolor="white")
ax.set_xlabel("Accuracy %"); ax.set_xlim(0, 115)
ax.axvline(80, color="red", linestyle="--", linewidth=0.8, label="80% threshold")
ax.legend(fontsize=8)
ax.set_title("Per-Class Accuracy — BiLSTM", fontsize=12, fontweight="bold")
for i, v in enumerate(pc_acc[::-1]):
    ax.text(v * 100 + 1, i, f"{v*100:.0f}%", va="center", fontsize=8)
fig.tight_layout()
charts["per_class"] = fig_to_b64(fig)
pc_path = os.path.join(MODEL_DIR, "bilstm_per_class_accuracy.png")
fig.savefig(pc_path, dpi=150, bbox_inches="tight")
print(f"  [SAVED] {pc_path}")

# --- HTML report ---
total_elapsed = time.time() - start_time

def mc(v, is_pct=True):
    val = v * 100 if is_pct else v
    if is_pct:
        cls = "cell-hi" if val >= 80 else ("cell-mid" if val >= 60 else "cell-lo")
        return f'<td class="{cls}">{val:.2f}%</td>'
    cls = "cell-hi" if val >= 0.6 else ("cell-mid" if val >= 0.3 else "cell-lo")
    return f'<td class="{cls}">{val:.3f}</td>'

report_html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CAPSULE BiLSTM Training Report</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI',sans-serif;background:#f0f2f5;color:#333;line-height:1.6}}
header{{background:linear-gradient(135deg,#0d1b2a,#1a237e);color:#fff;padding:2rem;text-align:center}}
header h1{{font-size:2rem;margin-bottom:.3rem}}
header p{{opacity:.8;font-size:.9rem}}
.hs{{display:flex;justify-content:center;gap:2rem;margin-top:1.2rem;flex-wrap:wrap}}
.hs-item .val{{font-size:1.8rem;font-weight:700}}.hs-item .lbl{{font-size:.75rem;opacity:.7}}
nav{{background:#1a237e;padding:.5rem 2rem;text-align:center;position:sticky;top:0;z-index:100}}
nav a{{color:#e8eaf6;text-decoration:none;margin:0 .8rem;font-size:.85rem;padding:.3rem .6rem;border-radius:4px}}
nav a:hover{{background:rgba(255,255,255,.2)}}
.container{{max-width:1200px;margin:0 auto;padding:1.5rem}}
section{{background:#fff;border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,.06);padding:2rem;margin-bottom:1.5rem}}
h2{{color:#1a237e;border-bottom:3px solid #3949ab;padding-bottom:.5rem;margin-bottom:1.2rem;font-size:1.3rem}}
h3{{color:#283593;margin:1.2rem 0 .6rem;font-size:1.05rem}}
table{{width:100%;border-collapse:collapse;font-size:.87rem;margin:.8rem 0}}
th{{background:#e8eaf6;color:#1a237e;padding:.6rem .8rem;text-align:left;font-weight:600}}
td{{padding:.5rem .8rem;border-bottom:1px solid #eee}}
tr:hover td{{background:#f8f9ff}}
.cell-hi{{color:#1b5e20;font-weight:600;background:#e8f5e9}}
.cell-mid{{color:#e65100;font-weight:600;background:#fff3e0}}
.cell-lo{{color:#b71c1c;font-weight:600;background:#ffebee}}
.stats-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:.8rem;margin:.8rem 0}}
.stat-card{{background:#f8f9fa;border-radius:10px;padding:1rem;text-align:center;border:1px solid #e0e0e0}}
.stat-card.accent{{background:linear-gradient(135deg,#e8eaf6,#c5cae9);border-color:#9fa8da}}
.stat-val{{font-size:1.5rem;font-weight:700;color:#1a237e}}
.stat-lbl{{font-size:.76rem;color:#666;margin-top:.3rem}}
img{{max-width:100%;height:auto;border-radius:8px;box-shadow:0 1px 6px rgba(0,0,0,.08);margin:.5rem 0}}
pre{{background:#f5f5f5;padding:1rem;border-radius:8px;overflow-x:auto;font-size:.78rem;line-height:1.5}}
.chart-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:1.2rem}}
footer{{text-align:center;padding:1.2rem;color:#888;font-size:.8rem;background:#fff;margin-top:1rem}}
</style></head><body>
<header>
  <h1>CAPSULE AI &mdash; BiLSTM Training Report</h1>
  <p>Bidirectional LSTM with Attention &nbsp;|&nbsp; Bilingual EN+AR</p>
  <p>{datetime.now():%Y-%m-%d %H:%M:%S}</p>
  <div class="hs">
    <div class="hs-item"><div class="val">{len(df):,}</div><div class="lbl">Records</div></div>
    <div class="hs-item"><div class="val">{n_classes}</div><div class="lbl">Classes</div></div>
    <div class="hs-item"><div class="val">{acc*100:.1f}%</div><div class="lbl">Test Accuracy</div></div>
    <div class="hs-item"><div class="val">{f1*100:.1f}%</div><div class="lbl">F1 Score</div></div>
    <div class="hs-item"><div class="val">{best_val_acc:.1f}%</div><div class="lbl">Best Val Acc</div></div>
    <div class="hs-item"><div class="val">{total_params:,}</div><div class="lbl">Parameters</div></div>
    <div class="hs-item"><div class="val">{bilstm_time:.0f}s</div><div class="lbl">Train Time</div></div>
  </div>
</header>
<nav>
  <a href="#overview">Dataset</a>
  <a href="#arch">Architecture</a>
  <a href="#metrics">Metrics</a>
  <a href="#curves">Training</a>
  <a href="#confusion">Confusion</a>
  <a href="#perclass">Per-Class</a>
  <a href="#report">Report</a>
</nav>
<div class="container">

<section id="overview">
  <h2>1. Dataset Overview</h2>
  <div class="stats-grid">
    <div class="stat-card accent"><div class="stat-val">{len(df):,}</div><div class="stat-lbl">Total Records</div></div>
    <div class="stat-card"><div class="stat-val">{len(ml_df):,}</div><div class="stat-lbl">Training Samples</div></div>
    <div class="stat-card"><div class="stat-val">{n_classes}</div><div class="stat-lbl">Drug Classes</div></div>
    <div class="stat-card"><div class="stat-val">{len(EN_COLS)}</div><div class="stat-lbl">EN Columns</div></div>
    <div class="stat-card"><div class="stat-val">{len(AR_COLS)}</div><div class="stat-lbl">AR Columns</div></div>
    <div class="stat-card"><div class="stat-val">{TEST_SIZE*100:.0f}%</div><div class="stat-lbl">Test Split</div></div>
  </div>
</section>

<section id="arch">
  <h2>2. Model Architecture</h2>
  <table>
    <tr><th>Component</th><th>Details</th></tr>
    <tr><td><strong>Model Type</strong></td><td>Bidirectional LSTM with Soft Attention</td></tr>
    <tr><td><strong>Embedding</strong></td><td>{EMBED_DIM} dimensions &nbsp;|&nbsp; Vocabulary: {len(w2i):,} tokens</td></tr>
    <tr><td><strong>LSTM</strong></td><td>{HIDDEN} hidden units &times; {N_LAYERS} layers (bidirectional = {HIDDEN*2} effective)</td></tr>
    <tr><td><strong>Attention</strong></td><td>Single-head soft attention over all LSTM time steps</td></tr>
    <tr><td><strong>Classifier</strong></td><td>FC({HIDDEN*2}) &rarr; Dropout({DROPOUT}) &rarr; FC({HIDDEN}) &rarr; ReLU &rarr; Dropout &rarr; FC({n_classes})</td></tr>
    <tr><td><strong>Total Parameters</strong></td><td>{total_params:,}</td></tr>
    <tr><td><strong>Trainable Parameters</strong></td><td>{train_params:,}</td></tr>
    <tr><td><strong>Optimizer</strong></td><td>Adam &nbsp;|&nbsp; LR={LR} &nbsp;|&nbsp; Weight decay=1e-4</td></tr>
    <tr><td><strong>Scheduler</strong></td><td>StepLR (step=7, gamma=0.5) &rarr; LR halved every 7 epochs</td></tr>
    <tr><td><strong>Gradient Clipping</strong></td><td>Max norm = 1.0</td></tr>
    <tr><td><strong>Loss Function</strong></td><td>Cross Entropy Loss</td></tr>
    <tr><td><strong>Batch Size</strong></td><td>{BATCH}</td></tr>
    <tr><td><strong>Epochs</strong></td><td>{EPOCHS}</td></tr>
    <tr><td><strong>Device</strong></td><td>{device}</td></tr>
  </table>
</section>

<section id="metrics">
  <h2>3. Test Set Metrics</h2>
  <div class="stats-grid">
    <div class="stat-card accent"><div class="stat-val">{acc*100:.2f}%</div><div class="stat-lbl">Accuracy</div></div>
    <div class="stat-card"><div class="stat-val">{bac*100:.2f}%</div><div class="stat-lbl">Balanced Acc</div></div>
    <div class="stat-card"><div class="stat-val">{f1*100:.2f}%</div><div class="stat-lbl">F1 Score</div></div>
    <div class="stat-card"><div class="stat-val">{prec*100:.2f}%</div><div class="stat-lbl">Precision</div></div>
    <div class="stat-card"><div class="stat-val">{rec*100:.2f}%</div><div class="stat-lbl">Recall</div></div>
    <div class="stat-card"><div class="stat-val">{mcc:.4f}</div><div class="stat-lbl">MCC</div></div>
    <div class="stat-card"><div class="stat-val">{best_val_acc:.2f}%</div><div class="stat-lbl">Best Val Acc</div></div>
    <div class="stat-card"><div class="stat-val">{bilstm_time:.1f}s</div><div class="stat-lbl">Train Time</div></div>
  </div>
  <h3>Summary Table</h3>
  <table><tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Test Accuracy</td>{mc(acc)}</tr>
    <tr><td>Balanced Accuracy</td>{mc(bac)}</tr>
    <tr><td>F1 Score (Weighted)</td>{mc(f1)}</tr>
    <tr><td>Precision (Weighted)</td>{mc(prec)}</tr>
    <tr><td>Recall (Weighted)</td>{mc(rec)}</tr>
    <tr><td>Matthews Corr. Coef.</td>{mc(mcc, is_pct=False)}</tr>
    <tr><td>Best Validation Accuracy</td>{mc(best_val_acc/100)}</tr>
  </table>
</section>

<section id="curves">
  <h2>4. Training Progress</h2>
  <img src="data:image/png;base64,{charts['curves']}" alt="Training Curves">
</section>

<section id="confusion">
  <h2>5. Confusion Matrix</h2>
  <img src="data:image/png;base64,{charts['cm']}" alt="Confusion Matrix">
</section>

<section id="perclass">
  <h2>6. Per-Class Accuracy</h2>
  <img src="data:image/png;base64,{charts['per_class']}" alt="Per-Class Accuracy">
</section>

<section id="report">
  <h2>7. Classification Report</h2>
  <pre>{rep}</pre>
</section>

</div>
<footer>CAPSULE AI &mdash; BiLSTM Training Report &nbsp;|&nbsp; {datetime.now():%Y-%m-%d %H:%M:%S} &nbsp;|&nbsp; Training time: {total_elapsed:.1f}s</footer>
</body></html>"""

report_path = os.path.join(MODEL_DIR, "bilstm_training_report.html")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_html)
print(f"  [SAVED] {report_path}  ({os.path.getsize(report_path)/1024:.0f} KB)")

# ===========================================================================
#  FINAL SUMMARY
# ===========================================================================
total_elapsed = time.time() - start_time
print(f"\n{SEP}")
print("  BiLSTM TRAINING COMPLETE")
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

print(f"\n  Test Accuracy  : {acc*100:.2f}%")
print(f"  Bal. Accuracy  : {bac*100:.2f}%")
print(f"  F1 Score       : {f1*100:.2f}%")
print(f"  MCC            : {mcc:.4f}")
print(f"  Total Time     : {total_elapsed:.1f}s")
print()
print(f"  Restart the Flask app to activate the updated BiLSTM model.")
print(SEP)
