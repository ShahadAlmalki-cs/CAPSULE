"""
CAPSULE AI Model
Integrates TF-IDF retrieval, ML classifier, and BiLSTM deep learning for
medication identification from natural language descriptions.
Supports the Drug Vision dataset (10 Philippine OTC drugs) out of the box,
with no trained model files required.

Dataset Note (Note 3): The KSA medication dataset contains 210 individual
medications organized into 42 therapeutic drug classes (num_text_classes=42
in ensemble_config.json). The BiLSTM and ML classifiers predict at the CLASS
level (42 outputs), while TF-IDF retrieval and filter matching operate at the
individual medication level (210 entries). This distinction is critical:
- 210 = total unique medications in the database
- 42  = therapeutic class labels used for classification model training
"""

import os
import pickle
import json
import traceback
import glob
import joblib

import numpy as np
import pandas as pd

from nlp_processor import MedicationNLPProcessor

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_models')
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Embedded OTC Drug Knowledge Base  (Drug Vision dataset — 10 drugs)
# Used as a fallback when trained_models/ksa_drug_database_clean.csv is absent.
# ---------------------------------------------------------------------------

OTC_DRUG_KNOWLEDGE = {
    "Alaxan": {
        "generic_name": "Ibuprofen + Paracetamol",
        "drug_class":   "Pain Relief / NSAID",
        "dosage_form":  "tablet",
        "color":        "red white",
        "shape":        "oval oblong",
        "indication":   "pain headache muscle pain arthritis fever toothache back pain",
        "description": (
            "Alaxan red white oval tablet ibuprofen paracetamol dual-action relief "
            "mild moderate pain headache toothache muscle pain back pain arthritis fever "
            "NSAID analgesic antipyretic combination"
        ),
    },
    "Bactidol": {
        "generic_name": "Hexetidine",
        "drug_class":   "Antiseptic / Oral Hygiene",
        "dosage_form":  "mouthwash liquid solution",
        "color":        "clear blue transparent",
        "shape":        "liquid solution bottle",
        "indication":   "sore throat mouth infection antiseptic gargle oral hygiene tonsillitis",
        "description": (
            "Bactidol clear blue antiseptic mouthwash solution hexetidine "
            "sore throat mouth throat infection oral hygiene tonsillitis "
            "antiseptic gargle antibacterial oral care liquid"
        ),
    },
    "Bioflu": {
        "generic_name": "Phenylephrine + Chlorphenamine + Paracetamol",
        "drug_class":   "Cold and Flu Remedy",
        "dosage_form":  "tablet",
        "color":        "yellow",
        "shape":        "oval oblong",
        "indication":   "flu cold fever runny nose nasal congestion sneezing chills body aches",
        "description": (
            "Bioflu yellow oval tablet phenylephrine chlorphenamine paracetamol "
            "flu symptoms fever runny nose nasal congestion sneezing body chills "
            "decongestant antihistamine antipyretic cold flu remedy"
        ),
    },
    "Biogesic": {
        "generic_name": "Paracetamol",
        "drug_class":   "Pain Relief / Antipyretic",
        "dosage_form":  "tablet",
        "color":        "white",
        "shape":        "round circular",
        "indication":   "fever headache body pain mild moderate pain paracetamol antipyretic",
        "description": (
            "Biogesic white round tablet paracetamol 500mg fever headache body pain "
            "mild moderate pain relief reduction fever safe common analgesic antipyretic"
        ),
    },
    "DayZinc": {
        "generic_name": "Zinc Sulfate",
        "drug_class":   "Vitamin / Mineral Supplement",
        "dosage_form":  "capsule",
        "color":        "orange",
        "shape":        "capsule oval",
        "indication":   "zinc supplement immune support vitamin mineral deficiency immunity boost daily",
        "description": (
            "DayZinc orange capsule zinc sulfate daily supplement "
            "immune function zinc deficiency immunity boost mineral vitamin "
            "overall health orange capsule supplement"
        ),
    },
    "Decolgen": {
        "generic_name": "Phenylpropanolamine + Chlorphenamine",
        "drug_class":   "Cold / Decongestant / Antihistamine",
        "dosage_form":  "tablet",
        "color":        "white",
        "shape":        "round circular",
        "indication":   "colds nasal congestion runny nose decongestant antihistamine sneezing allergy rhinitis",
        "description": (
            "Decolgen white round tablet phenylpropanolamine chlorphenamine "
            "colds nasal congestion runny nose sneezing allergic rhinitis "
            "decongestant antihistamine combination cold allergy"
        ),
    },
    "Fish Oil": {
        "generic_name": "Omega-3 Fatty Acids (EPA + DHA)",
        "drug_class":   "Nutritional / Omega-3 Supplement",
        "dosage_form":  "softgel capsule",
        "color":        "yellow gold amber",
        "shape":        "oval softgel",
        "indication":   "omega-3 heart health supplement cholesterol triglycerides cardiovascular brain fish",
        "description": (
            "Fish Oil yellow gold oval softgel capsule omega-3 fatty acids EPA DHA "
            "nutritional supplement heart health triglyceride reduction "
            "cardiovascular brain function omega-3 supplement"
        ),
    },
    "Kremil S": {
        "generic_name": "Aluminum Hydroxide + Magnesium Hydroxide + Simethicone",
        "drug_class":   "Antacid / Gastrointestinal",
        "dosage_form":  "chewable tablet",
        "color":        "white green",
        "shape":        "round chewable",
        "indication":   "antacid heartburn stomach acid indigestion ulcer bloating gas hyperacidity",
        "description": (
            "Kremil S white green chewable tablet aluminum hydroxide magnesium hydroxide simethicone "
            "antacid heartburn acid indigestion stomach ulcer gas bloating hyperacidity "
            "chewable tablet gastrointestinal relief"
        ),
    },
    "Medicol": {
        "generic_name": "Ibuprofen",
        "drug_class":   "Pain Relief / NSAID / Antipyretic",
        "dosage_form":  "capsule",
        "color":        "red white",
        "shape":        "capsule oval",
        "indication":   "pain headache fever ibuprofen NSAID muscle pain menstrual dysmenorrhea toothache",
        "description": (
            "Medicol red white capsule ibuprofen 200mg NSAID analgesic antipyretic "
            "mild moderate pain headache muscle pain menstrual cramps dysmenorrhea "
            "toothache fever red white capsule pain relief"
        ),
    },
    "Neozep": {
        "generic_name": "Phenylephrine + Chlorphenamine + Paracetamol",
        "drug_class":   "Cold and Flu Remedy",
        "dosage_form":  "tablet",
        "color":        "white",
        "shape":        "round circular",
        "indication":   "colds flu fever nasal congestion runny nose allergy sneezing body pain",
        "description": (
            "Neozep white round tablet phenylephrine chlorphenamine paracetamol "
            "colds flu nasal congestion runny nose fever allergy sneezing body pain "
            "cold flu remedy decongestant antihistamine"
        ),
    },
}


class CapsuleAIModel:
    """
    Ensemble medication identification model.
    Uses TF-IDF retrieval as the primary engine, with optional ML and BiLSTM components.
    Falls back to the embedded OTC drug knowledge base when no trained model files exist.
    """

    def __init__(self):
        self.nlp            = MedicationNLPProcessor()
        self.tfidf_model    = None
        self.tfidf_model_ml = None   # original TF-IDF for ML prediction (fixed vocabulary)
        self.ml_model       = None
        self.label_encoder  = None
        self.img_label_encoder = None
        self.vocab          = None
        self.drug_df        = None
        self.ensemble_config = None
        self.bilstm_model   = None
        self.image_db       = {}      # drug_name_lower → [path, ...]
        self.is_ready       = False
        self._load_all()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_all(self):
        """Load all artefacts independently — one failure never blocks the others."""
        # Drug DB must come before TF-IDF so the fallback builder can use it
        loaders = [
            ('Drug database',   self._load_drug_df),
            ('TF-IDF',          self._load_tfidf),
            ('ML model',        self._load_ml_model),
            ('Label encoder',   self._load_label_encoder),
            ('Vocab',           self._load_vocab),
            ('Ensemble config', self._load_ensemble_config),
            ('BiLSTM',          self._load_bilstm),
            ('Image DB',        self._load_image_db),
        ]
        for name, loader in loaders:
            try:
                loader()
            except Exception as exc:
                print(f"[CAPSULE AI] {name} failed to load: {exc}")
                traceback.print_exc()

        self.is_ready = (self.drug_df is not None)
        attr_map = {
            'Drug database': 'drug_df', 'TF-IDF': 'tfidf_model',
            'ML model': 'ml_model', 'Label encoder': 'label_encoder',
            'Vocab': 'vocab', 'Ensemble config': 'ensemble_config',
            'BiLSTM': 'bilstm_model',
        }
        loaded = [n for n, _ in loaders if getattr(self, attr_map.get(n, '_'), None) is not None]
        if self.image_db:
            loaded.append('Image DB')
        print(f"[CAPSULE AI] Loaded: {', '.join(loaded) or 'none'}. "
              f"Drugs: {len(self.drug_df) if self.drug_df is not None else 0}. "
              f"Images: {sum(len(v) for v in self.image_db.values())}. "
              f"Ready: {self.is_ready}")

    def _load_drug_df(self):
        # ── Priority 0: trained_models/capsule_dataset_clean.csv (from train_models.py) ─
        clean_csv = os.path.join(MODEL_DIR, 'capsule_dataset_clean.csv')
        if os.path.exists(clean_csv):
            df = pd.read_csv(clean_csv)
            for col in ['drug_name', 'generic_name', 'category', 'dosage_form',
                        'color', 'shape', 'text_description', 'description',
                        'strength', 'Strength Unit', 'Manufacture',
                        'Administration Route', 'Public price (SAR)',
                        'Storage Conditions', 'Legal Classification']:
                if col not in df.columns:
                    df[col] = ''
            df['drug_name'] = df['drug_name'].astype(str).str.lower().str.strip()
            self.drug_df = df
            print(f"[CAPSULE AI] capsule_dataset_clean.csv loaded: {len(self.drug_df)} drugs.")
            return

        # ── Priority 1: DataSet/capsule_dataset.csv.csv (bilingual raw) ──────
        bilingual_csv = os.path.join(os.path.dirname(BASE_DIR), 'DataSet', 'capsule_dataset.csv.csv')
        if os.path.exists(bilingual_csv):
            df = pd.read_csv(bilingual_csv)
            df = df.rename(columns={
                'drug_name_en':    'drug_name',
                'generic_name_en': 'generic_name',
                'drug_class_en':   'category',
                'dosage_form_en':  'dosage_form',
                'color_en':        'color',
                'shape_en':        'shape',
                'indications_en':  'indication',
                'manufacturer_en': 'manufacturer',
                'rx_otc_status_en': 'Legal Classification',
            })
            # Build text_description from available columns
            def _build_text(row):
                parts = [
                    f"drug {str(row.get('drug_name','')).lower()}",
                    f"generic {str(row.get('generic_name','')).lower()}",
                    f"form {str(row.get('dosage_form','')).lower()}",
                    f"color {str(row.get('color','')).lower()}",
                    f"shape {str(row.get('shape','')).lower()}",
                    f"category {str(row.get('category','')).lower()}",
                    str(row.get('indication', '')).lower(),
                ]
                ar_parts = [
                    str(row.get('drug_name_ar', '')),
                    str(row.get('generic_name_ar', '')),
                    str(row.get('drug_class_ar', '')),
                    str(row.get('indications_ar', '')),
                ]
                en = ' '.join(p for p in parts if p and p != 'nan')
                ar = ' '.join(p for p in ar_parts if p and p != 'nan')
                return (en + ' ' + ar).strip()
            df['text_description'] = df.apply(_build_text, axis=1)
            df['description'] = df['text_description']
            for col in ['drug_name', 'generic_name', 'category', 'dosage_form',
                        'color', 'shape', 'text_description', 'description',
                        'strength', 'Strength Unit', 'Manufacture',
                        'Administration Route', 'Public price (SAR)',
                        'Storage Conditions', 'Legal Classification']:
                if col not in df.columns:
                    df[col] = ''
            df['drug_name'] = df['drug_name'].astype(str).str.lower().str.strip()
            self.drug_df = df
            print(f"[CAPSULE AI] capsule_dataset.csv.csv loaded: {len(self.drug_df)} drugs.")
            return

        # ── Priority 2: capsule_dataset.csv (legacy notebook output) ─────────
        capsule_csv = os.path.join(BASE_DIR, 'datasets', 'capsule_dataset.csv')
        if os.path.exists(capsule_csv):
            df = pd.read_csv(capsule_csv)
            df = df.rename(columns={
                'drug_class':      'category',
                'nlp_description': 'text_description',
            })
            for col in ['drug_name', 'generic_name', 'category', 'dosage_form',
                        'color', 'shape', 'text_description', 'description',
                        'strength', 'Strength Unit', 'Manufacture',
                        'Administration Route', 'Public price (SAR)',
                        'Storage Conditions', 'Legal Classification']:
                if col not in df.columns:
                    df[col] = ''
            df['drug_name'] = df['drug_name'].astype(str).str.lower().str.strip()
            self.drug_df = df
            print(f"[CAPSULE AI] capsule_dataset.csv loaded: {len(self.drug_df)} drugs.")
            return

        # ── Priority 3: trained_models CSV ───────────────────────────────────
        path = os.path.join(MODEL_DIR, 'ksa_drug_database_clean.csv')
        if os.path.exists(path):
            self.drug_df = pd.read_csv(path)
            print(f"[CAPSULE AI] Drug database loaded: {len(self.drug_df)} records.")
            return

        # ── Priority 4: embedded OTC knowledge base ───────────────────────────
        self._build_otc_drug_df()
        print(f"[CAPSULE AI] Drug database built from OTC knowledge base: "
              f"{len(self.drug_df)} records.")

    def _build_otc_drug_df(self):
        """Build drug_df from the embedded OTC_DRUG_KNOWLEDGE dict."""
        rows = []
        for drug_name, info in OTC_DRUG_KNOWLEDGE.items():
            text = (
                f"drug {drug_name.lower()} "
                f"generic {info['generic_name'].lower()} "
                f"form {info['dosage_form']} "
                f"color {info['color']} "
                f"shape {info['shape']} "
                f"category {info['drug_class'].lower()} "
                f"{info['indication']} "
                f"{info['description']}"
            )
            rows.append({
                'drug_name':            drug_name.lower(),
                'generic_name':         info['generic_name'].lower(),
                'dosage_form':          info['dosage_form'],
                'category':             info['drug_class'],
                'color':                info['color'],
                'shape':                info['shape'],
                'text_description':     text,
                'description':          info['description'],
                'strength':             '',
                'Strength Unit':        '',
                'Manufacture':          '',
                'Administration Route': 'oral',
                'Public price (SAR)':   '',
                'Storage Conditions':   'Store below 30°C',
                'Legal Classification': 'OTC',
            })
        self.drug_df = pd.DataFrame(rows)

    def _load_tfidf(self):
        path = os.path.join(MODEL_DIR, 'tfidf_retrieval.pkl')
        if os.path.exists(path):
            self.tfidf_model = joblib.load(path)
            self.tfidf_model_ml = joblib.load(path)  # keep original for ML
            print("[CAPSULE AI] TF-IDF model loaded.")
        elif self.drug_df is not None:
            # No pre-trained pkl — build from the current drug database
            self._rebuild_tfidf_index()

    def _load_ml_model(self):
        path = os.path.join(MODEL_DIR, 'best_ml_model.pkl')
        if os.path.exists(path):
            self.ml_model = joblib.load(path)
            print("[CAPSULE AI] ML model loaded.")

    def _load_label_encoder(self):
        path = os.path.join(MODEL_DIR, 'label_encoder.pkl')
        if os.path.exists(path):
            self.label_encoder = joblib.load(path)
            print("[CAPSULE AI] Label encoder loaded.")
        img_path = os.path.join(MODEL_DIR, 'img_label_encoder.pkl')
        if os.path.exists(img_path):
            self.img_label_encoder = joblib.load(img_path)

    def _load_vocab(self):
        path = os.path.join(MODEL_DIR, 'vocab.pkl')
        if os.path.exists(path):
            self.vocab = joblib.load(path)

    def _load_ensemble_config(self):
        path = os.path.join(MODEL_DIR, 'ensemble_config.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.ensemble_config = json.load(f)

    def _load_bilstm(self):
        path = os.path.join(MODEL_DIR, 'bilstm_best.pt')
        if not os.path.exists(path) or self.vocab is None or self.ensemble_config is None:
            return
        try:
            import os as _os
            _os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
            import torch
            import torch.nn as nn

            class BiLSTMClassifier(nn.Module):
                def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
                    super().__init__()
                    self.embedding = nn.Embedding(vocab_size, embed_dim)
                    self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2,
                                        batch_first=True, bidirectional=True)
                    self.attention = nn.Linear(hidden_dim * 2, 1)
                    self.fc = nn.Sequential(
                        nn.Dropout(0.3),
                        nn.Linear(hidden_dim * 2, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(hidden_dim, num_classes),
                    )

                def forward(self, x):
                    emb = self.embedding(x)
                    out, _ = self.lstm(emb)
                    attn = torch.softmax(self.attention(out), dim=1)
                    context = (out * attn).sum(dim=1)
                    return self.fc(context)

            vocab_size  = self.ensemble_config.get('vocab_size', 57)
            num_classes = self.ensemble_config.get('num_text_classes', 9)
            model = BiLSTMClassifier(vocab_size=vocab_size, embed_dim=128,
                                     hidden_dim=256, num_classes=num_classes)
            model.load_state_dict(torch.load(path, map_location='cpu'))
            model.eval()
            self.bilstm_model = model
            print("[CAPSULE AI] BiLSTM model loaded.")
        except Exception as e:
            print(f"[CAPSULE AI] BiLSTM skipped: {e}")

    def _load_image_db(self):
        """
        Build image_db (drug_name_lower → [abs_path, ...]).

        Priority 1 — capsule_image_map.csv (notebook Step 3 output).
        Priority 2 — scan Drug Vision/Data Combined folders directly.
        """
        self.image_db = {}

        # ── Priority 1: capsule_image_map.csv ────────────────────────────────
        imap_path = os.path.join(BASE_DIR, 'datasets', 'capsule_image_map.csv')
        if os.path.exists(imap_path):
            try:
                imap = pd.read_csv(imap_path)
                for drug, group in imap.groupby('drug_name'):
                    paths = []
                    for rel in group['image_path'].tolist():
                        abs_p = rel if os.path.isabs(rel) else os.path.join(BASE_DIR, rel.replace('/', os.sep))
                        if os.path.isfile(abs_p):
                            paths.append(abs_p)
                    if paths:
                        self.image_db[str(drug).lower()] = sorted(paths)
                total = sum(len(v) for v in self.image_db.values())
                print(f"[CAPSULE AI] Image DB (capsule_image_map.csv): "
                      f"{len(self.image_db)} drugs, {total:,} images.")
                return
            except Exception as exc:
                print(f"[CAPSULE AI] image_map load error: {exc}")

        # ── Priority 2: scan Drug Vision folders ──────────────────────────────
        base = os.path.join(BASE_DIR, 'datasets', 'dataset2_pharma_drugs',
                            'Drug Vision', 'Data Combined')
        if not os.path.isdir(base):
            return
        for drug_folder in sorted(os.listdir(base)):
            folder_path = os.path.join(base, drug_folder)
            if not os.path.isdir(folder_path):
                continue
            imgs = sorted([
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            if imgs:
                self.image_db[drug_folder.lower()] = imgs
        total = sum(len(v) for v in self.image_db.values())
        if self.image_db:
            print(f"[CAPSULE AI] Image DB (folder scan): "
                  f"{len(self.image_db)} drugs, {total:,} images.")

    # ------------------------------------------------------------------
    # Public image lookup
    # ------------------------------------------------------------------

    def get_drug_image(self, drug_name: str, idx: int = 0) -> str | None:
        """Return an absolute image path for a given drug name, or None."""
        name_lower = str(drug_name).lower().strip()
        # Exact match
        if name_lower in self.image_db:
            imgs = self.image_db[name_lower]
            return imgs[idx % len(imgs)] if imgs else None
        # Partial match
        for key, imgs in self.image_db.items():
            if key in name_lower or name_lower in key:
                return imgs[idx % len(imgs)] if imgs else None
        return None

    # ------------------------------------------------------------------
    # Core identification
    # ------------------------------------------------------------------

    def identify_medication(self, description: str, top_k: int = 5,
                            color_filter: str = '', shape_filter: str = '',
                            packaging_filter: str = '', use_filter: str = '') -> dict:
        """
        Main public method: identify medication from a natural language description.
        Supports both English and Arabic input.
        """
        if not description or not description.strip():
            return self._empty_response('incomplete', 'No description provided.')

        from nlp_processor import is_arabic
        arabic_mode = is_arabic(description)

        effective_query = description
        if arabic_mode:
            effective_query = self.nlp.build_arabic_query(description)
            if not effective_query.strip():
                effective_query = description

        completeness = self.nlp.assess_completeness(description)

        if completeness['is_ambiguous']:
            return {
                'status': 'incomplete',
                'results': [],
                'completeness': completeness,
                'query_info': {'original': description, 'arabic_mode': arabic_mode},
            }

        # Direct drug name match — if the query matches a drug name exactly,
        # boost it to the top with high confidence regardless of TF-IDF score.
        direct_match_name = None
        if self.drug_df is not None:
            q_lower = description.strip().lower()
            q_raw   = description.strip()
            exact = self.drug_df[self.drug_df['drug_name'].str.lower().str.strip() == q_lower]
            if exact.empty and 'drug_name_ar' in self.drug_df.columns:
                # Try Arabic drug name column
                exact = self.drug_df[self.drug_df['drug_name_ar'].fillna('').str.strip() == q_raw]
            if exact.empty:
                # Try Arabic name match via text_description
                for idx, row in self.drug_df.iterrows():
                    td = str(row.get('text_description', ''))
                    if q_raw in td:
                        dn = str(row.get('drug_name', '')).lower().strip()
                        if dn:
                            direct_match_name = dn
                            break
            if not exact.empty and direct_match_name is None:
                direct_match_name = str(exact.iloc[0]['drug_name']).lower().strip()

        try:
            results = self._run_ensemble(effective_query, top_k)
        except Exception as exc:
            print(f"[CAPSULE AI] Identification error: {exc}")
            results = self._tfidf_fallback(effective_query, top_k)

        # If we found a direct name match, ensure it's in results and boost it
        if direct_match_name and results:
            for r in results:
                if r.get('medication', '').lower().strip() == direct_match_name:
                    r['confidence_score'] = max(r['confidence_score'], 0.97)
                    r['confidence_pct'] = round(r['confidence_score'] * 100, 1)
                    r['confidence_level'] = 'high'
                    break
            else:
                # Direct match not in results — add it from drug_df
                match_rows = self.drug_df[
                    self.drug_df['drug_name'].str.lower().str.strip() == direct_match_name
                ]
                if not match_rows.empty:
                    match_result = self._build_result(match_rows.iloc[0], 0.97, 1)
                    results.insert(0, match_result)
        elif direct_match_name and not results:
            match_rows = self.drug_df[
                self.drug_df['drug_name'].str.lower().str.strip() == direct_match_name
            ]
            if not match_rows.empty:
                results = [self._build_result(match_rows.iloc[0], 0.97, 1)]

        # Apply explicit filter weights (each filter = fixed % of total score)
        # Color: 10%, Shape: 10%, Packaging: 10%
        FILTER_WEIGHTS = {
            'color':     0.10,
            'shape':     0.10,
            'packaging': 0.10,
        }
        has_filters = color_filter or shape_filter or packaging_filter
        if results and has_filters:
            # The AI base score occupies the remaining weight
            active_weight = sum(
                w for key, w in FILTER_WEIGHTS.items()
                if (key == 'color' and color_filter)
                or (key == 'shape' and shape_filter)
                or (key == 'packaging' and packaging_filter)
            )
            base_weight = 1.0 - active_weight  # AI score portion

            for r in results:
                r_color = (r.get('color')       or '').lower()
                r_shape = (r.get('shape')       or '').lower()
                r_form  = (r.get('dosage_form') or '').lower()
                r_desc  = (r.get('description') or '').lower()

                filter_score = 0.0
                if color_filter:
                    if color_filter in r_color:
                        filter_score += FILTER_WEIGHTS['color']
                if shape_filter:
                    if shape_filter in r_shape:
                        filter_score += FILTER_WEIGHTS['shape']
                if packaging_filter:
                    if packaging_filter in r_form or packaging_filter in r_desc:
                        filter_score += FILTER_WEIGHTS['packaging']

                # Final score = (AI base × its weight) + filter matches
                r['confidence_score'] = max(0.05, min(1.0,
                    r['confidence_score'] * base_weight + filter_score))
                pct = round(r['confidence_score'] * 100, 1)
                r['confidence_pct'] = pct
                r['confidence_level'] = 'high' if pct >= 80 else ('medium' if pct >= 60 else 'low')

        # Penalise results whose category doesn't match the query's
        # extracted category. Applied post-calibration so it sticks.
        # e.g. searching "influenza" → category "cold flu"; Adalat (Calcium
        # Channel Blocker) has no overlap → heavy penalty pushes it below
        # the threshold or to the bottom of the list.
        query_cats = completeness.get('attributes', {}).get('categories', [])
        if results and query_cats:
            for r in results:
                r_cat  = (r.get('category')    or '').lower()
                r_desc = (r.get('description') or '').lower()
                r_gen  = (r.get('generic_name') or '').lower()
                cat_match = any(
                    self._flexible_text_match(qc, r_cat, r_desc, r_gen)
                    for qc in query_cats
                )
                if not cat_match:
                    r['confidence_score'] = max(0.05, r['confidence_score'] - 0.12)
                    pct = round(r['confidence_score'] * 100, 1)
                    r['confidence_pct'] = pct
                    r['confidence_level'] = 'high' if pct >= 80 else ('medium' if pct >= 60 else 'low')

        if not results:
            return {
                'status': 'no_match',
                'results': [],
                'completeness': completeness,
                'query_info': {'original': description, 'arabic_mode': arabic_mode},
            }

        results = [r for r in results if r['confidence_score'] >= 0.15]

        # Apply filter matching as a SCORING signal rather than hard elimination.
        # This ensures confidence_pct changes visibly when filters are toggled.
        # Weights: color=30%, shape=25%, packaging=20%, intended_use=25%.
        # The AI/NLP text match (drug name) is the PRIMARY score (60%),
        # filters contribute the remaining 40%.
        if color_filter or shape_filter or packaging_filter or use_filter:
            for r in results:
                r_color = (r.get('color')      or '').lower()
                r_shape = (r.get('shape')      or '').lower()
                r_form  = (r.get('dosage_form')or '').lower()
                r_cat   = (r.get('category')   or '').lower()
                r_desc  = (r.get('description')or '').lower()

                match_score = 0.0
                match_max   = 0.0
                if color_filter:
                    match_max += 0.30
                    if r_color and (color_filter in r_color or r_color in color_filter):
                        match_score += 0.30
                if shape_filter:
                    match_max += 0.25
                    if shape_filter in r_shape or r_shape in shape_filter:
                        match_score += 0.25
                if packaging_filter:
                    match_max += 0.20
                    if packaging_filter in r_form or packaging_filter in r_desc:
                        match_score += 0.20
                if use_filter:
                    match_max += 0.25
                    if use_filter in r_cat or use_filter in r_desc:
                        match_score += 0.25

                # Compute filter ratio and blend with AI confidence
                filter_ratio = (match_score / match_max) if match_max > 0 else 1.0
                ai_conf = r['confidence_score']
                # 60% AI/NLP text match (main) + 40% filter match (secondary)
                blended = 0.60 * ai_conf + 0.40 * filter_ratio
                r['confidence_score'] = round(max(0.05, min(1.0, blended)), 4)
                pct = round(r['confidence_score'] * 100, 1)
                r['confidence_pct'] = pct
                r['confidence_level'] = 'high' if pct >= 80 else ('medium' if pct >= 60 else 'low')

            # Re-sort by blended score and remove very low matches
            results.sort(key=lambda x: x['confidence_score'], reverse=True)
            results = [r for r in results if r['confidence_score'] >= 0.10]

        # Final direct-match boost — ensure exact drug name match is always top
        if direct_match_name and results:
            for r in results:
                if r.get('medication', '').lower().strip() == direct_match_name:
                    r['confidence_score'] = max(r['confidence_score'], 0.97)
                    r['confidence_pct'] = round(r['confidence_score'] * 100, 1)
                    r['confidence_level'] = 'high'
                    break
            results.sort(key=lambda x: x['confidence_score'], reverse=True)
            for i, r in enumerate(results):
                r['rank'] = i + 1

        # Attach image paths
        for r in results:
            r.setdefault('image_path', self.get_drug_image(r.get('medication', '')))

        return {
            'status': 'success' if results else 'no_match',
            'results': results,
            'completeness': completeness,
            'query_info': {
                'original':   description,
                'processed':  effective_query if arabic_mode else self.nlp.preprocess(description),
                'arabic_mode': arabic_mode,
            },
        }

    # ------------------------------------------------------------------
    # Ensemble pipeline
    # ------------------------------------------------------------------

    def _run_ensemble(self, description: str, top_k: int) -> list:
        enriched = self.nlp.build_query_vector(description)
        attrs    = self.nlp.extract_attributes(description)

        tfidf_results = self._tfidf_retrieve(enriched, top_k * 2)
        if not tfidf_results:
            return []

        if self.ml_model is not None:
            tfidf_results = self._ml_rerank(description, tfidf_results, top_k)

        if self.bilstm_model is not None and self.vocab is not None:
            tfidf_results = self._bilstm_blend(description, tfidf_results)

        tfidf_results = self._attribute_boost(tfidf_results, attrs)
        tfidf_results.sort(key=lambda x: x['confidence_score'], reverse=True)
        tfidf_results = self._calibrate_scores(tfidf_results)

        return tfidf_results[:top_k]

    def _tfidf_retrieve(self, query: str, top_k: int) -> list:
        if self.tfidf_model is None or self.drug_df is None:
            return self._keyword_fallback(query, top_k)
        try:
            from sklearn.metrics.pairwise import cosine_similarity

            if isinstance(self.tfidf_model, tuple):
                vectorizer, drug_matrix = self.tfidf_model
                q_vec = vectorizer.transform([query])
            elif hasattr(self.tfidf_model, 'transform'):
                vectorizer  = self.tfidf_model
                drug_texts  = self.drug_df['text_description'].fillna('').tolist()
                drug_matrix = vectorizer.transform(drug_texts)
                q_vec       = vectorizer.transform([query])
            else:
                return self._keyword_fallback(query, top_k)

            sims    = cosine_similarity(q_vec, drug_matrix).flatten()
            top_idx = sims.argsort()[::-1][:top_k]

            results = []
            for idx in top_idx:
                if sims[idx] < 0.05:
                    continue
                row = self.drug_df.iloc[idx]
                results.append(self._build_result(row, float(sims[idx]), len(results) + 1))
            return results
        except Exception as exc:
            print(f"[CAPSULE AI] TF-IDF error: {exc}")
            return self._keyword_fallback(query, top_k)

    def _ml_rerank(self, raw_query: str, candidates: list, top_k: int) -> list:
        try:
            is_pipeline = hasattr(self.ml_model, 'steps')
            if is_pipeline:
                proba    = self.ml_model.predict_proba([raw_query])[0]
                clf_step = self.ml_model.steps[-1][1]
                classes  = getattr(clf_step, 'classes_', None)
            elif (self.tfidf_model_ml or self.tfidf_model) is not None:
                vec = self.tfidf_model_ml or self.tfidf_model
                q_vec  = vec.transform([raw_query])
                proba  = self.ml_model.predict_proba(q_vec)[0]
                classes = getattr(self.ml_model, 'classes_', None)
            else:
                return candidates

            if classes is None or len(classes) == 0:
                return candidates

            if self.label_encoder is not None:
                class_names = [
                    self.label_encoder.classes_[c] if c < len(self.label_encoder.classes_) else str(c)
                    for c in classes
                ]
            else:
                class_names = [str(c) for c in classes]

            alpha = self.ensemble_config.get('alpha', 0.5) if self.ensemble_config else 0.5
            for cand in candidates:
                drug_name    = cand['medication'].lower().strip()
                matched_prob = 0.0
                for cls_idx, cls_name in enumerate(class_names):
                    if cls_name.lower() in drug_name or drug_name in cls_name.lower():
                        matched_prob = float(proba[cls_idx])
                        break
                cand['confidence_score'] = (1 - alpha) * cand['confidence_score'] + alpha * matched_prob
            return candidates
        except Exception as exc:
            print(f"[CAPSULE AI] ML rerank error: {exc}")
            return candidates

    def _bilstm_blend(self, query: str, candidates: list, weight: float = 0.25) -> list:
        try:
            import os as _os
            _os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
            import torch
            import torch.nn.functional as F

            max_seq_len = self.ensemble_config.get('max_seq_len', 64) if self.ensemble_config else 64
            tokens  = query.lower().split()
            indices = [self.vocab.get(t, 0) for t in tokens[:max_seq_len]]
            indices += [0] * (max_seq_len - len(indices))
            tensor  = torch.tensor([indices], dtype=torch.long)

            with torch.no_grad():
                logits = self.bilstm_model(tensor)
                probs  = torch.nn.functional.softmax(logits, dim=1)[0]

            if self.label_encoder is None:
                return candidates

            for cand in candidates:
                drug_name    = cand['medication'].lower().strip()
                bilstm_prob  = 0.0
                for cls_idx, cls_name in enumerate(self.label_encoder.classes_):
                    if cls_name.lower() in drug_name or drug_name in cls_name.lower():
                        bilstm_prob = float(probs[cls_idx]) if cls_idx < len(probs) else 0.0
                        break
                cand['confidence_score'] = (1 - weight) * cand['confidence_score'] + weight * bilstm_prob
            return candidates
        except Exception as exc:
            print(f"[CAPSULE AI] BiLSTM blend error: {exc}")
            return candidates
    
    def _normalize_match_tokens(self, text: str) -> set:
        """
        Convert text into meaningful tokens for flexible category matching.
        This helps match similar category phrases even when they are not written exactly the same.
        """
        import re

        text = str(text or '').lower()
        tokens = re.findall(r'\b\w+\b', text)

        stopwords = {
            'and', 'or', 'the', 'a', 'an', 'for', 'of', 'to', 'with',
            'used', 'use', 'remedy', 'medicine', 'drug', 'medication'
        }

        return {t for t in tokens if t not in stopwords and len(t) > 1}

    def _flexible_text_match(self, query_term: str, *candidate_texts: str) -> bool:
        """
        Flexible text matching for categories, indications, and symptoms.
        It compares meaningful words instead of requiring exact phrase matching.
        """
        query = str(query_term or '').lower().strip()
        combined = ' '.join(str(t or '').lower() for t in candidate_texts)

        if not query or not combined:
            return False

        if query in combined:
            return True

        query_tokens = self._normalize_match_tokens(query)
        candidate_tokens = self._normalize_match_tokens(combined)

        if not query_tokens or not candidate_tokens:
            return False

        overlap = query_tokens & candidate_tokens
        return len(overlap) == len(query_tokens)

    def _attribute_boost(self, results: list, attrs: dict) -> list:
        for r in results:
            boost        = 0.0
            desc_lower   = (r.get('description',   '') or '').lower()
            form_lower   = (r.get('dosage_form',   '') or '').lower()
            color_lower  = (r.get('color',         '') or '').lower()
            shape_lower  = (r.get('shape',         '') or '').lower()
            cat_lower    = (r.get('category',      '') or '').lower()
            generic_lower = (r.get('generic_name', '') or '').lower()

            matched = {'colors': [], 'forms': [], 'shapes': [], 'categories': []}

            for color in attrs.get('colors', []):
                if color in color_lower or color in desc_lower:
                    boost += 0.12
                    matched['colors'].append(color)

            for form in attrs.get('forms', []):
                if form in form_lower or form in desc_lower:
                    boost += 0.10
                    matched['forms'].append(form)

            for shape in attrs.get('shapes', []):
                if shape in shape_lower or shape in desc_lower:
                    boost += 0.10
                    matched['shapes'].append(shape)

            for cat in attrs.get('categories', []):
                if self._flexible_text_match(cat, cat_lower, desc_lower, generic_lower):
                    boost += 0.20
                    matched['categories'].append(cat)

            r['confidence_score'] = min(1.0, r['confidence_score'] + boost)
            r['matched_attributes'] = matched
        return results

    # ------------------------------------------------------------------
    # Fallback methods
    # ------------------------------------------------------------------

    def _tfidf_fallback(self, description: str, top_k: int) -> list:
        enriched = self.nlp.build_query_vector(description)
        return self._tfidf_retrieve(enriched, top_k)

    def _keyword_fallback(self, query: str, top_k: int) -> list:
        if self.drug_df is None:
            return []
        query_words = set(query.lower().split())
        scores = []
        for idx, row in self.drug_df.iterrows():
            text     = str(row.get('text_description', '')).lower()
            combined = f"{text} {str(row.get('drug_name','')).lower()} {str(row.get('generic_name','')).lower()}"
            overlap  = len(query_words & set(combined.split()))
            if overlap > 0:
                scores.append((idx, overlap / max(len(query_words), 1)))
        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in scores[:top_k]:
            results.append(self._build_result(self.drug_df.iloc[idx], score, len(results) + 1))
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_result(self, row, score: float, rank: int) -> dict:
        drug_name = str(row.get('drug_name', 'Unknown')).title()
        generic   = str(row.get('generic_name', '')).title()
        strength  = str(row.get('strength', ''))
        unit      = str(row.get('Strength Unit', ''))
        form      = str(row.get('dosage_form', ''))
        category  = str(row.get('category', ''))
        color     = str(row.get('color', ''))
        shape     = str(row.get('shape', ''))
        manufacturer = str(row.get('Manufacture', ''))
        route     = str(row.get('Administration Route', ''))
        price     = str(row.get('Public price (SAR)', ''))
        storage   = str(row.get('Storage Conditions', ''))
        legal     = str(row.get('Legal Classification', ''))
        description = str(row.get('text_description', ''))

        strength_label = f"{strength} {unit}".strip() if strength and strength != 'nan' else ''
        confidence_pct = round(score * 100, 1)
        level = 'high' if confidence_pct >= 80 else ('medium' if confidence_pct >= 60 else 'low')

        return {
            'rank':                 rank,
            'medication':           drug_name,
            'generic_name':         generic,
            'strength':             strength_label,
            'dosage_form':          form,
            'category':             category,
            'color':                color,
            'shape':                shape,
            'route':                route,
            'manufacturer':         manufacturer,
            'price':                price,
            'storage':              storage,
            'legal_classification': legal,
            'confidence_score':     round(score, 4),
            'confidence_pct':       confidence_pct,
            'confidence_level':     level,
            'description':          description,
            'matched_attributes':   {},
            'image_path':           self.get_drug_image(drug_name),
        }

    def _calibrate_scores(self, results: list) -> list:
        """
        Calibrate raw ensemble scores into display-ready matching scores.

        IMPORTANT (Note 12): The output 'confidence_score' and 'confidence_pct' values
        are RELATIVE MATCHING SCORES, not calibrated statistical probabilities.
        They represent weighted attribute similarity across the ensemble (TF-IDF,
        ML, BiLSTM, keyword filters) mapped via linear scaling + power-law compression.
        A score of 85% means "strong attribute alignment with the input," not "85%
        probability of correct identification." This distinction must be clear in
        all user-facing displays and documentation.
        """
        if not results:
            return results
        FLOOR, CEIL = 0.15, 0.97
        for r in results:
            s = r['confidence_score']
            # Map raw score (0–1) linearly into [FLOOR, CEIL], then apply
            # power-law compression to spread mid-range values.
            # Unlike min-max normalization, this preserves absolute quality:
            # a weak best-match (e.g. raw 0.3) won't be inflated to CEIL.
            mapped     = FLOOR + s * (CEIL - FLOOR)
            calibrated = mapped ** 0.85
            r['confidence_score'] = round(min(CEIL, calibrated), 4)
            pct = round(r['confidence_score'] * 100, 1)
            r['confidence_pct']   = pct
            r['confidence_level'] = 'high' if pct >= 80 else ('medium' if pct >= 60 else 'low')
        return results

    def augment_drug_database(self, drugs: list) -> None:
        """
        Merge the full application drug catalog into the AI model's search index.
        Called at startup (inside app context) so TF-IDF can search all DB drugs.
        """
        if not drugs:
            return

        rows = []
        existing_names = set()
        if self.drug_df is not None:
            existing_names = {str(n).lower() for n in self.drug_df['drug_name']}

        for d in drugs:
            name = str(d.get('drug_name') or '').lower().strip()
            if not name or name in existing_names:
                continue
            existing_names.add(name)

            generic  = str(d.get('generic_name',  '') or '').lower()
            form     = str(d.get('dosage_form',    '') or '').lower()
            category = str(d.get('category',       '') or '').lower()
            color    = str(d.get('color',          '') or '').lower()
            shape    = str(d.get('shape',          '') or '').lower()
            strength = str(d.get('strength',       '') or '')
            s_unit   = str(d.get('strength_unit',  '') or '')
            desc     = str(d.get('description',    '') or '').lower()

            # Arabic fields for bilingual search
            name_ar        = str(d.get('drug_name_ar',        '') or '')
            generic_ar     = str(d.get('generic_name_ar',     '') or '')
            indications    = str(d.get('indications',         '') or '').lower()
            indications_ar = str(d.get('indications_ar',      '') or '')

            parts = [
                f"drug {name}",
                f"generic {generic}"  if generic  else '',
                f"form {form}"        if form     else '',
                f"color {color}"      if color    else '',
                f"shape {shape}"      if shape    else '',
                f"category {category}" if category else '',
                f"strength {strength} {s_unit}".strip() if strength else '',
                desc[:200] if desc else '',
                name_ar           if name_ar        else '',
                generic_ar        if generic_ar     else '',
                indications[:150] if indications    else '',
                indications_ar    if indications_ar else '',
            ]
            text_description = ' '.join(p for p in parts if p).strip()

            rows.append({
                'drug_name':            name,
                'generic_name':         generic,
                'dosage_form':          form,
                'category':             category,
                'color':                color,
                'shape':                shape,
                'strength':             strength,
                'Strength Unit':        s_unit,
                'text_description':     text_description,
                'description':          desc,
                'Manufacture':          str(d.get('manufacturer',  '') or ''),
                'Administration Route': str(d.get('admin_route',   '') or ''),
                'Public price (SAR)':   str(d.get('price',         '') or ''),
                'Storage Conditions':   str(d.get('storage',       '') or ''),
                'Legal Classification': str(d.get('legal_class',   '') or ''),
            })

        # Enrich existing sparse rows with fields from SQLite
        if self.drug_df is not None:
            enrich_map = {'color': 'color', 'shape': 'shape',
                          'category': 'category', 'dosage_form': 'dosage_form'}
            ar_fields  = ['drug_name_ar', 'generic_name_ar', 'indications_ar',
                          'side_effects_ar', 'contraindications_ar']
            for d in drugs:
                name = str(d.get('drug_name') or '').lower().strip()
                if not name:
                    continue
                mask = self.drug_df['drug_name'].str.lower() == name
                if not mask.any():
                    continue
                for src_key, field in enrich_map.items():
                    val = str(d.get(src_key) or '').strip()
                    if field not in self.drug_df.columns:
                        self.drug_df[field] = ''
                    current = self.drug_df.loc[mask, field].iloc[0]
                    is_empty = pd.isna(current) or str(current).strip() in ('', 'nan')
                    if val and is_empty:
                        self.drug_df.loc[mask, field] = val.lower()
                # Store Arabic columns in drug_df
                for ar_key in ar_fields:
                    val = str(d.get(ar_key) or '').strip()
                    if ar_key not in self.drug_df.columns:
                        self.drug_df[ar_key] = ''
                    if val:
                        self.drug_df.loc[mask, ar_key] = val
                # Rebuild text_description — include Arabic fields
                idx_val = self.drug_df[mask].index[0]
                row = self.drug_df.loc[idx_val]
                def col(c):
                    v = str(row[c]).strip() if c in self.drug_df.columns and pd.notna(row[c]) else ''
                    return '' if v in ('', 'nan') else v
                name_ar     = str(d.get('drug_name_ar', '') or '').strip()
                generic_ar  = str(d.get('generic_name_ar', '') or '').strip()
                indic       = str(d.get('indications', '') or '').strip()
                indic_ar    = str(d.get('indications_ar', '') or '').strip()
                parts = [
                    f"drug {col('drug_name')}",
                    f"generic {col('generic_name')}"  if col('generic_name')  else '',
                    f"form {col('dosage_form')}"       if col('dosage_form')   else '',
                    f"color {col('color')}"            if col('color')         else '',
                    f"shape {col('shape')}"            if col('shape')         else '',
                    f"category {col('category')}"      if col('category')      else '',
                    col('text_description')[:200],
                    name_ar, generic_ar,
                    indic[:150] if indic else '',
                    indic_ar,
                ]
                self.drug_df.loc[idx_val, 'text_description'] = ' '.join(p for p in parts if p).strip()

        if rows:
            new_df = pd.DataFrame(rows)
            self.drug_df = pd.concat(
                [self.drug_df, new_df] if self.drug_df is not None else [new_df],
                ignore_index=True,
            )

        if self.drug_df is not None:
            print(f"[CAPSULE AI] Drug database augmented: {len(self.drug_df)} total drugs.")
            self._rebuild_tfidf_index()

    def _rebuild_tfidf_index(self) -> None:
        """Rebuild the TF-IDF index from all current drug_df text_descriptions."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            texts = self.drug_df['text_description'].fillna('').tolist()
            new_vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=1,
                max_features=5000,
                sublinear_tf=True,
            )
            new_vectorizer.fit(texts)
            self.tfidf_model = new_vectorizer
            print(f"[CAPSULE AI] TF-IDF index rebuilt: {len(new_vectorizer.vocabulary_)} terms, "
                  f"{len(texts)} documents.")
        except Exception as exc:
            print(f"[CAPSULE AI] TF-IDF rebuild failed: {exc}")

    def _empty_response(self, status: str, message: str) -> dict:
        return {'status': status, 'results': [], 'message': message,
                'completeness': {}, 'query_info': {}}

    def get_model_info(self) -> dict:
        return {
            'tfidf_loaded':         self.tfidf_model is not None,
            'ml_model_loaded':      self.ml_model is not None,
            'bilstm_loaded':        self.bilstm_model is not None,
            'label_encoder_loaded': self.label_encoder is not None,
            'drug_count':           len(self.drug_df) if self.drug_df is not None else 0,
            'image_db_count':       len(self.image_db),
            'is_ready':             self.is_ready,
            'ensemble_config':      self.ensemble_config,
        }
