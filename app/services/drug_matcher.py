"""
================================================================================
CAPSULE - Drug Matcher Service

TF-IDF + embedding-based medication retrieval with robust DB fallback.
================================================================================
"""

import logging
import numpy as np
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def get_tfidf_matches(query: str, top_k: int = 5) -> List[Dict]:
    """Use TF-IDF vectorizer to find matching medications."""
    from app.services.model_loader import get_model
    from app.services.nlp_pipeline import preprocess_text

    tfidf         = get_model('tfidf')
    label_encoder = get_model('label_encoder')
    drug_database = get_model('drug_database') or []

    if tfidf is None:
        return _database_fallback_search(query, top_k)

    try:
        clean_query = preprocess_text(query)

        if hasattr(tfidf, 'predict_proba'):
            query_vec = tfidf.transform([clean_query]) if not hasattr(tfidf, 'named_steps') else tfidf[:-1].transform([clean_query])
            proba     = tfidf.predict_proba(tfidf.transform([clean_query]) if hasattr(tfidf, 'predict_proba') else query_vec)[0]
            top_idxs  = np.argsort(proba)[::-1][:top_k]
            results   = []
            for idx in top_idxs:
                drug_name = label_encoder.classes_[idx] if label_encoder and idx < len(label_encoder.classes_) else f'Drug_{idx}'
                drug_info = _find_drug_in_db(drug_name, drug_database)
                results.append(_build_result(drug_info, drug_name, float(proba[idx]), 'tfidf', len(results) + 1))
            return results

        elif hasattr(tfidf, 'predict'):
            pred      = tfidf.predict([clean_query])[0]
            drug_name = pred if isinstance(pred, str) else (label_encoder.classes_[pred] if label_encoder else str(pred))
            drug_info = _find_drug_in_db(drug_name, drug_database)
            return [_build_result(drug_info, drug_name, 0.8, 'tfidf', 1)]

    except Exception as e:
        logger.error(f"TF-IDF matching error: {e}")

    return _database_fallback_search(query, top_k)


def get_embedding_matches(query: str, top_k: int = 5) -> List[Dict]:
    """Use pre-computed drug vectors (cosine similarity) for matching."""
    from app.services.model_loader import get_model
    from app.services.nlp_pipeline import preprocess_text

    drug_vectors  = get_model('drug_vectors')
    tfidf         = get_model('tfidf')
    label_encoder = get_model('label_encoder')
    drug_database = get_model('drug_database') or []

    if drug_vectors is None or tfidf is None:
        return []

    try:
        clean_query = preprocess_text(query)

        if not hasattr(tfidf, 'transform'):
            return []
        query_vec = tfidf.transform([clean_query]).toarray()

        # Find the vectors array
        vec_key = next((k for k in ('vectors', 'drug_vectors', 'X', 'data') if k in drug_vectors), None)
        if vec_key is None:
            vec_key = list(drug_vectors.keys())[0]
        doc_vecs = drug_vectors[vec_key]

        if doc_vecs.ndim == 1:
            doc_vecs = doc_vecs.reshape(1, -1)

        # Ensure compatible shapes for cosine similarity
        q_norm = np.linalg.norm(query_vec)
        if q_norm == 0:
            return []
        if doc_vecs.shape[1 if doc_vecs.ndim > 1 else 0] != query_vec.shape[-1]:
            logger.warning(f"Embedding shape mismatch: doc_vecs={doc_vecs.shape}, query={query_vec.shape} — skipping")
            return []
        d_norms = np.linalg.norm(doc_vecs, axis=1)
        sims = np.dot(doc_vecs, query_vec.T).flatten() / (d_norms * q_norm + 1e-10)
        top_idxs = np.argsort(sims)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_idxs):
            score = float(sims[idx])
            if score < 0.01:
                continue
            drug_name = label_encoder.classes_[idx] if label_encoder and idx < len(label_encoder.classes_) else f'Drug_{idx}'
            drug_info = _find_drug_in_db(drug_name, drug_database)
            results.append(_build_result(drug_info, drug_name, min(score, 1.0), 'embedding', rank + 1))
        return results

    except Exception as e:
        logger.error(f"Embedding matching error: {e}")
    return []


def get_ml_predictions(query: str, top_k: int = 5) -> List[Dict]:
    """Use Random Forest / XGBoost model for predictions."""
    from app.services.model_loader import get_model
    from app.services.nlp_pipeline import preprocess_text

    ml_model      = get_model('ml_model')
    tfidf         = get_model('tfidf')
    label_encoder = get_model('label_encoder')
    drug_database = get_model('drug_database') or []

    if ml_model is None:
        return []

    try:
        clean_query = preprocess_text(query)

        # If ml_model is a Pipeline it handles vectorisation internally
        is_pipeline = hasattr(ml_model, 'steps')
        if is_pipeline:
            features = [clean_query]          # raw text list for the pipeline
        elif tfidf is not None:
            features = tfidf.transform([clean_query])   # pre-vectorise
        else:
            return []

        if hasattr(ml_model, 'predict_proba'):
            proba = ml_model.predict_proba(features)[0]
            # Resolve class labels — LabelEncoder stores them in .classes_
            classes = None
            if label_encoder is not None and hasattr(label_encoder, 'classes_'):
                classes = label_encoder.classes_
            elif is_pipeline:
                clf_step = ml_model.steps[-1][1]
                classes = getattr(clf_step, 'classes_', None)
            elif hasattr(ml_model, 'classes_'):
                classes = ml_model.classes_
            top_idxs = np.argsort(proba)[::-1][:top_k]
            results  = []
            for rank, idx in enumerate(top_idxs):
                drug_name = classes[idx] if classes is not None and idx < len(classes) else f'Drug_{idx}'
                drug_info = _find_drug_in_db(str(drug_name), drug_database)
                results.append(_build_result(drug_info, str(drug_name), float(proba[idx]), 'ml', rank + 1))
            return results

        elif hasattr(ml_model, 'predict'):
            pred      = ml_model.predict(features)[0]
            drug_name = str(pred).title()
            return [_build_result({}, drug_name, 0.85, 'ml', 1)]

    except Exception as e:
        logger.error(f"ML prediction error: {e}")
    return []


# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_result(drug_info: Dict, drug_name: str, confidence: float, source: str, rank: int) -> Dict:
    """Build a standardised result dict."""
    strength = (
        str(drug_info.get('strength', '') or '') + ' ' +
        str(drug_info.get('Strength Unit', '') or '')
    ).strip()
    return {
        'rank':         rank,
        'name_en':      (drug_info.get('drug_name') or drug_name).title(),
        'generic_name': (drug_info.get('generic_name') or '').title(),
        'strength':     strength,
        'dosage_form':  (drug_info.get('dosage_form') or '').title(),
        'category':     (drug_info.get('category') or '').title(),
        'manufacturer': (drug_info.get('Manufacture') or drug_info.get('manufacturer') or ''),
        'confidence':   confidence,
        'source':       source,
    }


def _find_drug_in_db(drug_name: str, database: List[Dict]) -> Dict:
    """Find a drug in the CSV database by name (case-insensitive)."""
    name_lower = drug_name.lower().strip()
    # Exact match
    for drug in database:
        if drug.get('drug_name', '').lower().strip() == name_lower:
            return drug
        if drug.get('generic_name', '').lower().strip() == name_lower:
            return drug
    # Partial match
    for drug in database:
        dn = drug.get('drug_name', '').lower()
        gn = drug.get('generic_name', '').lower()
        if name_lower in dn or name_lower in gn:
            return drug
    return {'drug_name': drug_name}


def _database_fallback_search(query: str, top_k: int = 5) -> List[Dict]:
    """
    Keyword-based database search used when ML models are unavailable.
    Searches against the SQLite medication catalog.
    """
    from app.models.medication import Medication
    from app.services.nlp_pipeline import tokenize, preprocess_text
    from sqlalchemy import or_

    try:
        clean_query = preprocess_text(query)
        tokens      = tokenize(clean_query)

        if not tokens:
            meds = Medication.query.filter_by(is_available_ksa=True).limit(top_k).all()
        else:
            conditions = []
            for token in tokens[:4]:
                if len(token) >= 3:
                    conditions.extend([
                        Medication.name_en.ilike(f'%{token}%'),
                        Medication.generic_name.ilike(f'%{token}%'),
                        Medication.category.ilike(f'%{token}%'),
                        Medication.dosage_form.ilike(f'%{token}%'),
                        Medication.color.ilike(f'%{token}%'),
                        Medication.shape.ilike(f'%{token}%'),
                        Medication.usage_en.ilike(f'%{token}%'),
                    ])
            if conditions:
                meds = Medication.query.filter(or_(*conditions)).filter_by(
                    is_available_ksa=True
                ).limit(top_k).all()
            else:
                meds = Medication.query.filter_by(is_available_ksa=True).limit(top_k).all()

        results = []
        for rank, med in enumerate(meds):
            results.append({
                'rank':                 rank + 1,
                'medication_id':        med.id,
                'name_en':              med.name_en,
                'generic_name':         med.generic_name or '',
                'strength':             med.full_strength,
                'dosage_form':          med.dosage_form or '',
                'category':             med.category or '',
                'manufacturer':         med.manufacturer or '',
                'confidence':           max(0.35, 0.95 - rank * 0.1),
                'source':               'database',
                'image_path':           med.image_path,
                'has_long_term_effect': med.has_long_term_effect,
                'long_term_summary':    med.long_term_summary,
                'usage_en':             med.usage_en,
                'side_effects':         med.side_effects,
                'name_ar':              med.name_ar,
                'legal_classification': med.legal_classification,
                'public_price_sar':     med.public_price_sar,
            })
        return results

    except Exception as e:
        logger.error(f"Database fallback search error: {e}")
        return []
