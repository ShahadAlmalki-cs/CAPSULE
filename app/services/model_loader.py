"""
CAPSULE - Model Loader Service

Bridges the modular service layer to the monolithic CapsuleAIModel.
"""

import os
import sys
import logging

logger = logging.getLogger(__name__)

_ai_model = None


def _get_ai_model():
    """Lazy-load the CapsuleAIModel singleton."""
    global _ai_model
    if _ai_model is None:
        ai_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), 'AI_models')
        if ai_dir not in sys.path:
            sys.path.insert(0, ai_dir)
        from ai_model import CapsuleAIModel
        _ai_model = CapsuleAIModel()
    return _ai_model


def get_model(name: str):
    """
    Return a named model component from the CapsuleAIModel.

    Supported names:
        tfidf, ml_model, label_encoder, vocab, bilstm,
        ensemble_config, drug_database, drug_vectors
    """
    model = _get_ai_model()
    mapping = {
        'tfidf':           model.tfidf_model,
        'ml_model':        model.ml_model,
        'label_encoder':   model.label_encoder,
        'vocab':           model.vocab,
        'bilstm':          model.bilstm_model,
        'ensemble_config': model.ensemble_config,
        'drug_database':   (model.drug_df.to_dict('records')
                            if model.drug_df is not None else []),
        'drug_vectors':    None,
    }
    return mapping.get(name)
