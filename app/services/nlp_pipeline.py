"""
CAPSULE - NLP Pipeline Service

Bridges the modular service layer to MedicationNLPProcessor.
"""

import os
import sys
import re
import logging
from typing import List

logger = logging.getLogger(__name__)

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        ai_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), 'AI_models')
        if ai_dir not in sys.path:
            sys.path.insert(0, ai_dir)
        from nlp_processor import MedicationNLPProcessor
        _nlp = MedicationNLPProcessor()
    return _nlp


def preprocess_text(text: str) -> str:
    """Clean and normalize text for model input."""
    nlp = _get_nlp()
    if hasattr(nlp, 'preprocess'):
        return nlp.preprocess(text)
    # Fallback: basic cleaning
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def tokenize(text: str) -> List[str]:
    """Tokenize text into word list."""
    return preprocess_text(text).split()


def text_to_indices(text: str, vocab: dict, max_len: int = 64) -> List[int]:
    """Convert text to vocabulary indices for BiLSTM input."""
    tokens = tokenize(text)
    unk_idx = vocab.get('<UNK>', 1)
    indices = [vocab.get(t, unk_idx) for t in tokens[:max_len]]
    # Pad to max_len
    if len(indices) < max_len:
        pad_idx = vocab.get('<PAD>', 0)
        indices += [pad_idx] * (max_len - len(indices))
    return indices
