"""
================================================================================
CAPSULE - Helper Utilities
================================================================================
"""

import os
import re
import math
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


def format_confidence(score: float, as_percent: bool = True) -> str:
    """Format a confidence score for display."""
    if score is None:
        return 'N/A'
    if as_percent:
        return f"{score * 100:.1f}%"
    return f"{score:.3f}"


def get_confidence_class(score: float) -> str:
    """
    Get Bootstrap color class based on confidence score.

    Returns:
        'success'  (green)  — score >= 0.8
        'warning'  (yellow) — score >= 0.5
        'danger'   (red)    — score <  0.5
    """
    if score >= 0.8:
        return 'success'
    elif score >= 0.5:
        return 'warning'
    else:
        return 'danger'


def truncate_text(text: str, max_length: int = 100, suffix: str = '...') -> str:
    """Truncate text to max_length characters."""
    if not text:
        return ''
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)].rstrip() + suffix


def format_date(dt: Any, fmt: str = '%d %b %Y') -> str:
    """Format a datetime or date object to string."""
    if dt is None:
        return 'N/A'
    try:
        if hasattr(dt, 'strftime'):
            return dt.strftime(fmt)
        return str(dt)
    except Exception:
        return str(dt)


def format_price_sar(price: Optional[float]) -> str:
    """Format price in Saudi Riyals."""
    if price is None:
        return 'N/A'
    return f"SAR {price:,.2f}"


def paginate_list(items: List[Any], page: int, per_page: int) -> Dict:
    """Simple list paginator."""
    total       = len(items)
    total_pages = math.ceil(total / per_page) if per_page > 0 else 1
    start       = (page - 1) * per_page
    end         = start + per_page
    return {
        'items':       items[start:end],
        'total':       total,
        'page':        page,
        'per_page':    per_page,
        'total_pages': total_pages,
        'has_prev':    page > 1,
        'has_next':    page < total_pages,
        'prev_page':   page - 1,
        'next_page':   page + 1,
    }


def get_dashboard_url(role: str) -> str:
    """Get the appropriate dashboard URL for a user role."""
    role_url_map = {
        'patient':    'patient.dashboard',
        'user':       'patient.dashboard',
        'doctor':     'doctor.dashboard',
        'admin':      'doctor.dashboard',
        'pharmacist': 'pharmacist.dashboard',
    }
    return role_url_map.get(role, 'auth.login')


def generate_patient_id() -> str:
    """Generate a unique patient ID for display."""
    import uuid
    return f"PT-{uuid.uuid4().hex[:8].upper()}"


def sanitize_search_query(query: str) -> str:
    """Sanitize a search query string."""
    if not query:
        return ''
    query = re.sub(r'[;\'"\\]', '', query)
    query = re.sub(r'\s+', ' ', query).strip()
    return query[:200]


def build_medication_search_context(results: List[Dict]) -> List[Dict]:
    """
    Enrich medication results with display-ready fields.

    Added fields per result:
        confidence_pct       — e.g. "36.0%"
        confidence_class     — Bootstrap class: 'success' | 'warning' | 'danger'
        confidence_bar_width — integer 0–100 for the progress bar width
        name_display         — title-cased brand name
        generic_display      — title-cased generic name
        has_image            — bool

    Args:
        results: Raw prediction results from ensemble_predictor

    Returns:
        Enriched results ready for template rendering
    """
    enriched = []
    for result in results:
        confidence = result.get('confidence', 0)
        enriched.append({
            **result,
            'confidence_pct':       format_confidence(confidence),
            'confidence_class':     get_confidence_class(confidence),
            'confidence_bar_width': int(min(confidence * 100, 100)),
            'name_display':         result.get('name_en', 'Unknown').title(),
            'generic_display':      result.get('generic_name', '').title(),
            'has_image':            bool(result.get('image_path')),
        })
    return enriched


def get_status_icon(status: str) -> str:
    """Get Font Awesome icon class for a medication status."""
    icons = {
        'active':        'fas fa-check-circle text-success',
        'completed':     'fas fa-check-double text-secondary',
        'discontinued':  'fas fa-times-circle text-danger',
        'pending':       'fas fa-clock text-warning',
        'dispensed':     'fas fa-pills text-info',
    }
    return icons.get(status, 'fas fa-circle text-secondary')


def log_search(user_id: int, query: str, query_type: str, results: List[Dict]) -> None:
    """Log a search action to the database (non-critical, errors are swallowed)."""
    try:
        import json
        from app import db
        from app.models.search_history import SearchHistory
        history = SearchHistory(
            user_id=user_id,
            query_text=query[:500],
            query_type=query_type,
            results=json.dumps([{
                'name':       r.get('name_en', ''),
                'confidence': r.get('confidence', 0),
            } for r in results[:5]]),
            result_count=len(results),
        )
        db.session.add(history)
        db.session.commit()
    except Exception as e:
        logger.debug(f"Search history log failed (non-critical): {e}")
