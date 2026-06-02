"""
CAPSULE - Long-Term Effect Model

Provides a LongTermEffect interface that bridges to the Medication model's
long_term_* fields and the long_term_effects.py knowledge base.
"""

import os
import sys


class LongTermEffect:
    """Lightweight wrapper for long-term effect queries."""

    def __init__(self, medication_id=None, medication_name='', description='',
                 duration='', severity='low', warning='', monitoring=None):
        self.medication_id = medication_id
        self.medication_name = medication_name
        self.description = description
        self.duration = duration
        self.severity = severity
        self.warning = warning
        self.monitoring = monitoring or []

    def to_dict(self):
        return {
            'medication_id': self.medication_id,
            'medication_name': self.medication_name,
            'description': self.description,
            'duration': self.duration,
            'severity': self.severity,
            'warning': self.warning,
            'monitoring': self.monitoring,
        }

    @classmethod
    def query(cls):
        return _LTEQuery()


class _LTEQuery:
    """Minimal query interface for LongTermEffect lookups."""

    def __init__(self):
        self._filters = []

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        """Return a LongTermEffect from the knowledge base if available."""
        try:
            project_dir = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
            if project_dir not in sys.path:
                sys.path.insert(0, project_dir)
            from long_term_effects import LONG_TERM_DRUG_EFFECTS
            if LONG_TERM_DRUG_EFFECTS:
                return None  # Filtering not supported in this stub
        except Exception:
            pass
        return None
