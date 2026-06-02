"""
CAPSULE — Drug Interaction Checker
Queries the DrugInteraction table and returns risk-annotated results.
"""

from __future__ import annotations
from typing import List, Dict, Any
import re


# ── Risk badge metadata ────────────────────────────────────────────────────────
RISK_BADGE: Dict[str, Dict[str, str]] = {
    "high": {
        "label": "HIGH RISK",
        "color": "#dc2626",          # red-600
        "bg": "#fef2f2",             # red-50
        "border": "#fca5a5",         # red-300
        "icon": "fa-triangle-exclamation",
        "text_color": "#991b1b",
    },
    "medium": {
        "label": "MODERATE RISK",
        "color": "#d97706",          # amber-600
        "bg": "#fffbeb",             # amber-50
        "border": "#fcd34d",         # amber-300
        "icon": "fa-circle-exclamation",
        "text_color": "#92400e",
    },
    "low": {
        "label": "LOW RISK",
        "color": "#2563eb",          # blue-600
        "bg": "#eff6ff",             # blue-50
        "border": "#93c5fd",         # blue-300
        "icon": "fa-circle-info",
        "text_color": "#1e40af",
    },
}

# Risk sort order for display
RISK_ORDER = {"high": 0, "medium": 1, "low": 2}


def _normalize(name: str) -> str:
    """Lowercase, strip punctuation for fuzzy matching."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


class DrugInteractionChecker:
    """Checks for drug-drug interactions using the database."""

    def check_by_names(self, drug_names: List[str]) -> List[Dict[str, Any]]:
        """
        Given a list of drug name strings, return all pairwise interactions found.

        Each result dict contains:
            drug_a, drug_b, risk_level, interaction_description,
            mechanism, patient_guidance, clinical_note, badge
        """
        from app import DrugInteraction  # lazy import to avoid circular deps

        if len(drug_names) < 2:
            return []

        all_interactions: List[Dict[str, Any]] = []
        seen_pairs = set()

        # Fetch all interactions once — small table, fine for SQLite
        all_db = DrugInteraction.query.all()

        normed_inputs = [_normalize(n) for n in drug_names]

        for di in all_db:
            norm_a = _normalize(di.drug_a_name)
            norm_b = _normalize(di.drug_b_name)

            # Check if any input drug matches drug_a and another matches drug_b
            matched_input_a = None
            matched_input_b = None

            for raw, normed in zip(drug_names, normed_inputs):
                if norm_a in normed or normed in norm_a:
                    matched_input_a = raw
                elif norm_b in normed or normed in norm_b:
                    matched_input_b = raw

            if matched_input_a is None or matched_input_b is None:
                # Try both directions
                for raw, normed in zip(drug_names, normed_inputs):
                    if norm_b in normed or normed in norm_b:
                        matched_input_b = raw if matched_input_b is None else matched_input_b
                    if norm_a in normed or normed in norm_a:
                        matched_input_a = raw if matched_input_a is None else matched_input_a

            if matched_input_a and matched_input_b and matched_input_a != matched_input_b:
                pair_key = tuple(sorted([_normalize(matched_input_a), _normalize(matched_input_b)]))
                db_pair_key = tuple(sorted([norm_a, norm_b]))
                combined_key = (pair_key, db_pair_key)

                if combined_key not in seen_pairs:
                    seen_pairs.add(combined_key)
                    badge = RISK_BADGE.get(di.risk_level, RISK_BADGE["low"])
                    all_interactions.append({
                        "drug_a": di.drug_a_name,
                        "drug_b": di.drug_b_name,
                        "risk_level": di.risk_level,
                        "interaction_description": di.interaction_description or "",
                        "mechanism": di.mechanism or "",
                        "patient_guidance": di.patient_guidance or "Consult your healthcare provider.",
                        "clinical_note": di.clinical_note or "",
                        "badge": badge,
                    })

        # Sort by severity: high → medium → low
        all_interactions.sort(key=lambda x: RISK_ORDER.get(x["risk_level"], 99))
        return all_interactions

    def check_pair(self, drug_a: str, drug_b: str) -> Dict[str, Any] | None:
        """Check a single pair. Returns the interaction dict or None."""
        results = self.check_by_names([drug_a, drug_b])
        return results[0] if results else None

    @staticmethod
    def highest_risk(interactions: List[Dict[str, Any]]) -> str | None:
        """Return the highest risk level string from a list of interactions."""
        if not interactions:
            return None
        for level in ("high", "medium", "low"):
            if any(i["risk_level"] == level for i in interactions):
                return level
        return None
