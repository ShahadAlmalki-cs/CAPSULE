"""
CAPSULE — Long-Term Drug Effects Registry
Provides warnings for medications known to cause significant effects with extended use.
"""

from __future__ import annotations
from typing import Dict, Any, Optional


# ── Long-term effect definitions ───────────────────────────────────────────────
LONG_TERM_DRUG_EFFECTS: Dict[str, Dict[str, Any]] = {
    "amiodarone": {
        "drug_name": "Amiodarone",
        "effect_summary": "Pulmonary toxicity, thyroid dysfunction, liver damage, corneal deposits",
        "description": (
            "Long-term amiodarone use (>6 months) can cause pulmonary toxicity (lung scarring), "
            "thyroid disorders (both hypo- and hyperthyroidism), hepatotoxicity, photosensitivity, "
            "and corneal microdeposits. Regular monitoring of thyroid, liver, and lung function is essential."
        ),
        "monitoring": ["Thyroid function (TFTs) every 6 months", "Liver function tests (LFTs)", "Chest X-ray annually", "Pulmonary function tests", "Ophthalmology review"],
        "min_duration_months": 6,
        "severity": "high",
        "patient_guidance": (
            "Do not stop this medication without medical advice. Wear sunscreen and protective clothing outdoors. "
            "Report any shortness of breath, vision changes, or unusual weight changes immediately."
        ),
    },
    "bisphosphonates": {
        "drug_name": "Bisphosphonates (Alendronate / Risedronate)",
        "effect_summary": "Osteonecrosis of the jaw, atypical femoral fractures",
        "description": (
            "Extended use (>5 years) of bisphosphonates is associated with osteonecrosis of the jaw (ONJ), "
            "atypical femoral fractures, and esophageal irritation. A drug holiday may be recommended after 3–5 years."
        ),
        "monitoring": ["Dental examination before and during therapy", "Bone density (DEXA) scan", "Report thigh or hip pain"],
        "min_duration_months": 36,
        "severity": "medium",
        "patient_guidance": (
            "Maintain good oral hygiene. Inform your dentist you are taking this medication before any dental procedures. "
            "Report any new hip or thigh pain to your doctor."
        ),
    },
    "denosumab": {
        "drug_name": "Denosumab",
        "effect_summary": "Rebound bone loss on discontinuation, osteonecrosis of the jaw",
        "description": (
            "Abrupt discontinuation of denosumab causes rapid rebound bone loss and significantly increases fracture risk. "
            "Transition to another antiresorptive agent is required when stopping. ONJ risk similar to bisphosphonates."
        ),
        "monitoring": ["Bone density monitoring", "Calcium and vitamin D levels", "Dental review"],
        "min_duration_months": 12,
        "severity": "medium",
        "patient_guidance": (
            "Never stop this injection without medical supervision — stopping suddenly can weaken bones rapidly. "
            "Ensure adequate calcium and vitamin D intake."
        ),
    },
    "methotrexate": {
        "drug_name": "Methotrexate",
        "effect_summary": "Liver fibrosis/cirrhosis, pulmonary toxicity, bone marrow suppression",
        "description": (
            "Cumulative methotrexate doses above 1.5 g are associated with liver fibrosis and cirrhosis. "
            "Pulmonary toxicity (methotrexate pneumonitis) can occur at any dose. Bone marrow suppression "
            "requires regular blood count monitoring. Folic acid supplementation reduces mucosal side effects."
        ),
        "monitoring": ["FBC (full blood count) every 2–3 months", "LFTs every 2–3 months", "Chest X-ray if respiratory symptoms", "Renal function"],
        "min_duration_months": 3,
        "severity": "high",
        "patient_guidance": (
            "Take folic acid as prescribed. Avoid alcohol. Report any shortness of breath, mouth sores, or unusual bruising/bleeding. "
            "Use effective contraception — methotrexate is harmful to an unborn baby."
        ),
    },
    "isotretinoin": {
        "drug_name": "Isotretinoin (Accutane / Roaccutane)",
        "effect_summary": "Teratogenicity, depression, dry mucous membranes, elevated lipids",
        "description": (
            "Isotretinoin is highly teratogenic — even a single dose can cause severe birth defects. "
            "Psychiatric effects including depression and suicidal ideation have been reported. "
            "Dry eyes, lips, and skin are common. Liver enzymes and lipid levels require monitoring."
        ),
        "monitoring": ["Pregnancy test (females) monthly", "LFTs and lipid panel", "Mental health screening"],
        "min_duration_months": 1,
        "severity": "high",
        "patient_guidance": (
            "Absolutely do not become pregnant while taking isotretinoin or for 1 month after stopping. "
            "Use two forms of contraception. Report mood changes or depression immediately. "
            "Use moisturizing lip balm and eye drops as needed."
        ),
    },
    "dmpa": {
        "drug_name": "DMPA (Depo-Provera / Medroxyprogesterone)",
        "effect_summary": "Bone mineral density loss, prolonged amenorrhea",
        "description": (
            "Prolonged DMPA use (>2 years) is associated with reduced bone mineral density (BMD). "
            "BMD typically recovers after discontinuation. Return to fertility may be delayed by 6–18 months."
        ),
        "monitoring": ["Bone density assessment if >2 years of use", "Calcium and vitamin D supplementation recommended"],
        "min_duration_months": 24,
        "severity": "low",
        "patient_guidance": (
            "Ensure adequate calcium (1000–1300 mg/day) and vitamin D intake. "
            "Engage in weight-bearing exercise. Plan accordingly if pregnancy is desired — fertility may take time to return."
        ),
    },
    "corticosteroids": {
        "drug_name": "Corticosteroids (Prednisolone / Dexamethasone)",
        "effect_summary": "Adrenal suppression, osteoporosis, diabetes, cataracts, weight gain",
        "description": (
            "Long-term systemic corticosteroid use (>3 months) causes HPA-axis suppression, "
            "osteoporosis, hyperglycemia, Cushingoid features, increased infection risk, "
            "cataracts, skin thinning, and cardiovascular risk. "
            "Dose tapering is mandatory — never stop abruptly after prolonged use."
        ),
        "monitoring": ["Blood glucose", "Bone density (DEXA)", "Blood pressure", "Eye examination", "BMI / weight"],
        "min_duration_months": 3,
        "severity": "high",
        "patient_guidance": (
            "Never stop steroids suddenly — always taper as directed. Carry a steroid treatment card. "
            "Take calcium and vitamin D supplements. Monitor blood sugar. Report signs of infection promptly."
        ),
    },
    "finasteride": {
        "drug_name": "Finasteride",
        "effect_summary": "Post-finasteride syndrome (sexual dysfunction, depression) in some patients",
        "description": (
            "A subset of patients report persistent sexual dysfunction (decreased libido, erectile dysfunction), "
            "depression, and cognitive changes that may persist after stopping finasteride (Post-Finasteride Syndrome). "
            "PSA levels are reduced by ~50% — adjust PSA screening accordingly."
        ),
        "monitoring": ["PSA level (halve the result for true PSA)", "Mood and sexual function review"],
        "min_duration_months": 6,
        "severity": "low",
        "patient_guidance": (
            "Inform your doctor about any mood changes or sexual side effects. "
            "If used for BPH/hair loss, discuss benefit-risk at each visit. "
            "Women of childbearing age should not handle crushed tablets."
        ),
    },
    "leflunomide": {
        "drug_name": "Leflunomide",
        "effect_summary": "Hepatotoxicity, peripheral neuropathy, hypertension, teratogenicity",
        "description": (
            "Leflunomide can cause liver damage (requires LFT monitoring), peripheral neuropathy, "
            "and is highly teratogenic with an extremely long half-life (active metabolite persists for up to 2 years). "
            "Cholestyramine washout procedure is required before conception."
        ),
        "monitoring": ["LFTs monthly for 6 months, then every 6–8 weeks", "Blood pressure", "FBC", "Neuropathy symptoms"],
        "min_duration_months": 3,
        "severity": "high",
        "patient_guidance": (
            "Report any tingling/numbness in hands or feet. Avoid alcohol. "
            "Use effective contraception — the drug remains in the body for up to 2 years after stopping. "
            "Cholestyramine washout is required before trying to conceive."
        ),
    },
    "warfarin": {
        "drug_name": "Warfarin",
        "effect_summary": "Bleeding risk, drug/food interactions, INR instability",
        "description": (
            "Long-term warfarin therapy requires careful INR monitoring and is prone to numerous drug and "
            "food interactions (especially vitamin K-rich foods). Bleeding is the primary risk. "
            "Calciphylaxis and skin necrosis are rare but serious complications."
        ),
        "monitoring": ["INR check every 4 weeks (or more frequently if unstable)", "Signs of bleeding"],
        "min_duration_months": 3,
        "severity": "medium",
        "patient_guidance": (
            "Maintain a consistent diet — avoid sudden large changes in vitamin K intake (leafy greens). "
            "Report any unusual bleeding (gums, urine, stools). Carry a warfarin alert card. "
            "Consult your doctor before any new medication, including OTC drugs."
        ),
    },
}


def get_long_term_warning(drug_name: str) -> Optional[Dict[str, Any]]:
    """
    Return the long-term effect info dict for a drug name, or None if not found.
    Uses case-insensitive partial matching.
    """
    key = drug_name.lower().strip()
    # Direct key match
    if key in LONG_TERM_DRUG_EFFECTS:
        return LONG_TERM_DRUG_EFFECTS[key]

    # Partial match — check if any key is contained in the drug name or vice versa
    for effect_key, data in LONG_TERM_DRUG_EFFECTS.items():
        if effect_key in key or key in effect_key:
            return data
        # Also match by the stored drug_name field
        stored = data["drug_name"].lower()
        if key in stored or stored in key:
            return data

    return None


def get_warnings_for_medications(medication_names: list[str]) -> list[Dict[str, Any]]:
    """
    Given a list of medication names, return all applicable long-term warnings.
    Each entry in the returned list includes the original drug_name matched.
    """
    warnings = []
    seen = set()
    for name in medication_names:
        info = get_long_term_warning(name)
        if info and info["drug_name"] not in seen:
            seen.add(info["drug_name"])
            warnings.append(info)
    return warnings


# Severity badge metadata (reuses RISK_BADGE color scheme)
SEVERITY_BADGE = {
    "high": {
        "label": "HIGH CONCERN",
        "color": "#dc2626",
        "bg": "#fef2f2",
        "border": "#fca5a5",
        "icon": "fa-triangle-exclamation",
    },
    "medium": {
        "label": "MODERATE CONCERN",
        "color": "#d97706",
        "bg": "#fffbeb",
        "border": "#fcd34d",
        "icon": "fa-circle-exclamation",
    },
    "low": {
        "label": "MONITOR",
        "color": "#2563eb",
        "bg": "#eff6ff",
        "border": "#93c5fd",
        "icon": "fa-circle-info",
    },
}
