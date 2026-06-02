"""
CAPSULE NLP Processor
Processes and extracts medical attributes from natural language medication descriptions.
Supports both English and Arabic input.

Bilingual Approach (Note 7): This module uses a TRANSLATE-THEN-PROCESS strategy,
NOT native Arabic NLP. Arabic queries are decomposed into known medical attributes
(color, shape, form, category, route) via keyword dictionaries, then mapped to their
English equivalents before processing. This approach was chosen because:
1. The medication database and trained models operate entirely in English
2. Medical terminology in Arabic (especially brand names) is often transliterated
3. Dictionary-based translation for a closed attribute vocabulary is more reliable
   than general-purpose MT for domain-specific pharmaceutical terms
4. It avoids dependency on external Arabic NLP libraries or translation APIs

Limitation: Free-text Arabic descriptions beyond the keyword vocabulary are not
processed — only recognized attribute keywords are translated and utilized.
"""

import re
import string


# ---------------------------------------------------------------------------
# Arabic ↔ English keyword maps (for bilingual search support)
# ---------------------------------------------------------------------------

ARABIC_COLOR_MAP = {
    'أبيض': 'white', 'أبيضاء': 'white', 'أبيضي': 'white', 'كريمي': 'white',
    'عاجي': 'white', 'فاتح': 'white',
    'أصفر': 'yellow', 'صفراء': 'yellow', 'ذهبي': 'yellow', 'ذهبية': 'yellow',
    'برتقالي': 'orange', 'برتقالية': 'orange',
    'أحمر': 'red', 'حمراء': 'red', 'وردي': 'pink', 'وردية': 'pink',
    'زهري': 'pink', 'زهرية': 'pink',
    'أزرق': 'blue', 'زرقاء': 'blue', 'كحلي': 'blue', 'سماوي': 'blue',
    'أخضر': 'green', 'خضراء': 'green', 'زيتوني': 'green',
    'بنفسجي': 'purple', 'بنفسجية': 'purple', 'أرجواني': 'purple',
    'بني': 'brown', 'بنية': 'brown', 'بيج': 'brown', 'خاكي': 'brown',
    'رمادي': 'gray', 'رمادية': 'gray', 'فضي': 'gray',
    'أسود': 'black', 'سوداء': 'black',
}

ARABIC_SHAPE_MAP = {
    'دائري': 'round', 'دائرية': 'round', 'مستدير': 'round', 'مستديرة': 'round',
    'بيضاوي': 'oval', 'بيضاوية': 'oval', 'ممدود': 'oval',
    'كبسولة': 'capsule', 'كبسول': 'capsule',
    'مستطيل': 'square', 'مستطيلة': 'square', 'مربع': 'square',
    'سداسي': 'hexagonal', 'سداسية': 'hexagonal',
    'مثلث': 'triangle', 'مثلثة': 'triangle',
    'مطول': 'oblong', 'مستطيل مطول': 'oblong',
}

ARABIC_FORM_MAP = {
    'قرص': 'tablet', 'أقراص': 'tablet', 'حبة': 'tablet', 'حبوب': 'tablet',
    'كبسولة': 'capsule', 'كبسولات': 'capsule',
    'شراب': 'syrup', 'محلول': 'syrup', 'سائل': 'syrup', 'قطرة فموية': 'syrup',
    'حقن': 'injection', 'حقنة': 'injection', 'أمبول': 'injection',
    'كريم': 'cream', 'مرهم': 'cream', 'جل': 'cream', 'لوشن': 'cream',
    'بخاخ': 'inhaler', 'رذاذ': 'inhaler', 'مستنشق': 'inhaler',
    'قطرة': 'eye drop', 'قطرات': 'eye drop', 'قطرة عين': 'eye drop',
    'تحميلة': 'suppository', 'لبوس': 'suppository',
    'مسحوق': 'powder', 'بودرة': 'powder',
    'لاصق': 'patch', 'رقعة': 'patch',
    'تسريب': 'infusion', 'محلول وريدي': 'infusion',
}

ARABIC_CATEGORY_MAP = {
    'سكري': 'diabetes', 'للسكر': 'diabetes', 'سكر الدم': 'diabetes',
    'سكر': 'diabetes', 'إنسولين': 'diabetes', 'سكريات': 'diabetes',
    'ضغط الدم': 'hypertension', 'ضغط دم': 'hypertension', 'الضغط': 'hypertension',
    'ضغط': 'hypertension', 'ضغط عال': 'hypertension', 'انخفاض ضغط': 'hypertension',
    'مسكن': 'pain relief', 'مسكن ألم': 'pain relief', 'ألم': 'pain relief',
    'صداع': 'pain relief', 'وجع': 'pain relief', 'حمى': 'pain relief',
    'خافض للحمى': 'pain relief', 'مضاد للحمى': 'pain relief',
    'مضاد حيوي': 'antibiotic', 'مضاد للبكتيريا': 'antibiotic',
    'مضاد للجراثيم': 'antibiotic', 'التهاب': 'antibiotic',
    'مضاد التهاب': 'anti-inflammatory', 'مضاد للالتهاب': 'anti-inflammatory',
    'روماتيزم': 'anti-inflammatory', 'مفاصل': 'anti-inflammatory',
    'مضاد للغثيان': 'antiemetic', 'غثيان': 'antiemetic', 'قيء': 'antiemetic',
    'كوليسترول': 'cholesterol', 'دهون الدم': 'cholesterol', 'دهون': 'cholesterol',
    'قلب': 'hypertension', 'ضربات القلب': 'hypertension',
    'حساسية': 'antihistamine',
    'قلق': 'anxiolytic', 'أعصاب': 'neurological',
    'تنفس': 'respiratory', 'ربو': 'respiratory', 'حساسية صدر': 'respiratory',
    'نزلة': 'respiratory', 'سعال': 'respiratory',
    'مضاد للفطريات': 'antifungal',
    'فيروس': 'antiviral', 'مضاد للفيروسات': 'antiviral',
    'نوم': 'sedative', 'مهدئ': 'sedative',
    'معدة': 'gastrointestinal', 'حموضة': 'gastrointestinal', 'قرحة': 'gastrointestinal',
    'إمساك': 'gastrointestinal', 'إسهال': 'gastrointestinal',
    'الزهايمر': 'neurological', 'ذاكرة': 'neurological',
    'غدة درقية': 'thyroid',
    'تخثر الدم': 'anticoagulant', 'سيولة الدم': 'anticoagulant',
}

ARABIC_ROUTE_MAP = {
    'فموي': 'oral', 'عن طريق الفم': 'oral', 'يبتلع': 'oral',
    'وريدي': 'injection', 'حقن وريدية': 'injection', 'في الوريد': 'injection',
    'عضلي': 'injection', 'حقن عضلية': 'injection',
    'جلدي': 'topical', 'على الجلد': 'topical',
    'استنشاق': 'inhalation',
}


def is_arabic(text: str) -> bool:
    """Return True if the text contains a significant proportion of Arabic characters."""
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06ff')
    return arabic_chars > len(text) * 0.2


def _strip_arabic_prefix(word: str) -> str:
    """Strip common Arabic prefixes (وال، بال، لل، لـ، ل، و، في، من) from a word."""
    # Order matters: longer prefixes first to avoid partial matches
    prefixes = ['والـ', 'وال', 'بالـ', 'بال', 'لـل', 'للـ', 'لل', 'الـ', 'ال',
                'لـ', 'بـ', 'وـ', 'فـ', 'كـ',
                'ل', 'ب', 'و', 'ف', 'ك']  # single-letter prefixes last
    for p in prefixes:
        if word.startswith(p) and len(word) > len(p) + 1:
            return word[len(p):]
    return word


def translate_arabic_query(text: str) -> str:
    """
    Translate Arabic keywords in a query to their English equivalents.
    Returns an English-equivalent query string suitable for TF-IDF matching.
    """
    words = text.split()
    translated_parts = []

    # Try 2-word and 1-word matches
    i = 0
    while i < len(words):
        two = ' '.join(words[i:i+2]) if i + 1 < len(words) else ''
        # Check all maps
        matched = None
        if two and two in ARABIC_COLOR_MAP:
            matched = ARABIC_COLOR_MAP[two]
            i += 2
        elif two and two in ARABIC_SHAPE_MAP:
            matched = ARABIC_SHAPE_MAP[two]
            i += 2
        elif two and two in ARABIC_FORM_MAP:
            matched = ARABIC_FORM_MAP[two]
            i += 2
        elif two and two in ARABIC_CATEGORY_MAP:
            matched = ARABIC_CATEGORY_MAP[two]
            i += 2
        elif two and two in ARABIC_ROUTE_MAP:
            matched = ARABIC_ROUTE_MAP[two]
            i += 2
        else:
            # Try stripping Arabic prefixes and retry
            stripped = _strip_arabic_prefix(words[i])
            if stripped != words[i]:
                if stripped in ARABIC_COLOR_MAP:
                    matched = ARABIC_COLOR_MAP[stripped]
                elif stripped in ARABIC_SHAPE_MAP:
                    matched = ARABIC_SHAPE_MAP[stripped]
                elif stripped in ARABIC_FORM_MAP:
                    matched = ARABIC_FORM_MAP[stripped]
                elif stripped in ARABIC_CATEGORY_MAP:
                    matched = ARABIC_CATEGORY_MAP[stripped]
                elif stripped in ARABIC_ROUTE_MAP:
                    matched = ARABIC_ROUTE_MAP[stripped]
            # If still no match, keep original (drug name may be in Arabic)
            i += 1

        if matched and matched not in translated_parts:
            translated_parts.append(matched)

    return ' '.join(translated_parts) if translated_parts else text


def extract_arabic_attributes(text: str) -> dict:
    """Extract attributes directly from Arabic text, handling prefixes."""
    attrs = {'colors': [], 'shapes': [], 'forms': [], 'categories': [], 'routes': []}
    words = text.split()

    ALL_MAPS = [
        (ARABIC_COLOR_MAP, 'colors'),
        (ARABIC_SHAPE_MAP, 'shapes'),
        (ARABIC_FORM_MAP, 'forms'),
        (ARABIC_CATEGORY_MAP, 'categories'),
        (ARABIC_ROUTE_MAP, 'routes'),
    ]

    def check_word(word):
        # Try as-is first, then stripped
        candidates = [word, _strip_arabic_prefix(word)]
        for candidate in candidates:
            for mapping, key in ALL_MAPS:
                if candidate in mapping:
                    val = mapping[candidate]
                    if val not in attrs[key]:
                        attrs[key].append(val)

    for i in range(len(words)):
        # 2-word phrases: try both original and prefix-stripped first word
        if i + 1 < len(words):
            next_w = words[i+1]
            check_word(words[i] + ' ' + next_w)
            stripped_i = _strip_arabic_prefix(words[i])
            if stripped_i != words[i]:
                check_word(stripped_i + ' ' + next_w)
        # Single word
        check_word(words[i])

    return attrs


class MedicationNLPProcessor:
    """Processes medication descriptions using rule-based NLP and pattern matching."""

    COLOR_KEYWORDS = {
        'white': ['white', 'cream', 'ivory', 'off-white', 'pale'],
        'yellow': ['yellow', 'golden', 'amber', 'gold'],
        'orange': ['orange', 'peach', 'salmon'],
        'red': ['red', 'crimson', 'pink', 'rose', 'maroon'],
        'blue': ['blue', 'navy', 'cobalt', 'teal', 'cyan', 'light blue'],
        'green': ['green', 'olive', 'mint', 'lime', 'dark green'],
        'purple': ['purple', 'violet', 'lavender', 'mauve', 'lilac'],
        'brown': ['brown', 'tan', 'beige', 'khaki', 'chocolate'],
        'black': ['black', 'dark', 'charcoal'],
        'gray': ['gray', 'grey', 'silver'],
    }

    SHAPE_KEYWORDS = {
        'round': ['round', 'circular', 'circle', 'disc', 'disk'],
        'oval': ['oval', 'ellipse', 'elliptical', 'oblong'],
        'capsule': ['capsule', 'gelcap', 'gel cap', 'softgel', 'soft gel'],
        'square': ['square', 'rectangular', 'rectangle'],
        'hexagonal': ['hexagonal', 'hexagon', 'six-sided'],
        'triangle': ['triangle', 'triangular'],
    }

    FORM_KEYWORDS = {
        'tablet': ['tablet', 'tab', 'pill', 'caplet'],
        'capsule': ['capsule', 'cap', 'gelcap'],
        'injection': ['injection', 'injectable', 'ampoule', 'vial', 'syringe', 'iv', 'intravenous'],
        'syrup': ['syrup', 'liquid', 'oral solution', 'suspension', 'elixir', 'drops'],
        'cream': ['cream', 'ointment', 'lotion', 'gel', 'topical', 'paste'],
        'powder': ['powder', 'sachet', 'granules'],
        'patch': ['patch', 'transdermal'],
        'inhaler': ['inhaler', 'spray', 'nebulizer', 'puffer'],
        'suppository': ['suppository', 'rectal'],
        'eye drop': ['eye drop', 'ophthalmic', 'eye'],
        'infusion': ['infusion', 'drip', 'bag', 'emulsion'],
    }

    CATEGORY_KEYWORDS = {
        'diabetes': ['diabetes', 'diabetic', 'blood sugar', 'glucose', 'insulin', 'gliclazide',
                     'metformin', 'glipizide', 'sitagliptin', 'hypoglycemic'],
        'hypertension': ['hypertension', 'blood pressure', 'antihypertensive', 'amlodipine',
                         'lisinopril', 'losartan', 'atenolol', 'bp', 'heart', 'cardiac'],
        'pain relief': ['pain', 'analgesic', 'painkiller', 'paracetamol', 'ibuprofen',
                        'diclofenac', 'headache', 'fever', 'antipyretic'],
        'antibiotic': ['antibiotic', 'infection', 'bacteria', 'amoxicillin', 'azithromycin',
                       'ciprofloxacin', 'antibiotic', 'antibacterial'],
        'anti-inflammatory': ['anti-inflammatory', 'inflammation', 'nsaid', 'corticosteroid',
                              'deflazacort', 'prednisolone', 'arthritis', 'swelling'],
        'antiviral': ['antiviral', 'virus', 'viral', 'letermovir', 'antifungal'],
        'antiemetic': ['antiemetic', 'nausea', 'vomit', 'chemotherapy', 'palonosetron',
                       'ondansetron', 'motion sickness'],
        'antifungal': ['antifungal', 'fungal', 'fungus', 'ozenoxacin', 'fluconazole'],
        'neurological': ['alzheimer', 'dementia', 'donepezil', 'neurological', 'memory',
                         'parkinson', 'epilepsy', 'seizure'],
        'adhd': ['adhd', 'attention', 'hyperactivity', 'methylphenidate', 'ritalin', 'stimulant'],
        'nutrition': ['nutrition', 'amino acid', 'parenteral', 'feeding', 'glucose', 'lipid'],
        'vasopressor': ['vasopressor', 'vasopressin', 'shock', 'vasoconstrictor'],
        # ── OTC / Drug Vision dataset categories ──────────────────────────────
        'cold flu': ['cold', 'colds', 'flu', 'influenza', 'runny nose', 'stuffy nose',
                     'nasal congestion', 'congestion', 'sneezing', 'chills', 'flu symptoms',
                     'phenylephrine', 'chlorphenamine', 'phenylpropanolamine', 'decongestant',
                     'antihistamine', 'neozep', 'bioflu', 'decolgen'],
        'antiseptic': ['antiseptic', 'mouthwash', 'gargle', 'oral hygiene', 'sore throat',
                       'hexetidine', 'tonsillitis', 'mouth infection', 'throat', 'oral care',
                       'bactidol', 'antibacterial mouthwash'],
        'supplement': ['supplement', 'vitamin', 'zinc', 'omega', 'fish oil', 'mineral',
                       'nutritional', 'omega-3', 'epa', 'dha', 'immune', 'immunity',
                       'deficiency', 'dayzinc', 'fish oil supplement'],
        'antacid': ['antacid', 'heartburn', 'stomach acid', 'hyperacidity', 'acidity',
                    'indigestion', 'acid reflux', 'bloating', 'gas', 'kremil', 'ulcer pain',
                    'aluminum hydroxide', 'magnesium hydroxide', 'simethicone'],
    }

    ROUTE_KEYWORDS = {
        'oral': ['oral', 'mouth', 'swallow', 'tablet', 'pill', 'syrup'],
        'injection': ['injection', 'iv', 'intravenous', 'subcutaneous', 'im', 'intramuscular'],
        'topical': ['topical', 'skin', 'apply', 'cream', 'ointment'],
        'inhalation': ['inhale', 'inhaler', 'spray', 'breathe'],
    }

    PACKAGING_KEYWORDS = {
        'blister': ['blister', 'blister pack', 'strip'],
        'bottle': ['bottle', 'container', 'flask', 'jar'],
        'ampoule': ['ampoule', 'ampule', 'vial'],
        'sachet': ['sachet', 'packet', 'envelope'],
        'tube': ['tube', 'squeeze'],
        'bag': ['bag', 'infusion bag', 'iv bag'],
    }

    def __init__(self):
        self._build_reverse_maps()

    def _build_reverse_maps(self):
        """Build reverse lookup maps for efficient keyword matching."""
        self._color_map = {}
        for canonical, keywords in self.COLOR_KEYWORDS.items():
            for kw in keywords:
                self._color_map[kw.lower()] = canonical

        self._shape_map = {}
        for canonical, keywords in self.SHAPE_KEYWORDS.items():
            for kw in keywords:
                self._shape_map[kw.lower()] = canonical

        self._form_map = {}
        for canonical, keywords in self.FORM_KEYWORDS.items():
            for kw in keywords:
                self._form_map[kw.lower()] = canonical

        self._category_map = {}
        for canonical, keywords in self.CATEGORY_KEYWORDS.items():
            for kw in keywords:
                self._category_map[kw.lower()] = canonical

    def preprocess(self, text: str) -> str:
        """Clean and normalize text for processing."""
        if not text:
            return ''
        text = text.lower().strip()
        text = re.sub(r'[^\w\s\-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def extract_attributes(self, text: str) -> dict:
        """Extract medication attributes from free-text description."""
        processed = self.preprocess(text)
        tokens = processed.split()

        attributes = {
            'colors': [],
            'shapes': [],
            'forms': [],
            'categories': [],
            'routes': [],
            'packaging': [],
            'keywords': [],
        }

        # Multi-word matching (up to 3-word phrases)
        phrases = self._get_phrases(tokens, max_n=3)

        for phrase in phrases:
            if phrase in self._color_map and self._color_map[phrase] not in attributes['colors']:
                attributes['colors'].append(self._color_map[phrase])
            if phrase in self._shape_map and self._shape_map[phrase] not in attributes['shapes']:
                attributes['shapes'].append(self._shape_map[phrase])
            if phrase in self._form_map and self._form_map[phrase] not in attributes['forms']:
                attributes['forms'].append(self._form_map[phrase])

        # Category matching
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            if any(kw in processed for kw in keywords):
                if category not in attributes['categories']:
                    attributes['categories'].append(category)

        # Route matching
        for route, keywords in self.ROUTE_KEYWORDS.items():
            if any(kw in processed for kw in keywords):
                if route not in attributes['routes']:
                    attributes['routes'].append(route)

        # Packaging matching
        for pkg, keywords in self.PACKAGING_KEYWORDS.items():
            if any(kw in processed for kw in keywords):
                if pkg not in attributes['packaging']:
                    attributes['packaging'].append(pkg)

        # Extract meaningful keywords (remove stopwords)
        stopwords = {'a', 'an', 'the', 'is', 'it', 'for', 'of', 'to', 'and', 'or',
                     'with', 'this', 'that', 'my', 'i', 'me', 'small', 'big', 'large',
                     'used', 'use', 'using', 'takes', 'take'}
        attributes['keywords'] = [t for t in tokens if len(t) > 2 and t not in stopwords]

        return attributes

    def _get_phrases(self, tokens: list, max_n: int = 3) -> list:
        """Generate n-grams (1 to max_n) from token list."""
        phrases = []
        for n in range(1, max_n + 1):
            for i in range(len(tokens) - n + 1):
                phrases.append(' '.join(tokens[i:i + n]))
        return phrases

    def build_query_vector(self, text: str) -> str:
        """Build an enriched query string by appending extracted attributes.
        Handles both English and Arabic input.
        """
        if is_arabic(text):
            return self.build_arabic_query(text)

        attrs = self.extract_attributes(text)
        parts = [text]

        if attrs['forms']:
            parts.append('form ' + ' '.join(attrs['forms']))
        if attrs['colors']:
            parts.append('color ' + ' '.join(attrs['colors']))
        if attrs['categories']:
            parts.append('category ' + ' '.join(attrs['categories']))
        if attrs['shapes']:
            parts.append('shape ' + ' '.join(attrs['shapes']))

        return ' '.join(parts)

    def assess_completeness(self, text: str) -> dict:
        """Assess whether description has enough information for identification."""
        # Handle Arabic text
        if is_arabic(text):
            attrs = extract_arabic_attributes(text)
        else:
            attrs = self.extract_attributes(text)
        words = text.split()

        score = 0
        missing = []

        if attrs['colors']:
            score += 25
        else:
            missing.append('color — اللون (e.g., white أبيض, blue أزرق, red أحمر)')

        if attrs['shapes'] or attrs['forms']:
            score += 25
        else:
            missing.append('shape or form — الشكل (e.g., round دائري, tablet قرص, capsule كبسولة)')

        if attrs['categories'] or len(words) >= 4:
            score += 30
        else:
            missing.append('intended use — الاستخدام (e.g., diabetes سكري, blood pressure ضغط, pain ألم)')

        if len(words) >= 3:
            score += 20

        is_complete = score >= 50
        # A single word is NOT ambiguous if it looks like a drug name
        # (starts with uppercase, contains digits, or is Arabic text).
        # Only flag truly vague single-character or empty queries.
        if len(words) == 1:
            w = words[0]
            is_ambiguous = len(w) < 3
        else:
            is_ambiguous = len(words) < 1

        return {
            'score': score,
            'is_complete': is_complete,
            'is_ambiguous': is_ambiguous,
            'missing': missing,
            'attributes': attrs,
        }

    def build_arabic_query(self, text: str) -> str:
        """Build an enriched English query from Arabic text for TF-IDF matching."""
        attrs = extract_arabic_attributes(text)
        translated = translate_arabic_query(text)
        parts = [translated] if translated and translated != text else []

        if attrs['forms']:
            parts.append('form ' + ' '.join(attrs['forms']))
        if attrs['colors']:
            parts.append('color ' + ' '.join(attrs['colors']))
        if attrs['categories']:
            parts.append('category ' + ' '.join(attrs['categories']))
        if attrs['shapes']:
            parts.append('shape ' + ' '.join(attrs['shapes']))

        return ' '.join(parts) if parts else translated
