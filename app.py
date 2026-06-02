"""
CAPSULE - AI-Powered Medication Identification System
Flask Web Application — Production Ready
Taif University, Department of Computer Science, Fall 2025
"""

import os
import json
import uuid
from datetime import datetime, date
from werkzeug.utils import secure_filename
import pandas as pd

from flask import (Flask, render_template, request, redirect,
                   url_for, flash, jsonify, session, abort)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (LoginManager, UserMixin, login_user,
                         logout_user, login_required, current_user)
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'capsule-taif-university-2025-secret'
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(BASE_DIR, "capsule_database.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {'connect_args': {'timeout': 30}}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
# ---------------------------------------------------------------------------
# Security Configuration (Note 14: Data Protection & Privacy)
# ---------------------------------------------------------------------------
app.config['SESSION_COOKIE_HTTPONLY'] = True       # Prevent JS access to session cookie
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'     # CSRF mitigation
app.config['PERMANENT_SESSION_LIFETIME'] = 86400   # 24-hour idle timeout (seconds)
# NOTE: In production, set SESSION_COOKIE_SECURE=True and serve over HTTPS (TLS 1.2+)
# NOTE: Production deployment should use disk-level encryption or PostgreSQL pgcrypto
#       for data-at-rest protection. This prototype has not been assessed against
#       Saudi PDPL or HIPAA. A formal privacy impact assessment is required before
#       any clinical deployment.
# ---------------------------------------------------------------------------
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'images', 'pills', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'


# ---------------------------------------------------------------------------
# Security Headers (Note 14: Transport & Browser Security)
# ---------------------------------------------------------------------------
@app.after_request
def set_security_headers(response):
    """Apply security headers to all responses."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    # In production with HTTPS, uncomment:
    # response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    return response

# Ensure AI_models directory is on the Python path
import sys as _sys
_AI_MODELS_DIR = os.path.join(BASE_DIR, 'AI_models')
if _AI_MODELS_DIR not in _sys.path:
    _sys.path.insert(0, _AI_MODELS_DIR)

# Lazy-load AI model to avoid blocking startup
_ai_model = None


def get_ai_model():
    global _ai_model
    if _ai_model is None:
        from ai_model import CapsuleAIModel
        _ai_model = CapsuleAIModel()
        # Augment the tiny 9-drug training CSV with the full SQLite catalog
        try:
            drugs = Medication.query.filter_by(is_active=True).all()
            _ai_model.augment_drug_database([{
                'drug_name':           d.drug_name,
                'generic_name':        d.generic_name,
                'dosage_form':         d.dosage_form,
                'category':            d.category,
                'color':               d.color,
                'shape':               d.shape,
                'strength':            d.strength,
                'strength_unit':       d.strength_unit,
                'description':         d.description,
                'manufacturer':        d.manufacturer,
                'admin_route':         d.admin_route,
                'price':               d.price,
                'storage':             d.storage,
                'legal_class':         d.legal_class,
                'drug_name_ar':        d.drug_name_ar,
                'generic_name_ar':     d.generic_name_ar,
                'indications':         d.indications,
                'indications_ar':      d.indications_ar,
                'side_effects':        d.side_effects,
                'side_effects_ar':     d.side_effects_ar,
                'contraindications':   d.contraindications,
                'contraindications_ar': d.contraindications_ar,
            } for d in drugs])
        except Exception as _e:
            print(f"[CAPSULE AI] DB augment skipped (no app context yet): {_e}")
    return _ai_model


# ---------------------------------------------------------------------------
# Database Models
# ---------------------------------------------------------------------------

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), default='patient')        # admin|doctor|pharmacist|patient
    full_name = db.Column(db.String(120), default='')
    hospital = db.Column(db.String(120), default='')
    specialty = db.Column(db.String(80), default='')
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)

    searches = db.relationship('SearchHistory', backref='user', lazy=True,
                               cascade='all, delete-orphan')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    @property
    def role_label(self):
        labels = {'admin': 'Administrator', 'doctor': 'Doctor',
                  'pharmacist': 'Pharmacist', 'patient': 'Patient'}
        return labels.get(self.role, 'Patient')

    @property
    def dashboard_url(self):
        routes = {'admin': 'admin_dashboard', 'doctor': 'doctor_dashboard',
                  'pharmacist': 'pharmacist_dashboard', 'patient': 'patient_dashboard'}
        return url_for(routes.get(self.role, 'patient_dashboard'))


class Medication(db.Model):
    __tablename__ = 'medications'
    id = db.Column(db.Integer, primary_key=True)
    reg_no = db.Column(db.String(20), unique=True, nullable=True)
    drug_name = db.Column(db.String(120), nullable=False)
    drug_name_ar = db.Column(db.String(120), default='')
    generic_name = db.Column(db.String(120), default='')
    generic_name_ar = db.Column(db.String(120), default='')
    strength = db.Column(db.String(50), default='')
    strength_unit = db.Column(db.String(20), default='')
    admin_route = db.Column(db.String(80), default='')
    dosage_form = db.Column(db.String(80), default='')
    package_size = db.Column(db.String(20), default='')
    package_type = db.Column(db.String(50), default='')
    legal_class = db.Column(db.String(50), default='')
    drug_type = db.Column(db.String(50), default='')
    shelf_life = db.Column(db.Integer, default=24)
    storage = db.Column(db.String(200), default='')
    price = db.Column(db.String(20), default='')
    manufacturer = db.Column(db.String(120), default='')
    color = db.Column(db.String(50), default='')
    shape = db.Column(db.String(50), default='')
    category = db.Column(db.String(80), default='')
    description = db.Column(db.Text, default='')
    # Bilingual / clinical fields from capsule_dataset.csv
    indications = db.Column(db.Text, default='')
    indications_ar = db.Column(db.Text, default='')
    side_effects = db.Column(db.Text, default='')
    side_effects_ar = db.Column(db.Text, default='')
    contraindications = db.Column(db.Text, default='')
    contraindications_ar = db.Column(db.Text, default='')
    salt_composition = db.Column(db.String(200), default='')
    imprint_code = db.Column(db.String(50), default='')
    image_path = db.Column(db.String(300), default='')
    # Long-term effect tracking
    long_term_effect = db.Column(db.Boolean, default=False)
    long_term_description = db.Column(db.Text, default='')
    long_term_duration = db.Column(db.String(200), default='')
    requires_warning = db.Column(db.Boolean, default=False)
    warning_text = db.Column(db.Text, default='')
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @property
    def strength_label(self):
        return f"{self.strength} {self.strength_unit}".strip()

    @property
    def display_name(self):
        return self.drug_name.title()


class SearchHistory(db.Model):
    __tablename__ = 'search_history'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    description = db.Column(db.Text, nullable=False)
    results_json = db.Column(db.Text, default='[]')
    top_result = db.Column(db.String(120), default='')
    top_confidence = db.Column(db.Float, default=0.0)
    result_count = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    @property
    def results(self):
        try:
            return json.loads(self.results_json)
        except Exception:
            return []

    @property
    def time_ago(self):
        delta = datetime.utcnow() - self.created_at
        if delta.days > 0:
            return f"{delta.days}d ago"
        hours = delta.seconds // 3600
        if hours > 0:
            return f"{hours}h ago"
        minutes = delta.seconds // 60
        return f"{minutes}m ago"


class SystemLog(db.Model):
    __tablename__ = 'system_logs'
    id = db.Column(db.Integer, primary_key=True)
    action = db.Column(db.String(100), nullable=False)
    user_id = db.Column(db.Integer, nullable=True)
    username = db.Column(db.String(80), default='system')
    details = db.Column(db.Text, default='')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class PatientMedication(db.Model):
    """Medication list for a patient (current or past)."""
    __tablename__ = 'patient_medications'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    medication_id = db.Column(db.Integer, db.ForeignKey('medications.id'), nullable=True)
    # Allow free-text when drug not in DB
    medication_name = db.Column(db.String(200), nullable=False)
    status = db.Column(db.String(20), default='current')   # current | previous
    start_date = db.Column(db.Date, nullable=True)
    end_date = db.Column(db.Date, nullable=True)
    dosage = db.Column(db.String(100), default='')          # e.g. "500mg twice daily"
    prescribed_by = db.Column(db.String(200), default='')
    notes = db.Column(db.Text, default='')
    added_at = db.Column(db.DateTime, default=datetime.utcnow)

    medication = db.relationship('Medication', backref='patient_entries', lazy=True)

    @property
    def long_term_flag(self):
        if self.medication:
            return self.medication.long_term_effect
        from long_term_effects import LONG_TERM_DRUG_EFFECTS
        name = self.medication_name.lower()
        return any(k.lower() in name for k in LONG_TERM_DRUG_EFFECTS)

    @property
    def long_term_info(self):
        if self.medication and self.medication.long_term_effect:
            return {
                'description': self.medication.long_term_description or '',
                'duration': self.medication.long_term_duration or '',
                'warning': self.medication.warning_text or '',
                'severity': 'high' if self.medication.requires_warning else 'low',
                'effect_summary': '',
                'monitoring': [],
            }
        try:
            from long_term_effects import LONG_TERM_DRUG_EFFECTS
            name = self.medication_name.lower()
            for k, v in LONG_TERM_DRUG_EFFECTS.items():
                if k.lower() in name or name in k.lower():
                    months = v.get('min_duration_months', 0)
                    return {
                        'description': v.get('description', ''),
                        'duration': f'{months} months+' if months else '',
                        'warning': v.get('patient_guidance', ''),
                        'severity': v.get('severity', 'low'),
                        'effect_summary': v.get('effect_summary', ''),
                        'monitoring': v.get('monitoring', []),
                    }
        except Exception:
            pass
        return None


class DrugInteraction(db.Model):
    """Known interaction between two drug name patterns."""
    __tablename__ = 'drug_interactions'
    id = db.Column(db.Integer, primary_key=True)
    drug_a_name = db.Column(db.String(200), nullable=False)
    drug_b_name = db.Column(db.String(200), nullable=False)
    risk_level = db.Column(db.String(10), nullable=False)   # high | medium | low
    interaction_description = db.Column(db.Text, nullable=False)
    mechanism = db.Column(db.Text, default='')
    patient_guidance = db.Column(db.Text, nullable=False)
    clinical_note = db.Column(db.Text, default='')
    source = db.Column(db.String(100), default='system')


class DispenseLog(db.Model):
    """Record of pharmacist dispensing a medication."""
    __tablename__ = 'dispense_logs'
    id = db.Column(db.Integer, primary_key=True)
    pharmacist_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    patient_name = db.Column(db.String(200), default='')
    medication_id = db.Column(db.Integer, db.ForeignKey('medications.id'), nullable=True)
    medication_name = db.Column(db.String(200), default='')
    dispensed_at = db.Column(db.DateTime, default=datetime.utcnow)
    long_term_warning_shown = db.Column(db.Boolean, default=False)
    patient_informed = db.Column(db.Boolean, default=False)
    notes = db.Column(db.Text, default='')

    medication = db.relationship('Medication', backref='dispense_records', lazy=True)


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


def role_required(*roles):
    """Decorator that restricts a view to specific user roles."""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if not current_user.is_authenticated:
                return redirect(url_for('login'))
            if current_user.role not in roles:
                flash('You do not have permission to access this page.', 'danger')
                return redirect(url_for('index'))
            return f(*args, **kwargs)
        return decorated
    return decorator


def log_action(action, details=''):
    try:
        log = SystemLog(
            action=action,
            user_id=current_user.id if current_user.is_authenticated else None,
            username=current_user.username if current_user.is_authenticated else 'system',
            details=details,
        )
        db.session.add(log)
        db.session.commit()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Public routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('landing.html')


@app.route('/capsule_landing')
def capsule_landing():
    return render_template('landing.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(current_user.dashboard_url)

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        remember = request.form.get('remember', False)

        user = User.query.filter(
            (User.username == username) | (User.email == username)
        ).first()

        if user and user.check_password(password):
            if not user.is_active:
                flash('Your account has been deactivated. Contact admin.', 'danger')
                return render_template('login.html')
            login_user(user, remember=bool(remember))
            user.last_login = datetime.utcnow()
            db.session.commit()
            log_action('login', f'User {username} logged in')
            next_page = request.args.get('next')
            if next_page:
                return redirect(next_page)
            return redirect(user.dashboard_url)
        else:
            flash('Invalid username or password.', 'danger')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    log_action('logout')
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(current_user.dashboard_url)

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm_password', '')
        role = request.form.get('role', 'patient')
        full_name = request.form.get('full_name', '').strip()
        hospital = request.form.get('hospital', '').strip()

        # Validate
        errors = []
        if len(username) < 3:
            errors.append('Username must be at least 3 characters.')
        if len(password) < 6:
            errors.append('Password must be at least 6 characters.')
        if password != confirm:
            errors.append('Passwords do not match.')
        if User.query.filter_by(username=username).first():
            errors.append('Username already taken.')
        if User.query.filter_by(email=email).first():
            errors.append('Email already registered.')
        if role not in ('doctor', 'pharmacist', 'patient'):
            role = 'patient'

        if errors:
            for e in errors:
                flash(e, 'danger')
            return render_template('register.html')

        user = User(username=username, email=email, role=role,
                    full_name=full_name, hospital=hospital)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        log_action('register', f'New user {username} registered as {role}')
        login_user(user)
        flash(f'Welcome, {full_name or username}! Your account has been created.', 'success')
        return redirect(user.dashboard_url)

    return render_template('register.html')


# ---------------------------------------------------------------------------
# Dashboard routes
# ---------------------------------------------------------------------------

@app.route('/dashboard')
@login_required
def dashboard():
    if current_user.role == 'admin':
        return redirect(url_for('admin_dashboard'))
    if current_user.role == 'doctor':
        return redirect(url_for('doctor_dashboard'))
    if current_user.role == 'pharmacist':
        return redirect(url_for('pharmacist_dashboard'))
    if current_user.role == 'patient':
        return redirect(url_for('patient_dashboard'))

    recent = SearchHistory.query.filter_by(user_id=current_user.id)\
        .order_by(SearchHistory.created_at.desc()).limit(5).all()
    total_searches = SearchHistory.query.filter_by(user_id=current_user.id).count()
    return render_template('dashboard.html', recent=recent, total_searches=total_searches)


@app.route('/doctor/dashboard')
@login_required
@role_required('doctor', 'admin')
def doctor_dashboard():
    recent = SearchHistory.query.filter_by(user_id=current_user.id)\
        .order_by(SearchHistory.created_at.desc()).limit(8).all()
    total_searches = SearchHistory.query.filter_by(user_id=current_user.id).count()
    drugs = Medication.query.filter_by(is_active=True).limit(20).all()

    # Patient tracking — all registered patients in the system
    patients = User.query.filter(
        User.role.in_(['patient'])
    ).order_by(User.created_at.desc()).all()

    # Annotate each patient with medication counts and risk flags
    patient_data = []
    for p in patients:
        meds = PatientMedication.query.filter_by(patient_id=p.id, status='current').all()
        lt_count = sum(1 for m in meds if m.long_term_flag)
        patient_data.append({
            'user': p,
            'current_meds': len(meds),
            'lt_warnings': lt_count,
            'is_high_risk': lt_count > 0,
        })

    # Summary stats
    total_patients = len(patients)
    high_risk_count = sum(1 for pd in patient_data if pd['is_high_risk'])

    return render_template('doctor_dashboard.html',
                           recent=recent,
                           total_searches=total_searches,
                           drugs=drugs,
                           patient_data=patient_data,
                           total_patients=total_patients,
                           high_risk_count=high_risk_count)


@app.route('/pharmacist/dashboard')
@login_required
@role_required('pharmacist', 'admin')
def pharmacist_dashboard():
    drugs = Medication.query.filter_by(is_active=True)\
        .order_by(Medication.drug_name).all()
    total_drugs = len(drugs)
    categories = db.session.query(Medication.category)\
        .filter(Medication.is_active == True)\
        .distinct().all()

    # ── Group 162 granular categories into major therapeutic areas ──
    DISEASE_GROUPS = [
        {'name': 'Pain & Inflammation',
         'icon': 'fa-fire-flame-curved', 'color': '#ef4444',
         'keywords': ['pain', 'nsaid', 'analgesic', 'anti-inflammatory', 'antipyretic', 'cox-2', 'antigout', 'xanthine oxidase']},
        {'name': 'Antibiotics & Anti-infectives',
         'icon': 'fa-shield-virus', 'color': '#059669',
         'keywords': ['antibiotic', 'antifungal', 'antiviral', 'antimalarial', 'antiparasitic', 'penicillin',
                      'cephalosporin', 'macrolide', 'fluoroquinolone', 'carbapenem', 'aminoglycoside',
                      'glycopeptide', 'lincosamide', 'tetracycline', 'oxazolidinone', 'nitrofuran',
                      'nitroimidazole', 'sulfonamide', 'antiseptic', 'fusidane', 'pseudomonic']},
        {'name': 'Cardiovascular',
         'icon': 'fa-heart-pulse', 'color': '#dc2626',
         'keywords': ['hypertension', 'ace inhibitor', 'arb', 'angiotensin', 'beta blocker', 'calcium channel',
                      'ccb', 'nitrate', 'cardiac', 'diuretic', 'vasopressor', 'centrally acting antihypertensive',
                      'alpha/beta', 'alpha-1 adrenergic']},
        {'name': 'Diabetes & Metabolism',
         'icon': 'fa-droplet', 'color': '#7c3aed',
         'keywords': ['diabetes', 'biguanide', 'sulfonylurea', 'sglt-2', 'dpp-4', 'glp-1', 'insulin',
                      'thiazolidinedione']},
        {'name': 'Gastrointestinal',
         'icon': 'fa-kit-medical', 'color': '#d97706',
         'keywords': ['gastrointestinal', 'proton pump', 'ppi', 'h2 receptor', 'antacid', 'alginate',
                      'antiemetic', 'prokinetic', 'antidiarrheal', 'laxative', 'antispasmodic']},
        {'name': 'Mental Health',
         'icon': 'fa-brain', 'color': '#8b5cf6',
         'keywords': ['ssri', 'snri', 'antipsychotic', 'benzodiazepine', 'anxiolytic', 'adhd',
                      'melatonin', 'serotonin']},
        {'name': 'Respiratory & Allergy',
         'icon': 'fa-lungs', 'color': '#0ea5e9',
         'keywords': ['respiratory', 'asthma', 'ics', 'laba', 'lama', 'bronchodilator', 'leukotriene',
                      'inhaled', 'antihistamine', 'intranasal', 'cold', 'flu', 'decongestant',
                      'expectorant', 'anticholinergic']},
        {'name': 'Blood Thinners',
         'icon': 'fa-syringe', 'color': '#e11d48',
         'keywords': ['anticoagulant', 'doac', 'heparin', 'antiplatelet', 'thrombin', 'factor xa']},
        {'name': 'Cholesterol & Lipids',
         'icon': 'fa-chart-line', 'color': '#f59e0b',
         'keywords': ['cholesterol', 'statin', 'hmg-coa']},
        {'name': 'Neurology & Epilepsy',
         'icon': 'fa-head-side-virus', 'color': '#6366f1',
         'keywords': ['antiepileptic', 'anticonvulsant', 'neurological', 'alzheimer', 'neuropathic',
                      'gaba', 'sv2a', 'sodium channel']},
        {'name': 'Hormones & Endocrine',
         'icon': 'fa-dna', 'color': '#14b8a6',
         'keywords': ['corticosteroid', 'thyroid', 'antithyroid', 'mineralocorticoid', 'serm',
                      'estrogen']},
        {'name': 'Dermatology',
         'icon': 'fa-hand-dots', 'color': '#f97316',
         'keywords': ['topical', 'retinoid', 'vasodilator (hair', 'local anesthetic']},
        {'name': 'Vitamins & Supplements',
         'icon': 'fa-capsules', 'color': '#22c55e',
         'keywords': ['vitamin', 'supplement', 'mineral', 'iron', 'calcium', 'zinc', 'biotin', 'folate',
                      'multivitamin', 'omega-3', 'nutrition', 'electrolyte', 'magnesium',
                      'parenteral nutrition']},
        {'name': 'Urology',
         'icon': 'fa-mars', 'color': '#3b82f6',
         'keywords': ['5-alpha reductase', 'antimuscarinic', 'pde-5', 'urinary']},
        {'name': 'Ophthalmology',
         'icon': 'fa-eye', 'color': '#06b6d4',
         'keywords': ['eye drop', 'ocular', 'artificial tear', 'prostaglandin analogue eye']},
        {'name': 'Bone & Joint',
         'icon': 'fa-bone', 'color': '#a3a3a3',
         'keywords': ['bisphosphonate', 'dmard', 'antimetabolite']},
    ]

    # Build grouped categories: each group gets a list of matching DB categories
    all_cats = [c[0] for c in categories if c[0]]
    disease_groups = []
    assigned = set()
    for g in DISEASE_GROUPS:
        matched = []
        for cat in all_cats:
            cl = cat.lower()
            if any(kw in cl for kw in g['keywords']):
                matched.append(cat)
                assigned.add(cat)
        if matched:
            # Count drugs in this group
            drug_count = Medication.query.filter(
                Medication.is_active == True,
                Medication.category.in_(matched)
            ).count()
            disease_groups.append({
                'name': g['name'], 'icon': g['icon'], 'color': g['color'],
                'categories': matched, 'drug_count': drug_count
            })
    # Collect unassigned categories into "Other"
    other_cats = [c for c in all_cats if c not in assigned]
    if other_cats:
        other_count = Medication.query.filter(
            Medication.is_active == True,
            Medication.category.in_(other_cats)
        ).count()
        disease_groups.append({
            'name': 'Other', 'icon': 'fa-pills', 'color': '#6b7280',
            'categories': other_cats, 'drug_count': other_count
        })

    return render_template('pharmacist_dashboard.html', drugs=drugs,
                           total_drugs=total_drugs, categories=categories,
                           disease_groups=disease_groups)


@app.route('/admin')
@login_required
@role_required('admin')
def admin_dashboard():
    from collections import Counter
    from datetime import timedelta

    total_users = User.query.count()
    total_doctors = User.query.filter_by(role='doctor').count()
    total_pharmacists = User.query.filter_by(role='pharmacist').count()
    total_patients = User.query.filter(User.role.in_(['patient'])).count()
    total_drugs = Medication.query.filter_by(is_active=True).count()
    total_searches = SearchHistory.query.count()
    recent_users = User.query.order_by(User.created_at.desc()).limit(5).all()
    recent_searches = SearchHistory.query.order_by(SearchHistory.created_at.desc()).limit(10).all()
    recent_logs = SystemLog.query.order_by(SystemLog.created_at.desc()).limit(20).all()
    users = User.query.order_by(User.created_at.desc()).all()
    drugs = Medication.query.order_by(Medication.drug_name).all()
    model_info = get_ai_model().get_model_info()

    # ── Analytics ─────────────────────────────────────────────────────────────
    # Top searched medications (last 200 searches)
    all_searches = SearchHistory.query.filter(
        SearchHistory.top_result != ''
    ).order_by(SearchHistory.created_at.desc()).limit(200).all()

    result_counter = Counter(s.top_result for s in all_searches if s.top_result)
    top_medications = result_counter.most_common(8)  # [(name, count), ...]

    # Searches per day — last 7 days
    today = datetime.utcnow().date()
    daily_counts = []
    for i in range(6, -1, -1):
        day = today - timedelta(days=i)
        count = SearchHistory.query.filter(
            db.func.date(SearchHistory.created_at) == day
        ).count()
        daily_counts.append({'day': day.strftime('%a'), 'count': count})

    # Confidence distribution buckets
    high_conf = SearchHistory.query.filter(SearchHistory.top_confidence >= 0.8).count()
    med_conf = SearchHistory.query.filter(
        SearchHistory.top_confidence >= 0.5, SearchHistory.top_confidence < 0.8
    ).count()
    low_conf = SearchHistory.query.filter(
        SearchHistory.top_confidence > 0, SearchHistory.top_confidence < 0.5
    ).count()

    # User role distribution for chart
    role_counts = {
        'Doctors': total_doctors,
        'Pharmacists': total_pharmacists,
        'Patients': total_patients,
        'Admins': User.query.filter_by(role='admin').count(),
    }

    return render_template('admin_dashboard.html',
                           total_users=total_users, total_doctors=total_doctors,
                           total_pharmacists=total_pharmacists, total_patients=total_patients,
                           total_drugs=total_drugs, total_searches=total_searches,
                           recent_users=recent_users, recent_searches=recent_searches,
                           recent_logs=recent_logs, users=users, drugs=drugs,
                           model_info=model_info,
                           top_medications=top_medications,
                           daily_counts=daily_counts,
                           high_conf=high_conf, med_conf=med_conf, low_conf=low_conf,
                           role_counts=role_counts)


# ---------------------------------------------------------------------------
# Identification routes
# ---------------------------------------------------------------------------

@app.route('/identify')
@login_required
def identify():
    recent = SearchHistory.query.filter_by(user_id=current_user.id)\
        .order_by(SearchHistory.created_at.desc()).limit(5).all()
    return render_template('identify.html', recent=recent)


def _med_to_result(drug: 'Medication', score: float = 0.5) -> dict:
    """Convert a Medication ORM object to a standard result dict."""
    pct = round(score * 100, 1)
    level = 'high' if pct >= 80 else ('medium' if pct >= 60 else 'low')
    return {
        'rank': 1,
        'drug_id': drug.id,
        'medication': drug.drug_name,
        'generic_name': drug.generic_name or '',
        'strength': drug.strength_label,
        'dosage_form': drug.dosage_form or '',
        'category': drug.category or '',
        'color': drug.color or '',
        'shape': drug.shape or '',
        'route': drug.admin_route or '',
        'manufacturer': drug.manufacturer or '',
        'price': drug.price or '',
        'storage': drug.storage or '',
        'legal_classification': drug.legal_class or '',
        'description': drug.description or '',
        'confidence_score': round(score, 4),
        'confidence_pct': pct,
        'confidence_level': level,
        'matched_attributes': {},
    }


def _score_result(drug: 'Medication', color_f: str, shape_f: str,
                  use_f: str, packaging_f: str, rank: int = 0) -> dict:
    """Score a DB result by fixed per-filter weights.
    Color: 10 %, Shape: 10 %, Packaging: 10 %, Use: 10 %.
    Each active filter adds its weight on match; unmatched filters add 0.
    The remaining weight (100 % minus active filters) is a base score of 1.0,
    so only active filters can lower the total.
    """
    WEIGHTS = {'color': 0.10, 'shape': 0.10, 'packaging': 0.10, 'use': 0.10}

    drug_color = (drug.color       or '').lower()
    drug_shape = (drug.shape       or '').lower()
    drug_cat   = (drug.category    or '').lower()
    drug_form  = (drug.dosage_form or '').lower() + ' ' + (drug.package_type or '').lower()

    active_weight = 0.0
    filter_score  = 0.0
    matched: list[str] = []

    if color_f:
        active_weight += WEIGHTS['color']
        if color_f in drug_color:
            filter_score += WEIGHTS['color']
            matched.append('color')
    if shape_f:
        active_weight += WEIGHTS['shape']
        if shape_f in drug_shape:
            filter_score += WEIGHTS['shape']
            matched.append('shape')
    if packaging_f:
        active_weight += WEIGHTS['packaging']
        if packaging_f in drug_form:
            filter_score += WEIGHTS['packaging']
            matched.append('packaging')
    if use_f:
        active_weight += WEIGHTS['use']
        if use_f in drug_cat:
            filter_score += WEIGHTS['use']
            matched.append('use')

    base_weight = 1.0 - active_weight
    normalised  = base_weight + filter_score   # max = 1.0 when all match
    # Small rank-based penalty so rank-1 always scores highest
    normalised = max(0.20, normalised - rank * 0.02)

    result = _med_to_result(drug, round(normalised, 4))
    result['matched_attributes'] = matched
    return result

def _inject_drug_ids(results: list) -> None:
    """
    Attach the correct Medication.id to each result.
    Priority:
    1) Exact displayed drug name, مثل Adol → Adol
    2) Cleaned displayed name without parentheses
    3) Existing drug_id if already valid
    4) Generic-name fallback only as last option
    """
    import re

    if not results:
        return

    for r in results:
        if not isinstance(r, dict):
            continue

        displayed_name = (
            r.get('medication')
            or r.get('drug_name')
            or r.get('name')
            or ''
        )

        displayed_name = str(displayed_name).strip()
        cleaned_name = re.sub(r'\s*\([^)]*\)\s*$', '', displayed_name).strip()

        db_med = None

        # 1) Exact match by displayed drug name
        if displayed_name:
            db_med = Medication.query.filter(
                db.func.lower(Medication.drug_name) == displayed_name.lower()
            ).first()

        # 2) Exact match by cleaned name
        if db_med is None and cleaned_name and cleaned_name != displayed_name:
            db_med = Medication.query.filter(
                db.func.lower(Medication.drug_name) == cleaned_name.lower()
            ).first()

        # 3) If exact name found, override wrong ID
        if db_med:
            r['drug_id'] = db_med.id
            continue

        # 4) If no exact name found, keep existing ID if it exists
        if r.get('drug_id'):
            continue

        # 5) Last fallback: generic name
        generic_name = (
            r.get('generic_name')
            or r.get('generic')
            or ''
        )

        generic_name = str(generic_name).strip()

        if generic_name:
            db_med = Medication.query.filter(
                db.func.lower(Medication.generic_name) == generic_name.lower()
            ).first()

            if db_med:
                r['drug_id'] = db_med.id
            else:
                r['drug_id'] = None



# ---------------------------------------------------------------------------
# Query-quality helpers for medication identification
# ---------------------------------------------------------------------------

def _normalize_query_token(token: str) -> str:
    """Normalize Arabic/common words to the English values used in the DB."""
    token = (token or '').lower().strip()
    mapping = {
        # Arabic colors
        'ابيض': 'white', 'أبيض': 'white', 'بيضاء': 'white',
        'اصفر': 'yellow', 'أصفر': 'yellow', 'صفراء': 'yellow',
        'ازرق': 'blue', 'أزرق': 'blue', 'زرقاء': 'blue',
        'احمر': 'red', 'أحمر': 'red', 'حمراء': 'red',
        'وردي': 'pink',
        'برتقالي': 'orange', 'برتقالية': 'orange',
        'اخضر': 'green', 'أخضر': 'green', 'خضراء': 'green',

        # Arabic shapes
        'دائري': 'round', 'دائرية': 'round',
        'بيضاوي': 'oval', 'بيضاوية': 'oval',
        'طويل': 'oblong', 'طويلة': 'oblong',
        'صغير': 'small', 'صغيرة': 'small',
        'كبير': 'large', 'كبيرة': 'large',

        # Arabic dosage forms
        'حبة': 'tablet', 'حبوب': 'tablet', 'قرص': 'tablet', 'أقراص': 'tablet',
        'كبسولة': 'capsule', 'كبسولات': 'capsule',
        'شراب': 'syrup', 'سيرب': 'syrup',
        'كريم': 'cream', 'مرهم': 'ointment', 'جل': 'gel',
        'حقنة': 'injection', 'إبرة': 'injection',
        'قطرة': 'drops',
        'دواء': 'medicine', 'علاج': 'medicine',

        # Arabic symptoms/use words
        'حرارة': 'fever', 'حمى': 'fever',
        'صداع': 'headache',
        'ألم': 'pain', 'الم': 'pain',
        'زكام': 'cold',
        'انفلونزا': 'flu', 'إنفلونزا': 'flu',
        'سكري': 'diabetes', 'سكر': 'sugar',
        'ضغط': 'pressure',
        'كوليسترول': 'cholesterol',
        'التهاب': 'infection', 'مضاد': 'antibiotic',
        'حساسية': 'allergy',
        'ربو': 'asthma',
        'حموضة': 'acidity', 'معدة': 'stomach',
        'كحة': 'cough', 'سعال': 'cough',
        'سيلان': 'runny', 'أنف': 'nose', 'الأنف': 'nose',
        'احتقان': 'congestion',
        'مناعة': 'immunity', 'مكمل': 'supplement', 'زنك': 'zinc',
        'تنظيم': 'control', 'السكر': 'sugar', 'الدم': 'blood',
    }
    return mapping.get(token, token)


def _tokenize_query(text: str) -> list[str]:
    import re
    return [_normalize_query_token(t) for t in re.findall(r"\b\w+\b", (text or '').lower().strip())]


def _extract_query_terms(description='', color_filter='', shape_filter='', packaging_filter='', use_filter='') -> dict:
    """Extract recognized color/shape/form/use terms from description and filters."""
    color_terms = {
        'white', 'yellow', 'blue', 'red', 'pink', 'orange', 'green', 'brown', 'black',
        'gold', 'amber'
    }
    shape_terms = {'round', 'oval', 'oblong', 'circular', 'square', 'small', 'large'}
    form_terms = {
        'pill', 'tablet', 'capsule', 'caplet', 'medicine', 'drug', 'medication',
        'syrup', 'liquid', 'solution', 'suspension', 'drops', 'cream', 'ointment',
        'gel', 'lotion', 'injection', 'vial', 'ampoule', 'mouthwash'
    }
    use_terms = {
        'fever', 'headache', 'pain', 'flu', 'cold', 'diabetes', 'sugar', 'pressure',
        'hypertension', 'cholesterol', 'infection', 'antibiotic', 'allergy', 'asthma',
        'heartburn', 'acidity', 'stomach', 'cough', 'runny', 'nose', 'nasal',
        'congestion', 'immunity', 'immune', 'zinc', 'supplement', 'blood', 'glucose',
        'control', 'relief', 'treat', 'treatment', 'remedy', 'cold', 'flu',
        'mouth', 'oral', 'hygiene', 'sore', 'throat',
        'antiseptic', 'gargle', 'tonsillitis', 'hexetidine', 'bactidol'
    }

    terms = {'colors': [], 'shapes': [], 'forms': [], 'uses': []}
    for token in _tokenize_query(description):
        if token in color_terms:
            terms['colors'].append(token)
        elif token in shape_terms:
            terms['shapes'].append(token)
        elif token in form_terms:
            # Normalize generic pill into tablet for DB searching.
            terms['forms'].append('tablet' if token == 'pill' else token)
        elif token in use_terms:
            terms['uses'].append(token)

    if color_filter:
        terms['colors'].append(_normalize_query_token(color_filter))
    if shape_filter:
        terms['shapes'].append(_normalize_query_token(shape_filter))
    if packaging_filter:
        pack = _normalize_query_token(packaging_filter)
        terms['forms'].append('tablet' if pack == 'pill' else pack)
    if use_filter:
        terms['uses'].append(_normalize_query_token(use_filter))

    for key, values in terms.items():
        seen = set()
        unique = []
        for value in values:
            if value and value not in seen:
                seen.add(value)
                unique.append(value)
        terms[key] = unique
    return terms


def _exact_drug_match(description: str):
    """Return a Medication if the description is an exact drug/generic name."""
    q = (description or '').strip()
    if not q:
        return None
    return Medication.query.filter(
        Medication.is_active == True,
        (
            (Medication.drug_name.ilike(q)) |
            (Medication.generic_name.ilike(q)) |
            (Medication.drug_name_ar.ilike(q)) |
            (Medication.generic_name_ar.ilike(q))
        )
    ).first()


def _has_any_db_text_match(description: str) -> bool:
    """Check whether the text appears anywhere meaningful in the DB."""
    q = (description or '').strip()
    if not q:
        return False
    like = f'%{q[:80]}%'
    return Medication.query.filter(
        Medication.is_active == True,
        (
            (Medication.drug_name.ilike(like)) |
            (Medication.generic_name.ilike(like)) |
            (Medication.description.ilike(like)) |
            (Medication.category.ilike(like)) |
            (Medication.indications.ilike(like)) |
            (Medication.drug_name_ar.ilike(like)) |
            (Medication.generic_name_ar.ilike(like)) |
            (Medication.indications_ar.ilike(like))
        )
    ).limit(1).first() is not None


def _has_contradictory_dosage_forms(description: str) -> tuple[bool, list[str]]:
    """
    Detect contradictory dosage-form groups in the same user description.
    Example: cream + tablet, capsule + syrup, gel + pill.
    """
    tokens = set(_tokenize_query(description))
    form_groups = {
    'solid_oral': {
        'pill', 'tablet', 'capsule', 'caplet'
    },

    'liquid_oral': {
        'syrup', 'liquid', 'solution', 'suspension',
        'drops', 'mouthwash', 'gargle'
    },

    'topical': {
        'cream', 'ointment', 'gel', 'lotion'
    },

    'injection': {
        'injection', 'vial', 'ampoule'
    },

    'inhaled': {
        'inhaler'
    }
}
    
    found_groups = []
    for group, terms in form_groups.items():
        if tokens & terms:
            found_groups.append(group)
    return (len(found_groups) > 1, found_groups)


def _is_weak_meaningful_description(description: str) -> bool:
    """
    A weak meaningful description should return possible matches with low confidence,
    not high confidence and not no_match.
    Examples: red, fever, yellow oval tablet.
    """
    tokens = [t for t in _tokenize_query(description) if t not in {
        'for', 'used', 'use', 'to', 'and', 'or', 'with', 'the', 'a', 'an',
        'of', 'in', 'on', 'by', 'ل', 'لل', 'من', 'في', 'و', 'او', 'أو'
    }]
    if not tokens:
        return False

    terms = _extract_query_terms(description)
    recognized_count = sum(len(v) for v in terms.values())
    if recognized_count == 0:
        return False

    # Exact drug names should not be treated as weak descriptions.
    if _exact_drug_match(description):
        return False

    # One recognized clue, such as red or fever, is weak.
    if len(tokens) <= 1:
        return True

    # Physical-only clues are weak: yellow oval tablet.
    if recognized_count == len(tokens) and not terms['uses']:
        return True

    # Symptom/use-only clues are weak unless they are quite detailed.
    if recognized_count == len(tokens) and terms['uses'] and not (terms['colors'] or terms['shapes'] or terms['forms']):
        return len(tokens) <= 3

    # Few clues made only of weak terms are still weak.
    return recognized_count == len(tokens) and len(tokens) < 4


def _low_confidence_score_for_weak_match(drug, terms: dict, rank: int = 0) -> float:
    """Score broad/weak matches with intentionally low confidence."""
    drug_color = (drug.color or '').lower()
    drug_shape = (drug.shape or '').lower()
    drug_form = ((drug.dosage_form or '') + ' ' + (drug.package_type or '')).lower()
    drug_text = ' '.join([
        drug_color, drug_shape, drug_form,
        (drug.category or '').lower(),
        (drug.description or '').lower(),
        (drug.indications or '').lower(),
        (drug.generic_name or '').lower(),
        (drug.drug_name or '').lower(),
    ])

    matched = 0
    total = 0
    if terms.get('colors'):
        total += 1
        if any(v in drug_color or v in drug_text for v in terms['colors']):
            matched += 1
    if terms.get('shapes'):
        total += 1
        if any(v in drug_shape or v in drug_text for v in terms['shapes']):
            matched += 1
    if terms.get('forms'):
        total += 1
        if any(v in drug_form or v in drug_text for v in terms['forms']):
            matched += 1
    if terms.get('uses'):
        total += 1
        if any(v in drug_text for v in terms['uses']):
            matched += 1

    ratio = (matched / total) if total else 0.0
    score = 0.18 + (0.27 * ratio)

    if total <= 1:
        score = min(score, 0.32)
    if not terms.get('uses'):
        score = min(score, 0.42)

    score = max(0.12, score - (rank * 0.025))
    return round(score, 4)


def _weak_description_response(description='', color_filter='', shape_filter='', packaging_filter='', use_filter='', top_k=5):
    """
    Return possible matches for weak-but-meaningful input with low confidence.
    If the DB truly has no candidate for the recognized terms, return no_match.
    """
    terms = _extract_query_terms(description, color_filter, shape_filter, packaging_filter, use_filter)
    q = Medication.query.filter_by(is_active=True)

    # Strict search first.
    if terms['colors']:
        q = q.filter(Medication.color.ilike(f"%{terms['colors'][0]}%"))
    if terms['shapes']:
        q = q.filter(Medication.shape.ilike(f"%{terms['shapes'][0]}%"))
    if terms['forms']:
        q = q.filter(
            (Medication.dosage_form.ilike(f"%{terms['forms'][0]}%")) |
            (Medication.package_type.ilike(f"%{terms['forms'][0]}%"))
        )
    if terms['uses']:
        use = terms['uses'][0]
        q = q.filter(
            (Medication.category.ilike(f"%{use}%")) |
            (Medication.description.ilike(f"%{use}%")) |
            (Medication.indications.ilike(f"%{use}%")) |
            (Medication.generic_name.ilike(f"%{use}%"))
        )

    db_drugs = q.order_by(Medication.drug_name).limit(top_k).all()

    # Relaxed OR search if strict AND gives nothing.
    if not db_drugs:
        all_terms = terms['colors'] + terms['shapes'] + terms['forms'] + terms['uses']
        if all_terms:
            conditions = []
            for term in all_terms:
                conditions.extend([
                    Medication.color.ilike(f"%{term}%"),
                    Medication.shape.ilike(f"%{term}%"),
                    Medication.dosage_form.ilike(f"%{term}%"),
                    Medication.package_type.ilike(f"%{term}%"),
                    Medication.category.ilike(f"%{term}%"),
                    Medication.description.ilike(f"%{term}%"),
                    Medication.indications.ilike(f"%{term}%"),
                    Medication.generic_name.ilike(f"%{term}%"),
                    Medication.drug_name.ilike(f"%{term}%"),
                ])
            db_drugs = Medication.query.filter(
                Medication.is_active == True,
                db.or_(*conditions)
            ).order_by(Medication.drug_name).limit(top_k).all()

    if not db_drugs:
        return jsonify({
            'status': 'no_match',
            'results': [],
            'message': 'No medication found in the database for this description.',
            'query_specificity': {
                'level': 'weak',
                'score': 0.0,
                'reasons': ['weak_meaningful_description_no_db_candidate']
            },
            'query_info': {
                'original': description,
                'color': color_filter,
                'shape': shape_filter,
                'packaging': packaging_filter,
                'intended_use': use_filter,
                'recognized_terms': terms,
            }
        })

    results = []
    for i, drug in enumerate(db_drugs):
        score = _low_confidence_score_for_weak_match(drug, terms, i)
        r = _med_to_result(drug, score)
        r['rank'] = i + 1
        r['confidence_note'] = 'Low confidence: the description is broad and lacks enough identifying details.'
        r['matched_attributes'] = {'recognized_terms': terms}
        results.append(r)

    _inject_drug_ids(results)
    _inject_image_urls(results)
    return jsonify({
        'status': 'success',
        'results': results,
        'message': 'Possible matches are shown with low confidence because the description is general.',
        'query_specificity': {
            'level': 'weak',
            'score': 0.25,
            'reasons': ['weak_meaningful_description_low_confidence']
        },
        'query_info': {
            'original': description,
            'color': color_filter,
            'shape': shape_filter,
            'packaging': packaging_filter,
            'intended_use': use_filter,
            'recognized_terms': terms,
        }
    })


def _contradictory_description_response(description: str, groups: list[str]):
    return jsonify({
        'status': 'contradictory',
        'results': [],
        'message': 'The description contains conflicting dosage forms, such as cream and tablet together. Please correct the medication form.',
        'query_specificity': {
            'level': 'contradictory',
            'score': 0.0,
            'reasons': ['conflicting_dosage_forms'],
            'form_groups': groups,
        },
        'query_info': {'original': description}
    })


def _random_text_no_match_response(description: str):
    return jsonify({
        'status': 'no_match',
        'results': [],
        'message': 'No medication match found. Please check the description or enter clearer medication details.',
        'query_specificity': {
            'level': 'unknown',
            'score': 0.0,
            'reasons': ['unrecognized_random_text']
        },
        'query_info': {'original': description}
    })

@app.route('/api/identify', methods=['POST'])
@login_required


def api_identify():
    data = request.get_json(silent=True) or {}
    description = data.get('description', '').strip()
    top_k = min(int(data.get('top_k', 5)), 10)
    color_filter = data.get('color', '').strip().lower()
    shape_filter = data.get('shape', '').strip().lower()
    packaging_filter = data.get('packaging', '').strip().lower()
    use_filter = data.get('intended_use', '').strip().lower()

    # Case 1: empty input.
    if not description and not color_filter and not shape_filter and not packaging_filter and not use_filter:
        return jsonify({
            'status': 'incomplete',
            'results': [],
            'message': 'Please provide a medication description or select at least one filter.'
        }), 400

    # Filters only: return possible matches with low confidence instead of high filter-based scores.
    if not description and (color_filter or shape_filter or packaging_filter or use_filter):
        return _weak_description_response('', color_filter, shape_filter, packaging_filter, use_filter, top_k)

    # Case 12: contradictory dosage form, e.g., cream + tablet.
    has_contradiction, contradiction_groups = _has_contradictory_dosage_forms(description)
    if has_contradiction:
        return _contradictory_description_response(description, contradiction_groups)

    # Exact drug/generic names should go through the AI/direct-match path.
    exact_drug = _exact_drug_match(description)

    # Weak but meaningful input: red, fever, yellow oval tablet.
    # Show possible matches with low confidence.
    if not exact_drug and _is_weak_meaningful_description(description):
        return _weak_description_response(description, color_filter, shape_filter, packaging_filter, use_filter, top_k)

    # Case 2 / 17: random or unrecognized text.
    # If the text has no recognizable query terms and no DB text match, return no_match.
    recognized_terms = _extract_query_terms(description, color_filter, shape_filter, packaging_filter, use_filter)
    recognized_count = sum(len(v) for v in recognized_terms.values())
    if not exact_drug and recognized_count == 0 and not _has_any_db_text_match(description):
        return _random_text_no_match_response(description)

    model = get_ai_model()
    result = model.identify_medication(description, top_k=top_k,
                                       color_filter=color_filter, shape_filter=shape_filter,
                                       packaging_filter=packaging_filter, use_filter=use_filter)

    # Filter blending is handled inside ai_model.identify_medication().
    has_filters = color_filter or shape_filter or packaging_filter or use_filter
    if has_filters and result.get('status') == 'success' and result.get('results'):
        result['results'].sort(key=lambda x: x.get('confidence_score', 0), reverse=True)
        for i, r in enumerate(result['results']):
            r['rank'] = i + 1

    # If AI returned no match/error, try direct DB search using extracted attributes.
    # Do not fallback for incomplete because incomplete means the input should not be forced.
    if result.get('status') in ('no_match', 'error') and description:
        from nlp_processor import is_arabic, extract_arabic_attributes, translate_arabic_query
        from nlp_processor import MedicationNLPProcessor
        nlp = MedicationNLPProcessor()

        q = Medication.query.filter_by(is_active=True)

        if is_arabic(description):
            attrs = extract_arabic_attributes(description)
            translated = translate_arabic_query(description)
            search_colors = attrs.get('colors', [])
            search_shapes = attrs.get('shapes', [])
            search_categories = attrs.get('categories', [])
            search_forms = attrs.get('forms', [])

            ar_like = f'%{description.strip()}%'
            q_ar = q.filter(
                (Medication.drug_name_ar.ilike(ar_like)) |
                (Medication.generic_name_ar.ilike(ar_like)) |
                (Medication.indications_ar.ilike(ar_like)) |
                (Medication.description.ilike(ar_like))
            )
            if q_ar.limit(1).first():
                q = q_ar
            else:
                ar_words = [w for w in description.split() if len(w) > 2]
                matched_word_query = None
                for word in ar_words:
                    q_w = q.filter(
                        (Medication.drug_name_ar.ilike(f'%{word}%')) |
                        (Medication.generic_name_ar.ilike(f'%{word}%')) |
                        (Medication.indications_ar.ilike(f'%{word}%')) |
                        (Medication.description.ilike(f'%{word}%'))
                    )
                    if q_w.limit(1).first():
                        matched_word_query = q_w
                        break
                if matched_word_query is not None:
                    q = matched_word_query
                elif translated and translated != description:
                    for part in translated.split():
                        if len(part) > 3:
                            q2 = q.filter(
                                (Medication.drug_name.ilike(f'%{part}%')) |
                                (Medication.description.ilike(f'%{part}%')) |
                                (Medication.category.ilike(f'%{part}%')) |
                                (Medication.indications.ilike(f'%{part}%'))
                            )
                            if q2.limit(1).first():
                                q = q2
                                break
        else:
            attrs = nlp.extract_attributes(description)
            search_colors = attrs.get('colors', [])
            search_shapes = attrs.get('shapes', [])
            search_categories = attrs.get('categories', [])
            search_forms = attrs.get('forms', [])
            like = f'%{description[:40]}%'
            q = q.filter(
                (Medication.drug_name.ilike(like)) |
                (Medication.generic_name.ilike(like)) |
                (Medication.description.ilike(like)) |
                (Medication.category.ilike(like)) |
                (Medication.indications.ilike(like))
            )

        eff_color = color_filter or (search_colors[0] if search_colors else '')
        eff_shape = shape_filter or (search_shapes[0] if search_shapes else '')
        eff_cat = use_filter or (search_categories[0] if search_categories else '')
        eff_packaging = packaging_filter or (search_forms[0] if search_forms else '')

        if eff_color:
            q = q.filter(Medication.color.ilike(f'%{eff_color}%'))
        if eff_shape:
            q = q.filter(Medication.shape.ilike(f'%{eff_shape}%'))
        if eff_cat:
            q = q.filter(Medication.category.ilike(f'%{eff_cat}%'))
        if eff_packaging:
            q = q.filter(
                (Medication.dosage_form.ilike(f'%{eff_packaging}%')) |
                (Medication.package_type.ilike(f'%{eff_packaging}%'))
            )

        db_drugs = q.order_by(Medication.drug_name).limit(top_k).all()

        if db_drugs:
            result['results'] = [
                _score_result(d, eff_color, eff_shape, eff_cat, eff_packaging, i)
                for i, d in enumerate(db_drugs)
            ]
            for i, r in enumerate(result['results']):
                r['rank'] = i + 1
            result['status'] = 'success'
        else:
            result = {
                'status': 'no_match',
                'results': [],
                'message': 'No medication found in the database for this description.',
                'query_info': {
                    'original': description,
                    'color': color_filter,
                    'shape': shape_filter,
                    'packaging': packaging_filter,
                    'intended_use': use_filter,
                }
            }

    if result.get('results'):
        _inject_drug_ids(result['results'])
        _inject_image_urls(result['results'])

    if result.get('results'):
        top = result['results'][0]
        parts = [p for p in [description, color_filter and f'color:{color_filter}',
                              shape_filter and f'shape:{shape_filter}',
                              packaging_filter and f'pack:{packaging_filter}',
                              use_filter and f'use:{use_filter}'] if p]
        query_text = ' | '.join(parts) or 'unknown'
        history = SearchHistory(
            user_id=current_user.id,
            description=query_text,
            results_json=json.dumps(result['results']),
            top_result=top.get('medication', ''),
            top_confidence=top.get('confidence_score', 0.0),
            result_count=len(result['results']),
        )
        db.session.add(history)
        db.session.commit()
        log_action('identify', f'Query: "{query_text[:60]}" → {top.get("medication", "")}')

    return jsonify(result)


@app.route('/api/drugs/filter')
@login_required
def api_drugs_filter():
    """Filter drugs directly by color, shape, category, or form — no AI required."""
    color = request.args.get('color', '').strip().lower()
    shape = request.args.get('shape', '').strip().lower()
    category = request.args.get('category', '').strip()
    form = request.args.get('form', '').strip()
    q_text = request.args.get('q', '').strip()
    page = max(1, int(request.args.get('page', 1)))
    per_page = min(50, int(request.args.get('per_page', 20)))

    query = Medication.query.filter_by(is_active=True)
    if color:
        query = query.filter(Medication.color.ilike(f'%{color}%'))
    if shape:
        query = query.filter(Medication.shape.ilike(f'%{shape}%'))
    if category:
        query = query.filter(Medication.category.ilike(f'%{category}%'))
    if form:
        query = query.filter(Medication.dosage_form.ilike(f'%{form}%'))
    if q_text:
        like = f'%{q_text}%'
        query = query.filter(
            (Medication.drug_name.ilike(like)) |
            (Medication.generic_name.ilike(like)) |
            (Medication.description.ilike(like))
        )

    paginated = query.order_by(Medication.drug_name).paginate(
        page=page, per_page=per_page, error_out=False)

    return jsonify({
        'drugs': [{
            'id': d.id, 'drug_name': d.drug_name, 'generic_name': d.generic_name,
            'strength': d.strength_label, 'dosage_form': d.dosage_form,
            'category': d.category, 'color': d.color, 'shape': d.shape,
            'price': d.price, 'manufacturer': d.manufacturer,
            'long_term_effect': d.long_term_effect,
        } for d in paginated.items],
        'total': paginated.total,
        'pages': paginated.pages,
        'current_page': page,
    })


@app.route('/api/statistics')
@login_required
def api_statistics():
    total_searches = SearchHistory.query.count()
    user_searches = SearchHistory.query.filter_by(user_id=current_user.id).count()
    total_drugs = Medication.query.filter_by(is_active=True).count()
    total_users = User.query.count()
    model_info = get_ai_model().get_model_info()

    return jsonify({
        'total_searches': total_searches,
        'user_searches': user_searches,
        'total_drugs': total_drugs,
        'total_users': total_users,
        'model_info': model_info,
    })


@app.route('/api/drug/<int:drug_id>')
@login_required
def api_drug_detail(drug_id):
    drug = db.session.get(Medication, drug_id)
    if not drug:
        return jsonify({'error': 'Drug not found'}), 404
    return jsonify({
        'id': drug.id, 'drug_name': drug.drug_name,
        'generic_name': drug.generic_name, 'strength': drug.strength_label,
        'dosage_form': drug.dosage_form, 'category': drug.category,
        'color': drug.color, 'shape': drug.shape, 'route': drug.admin_route,
        'manufacturer': drug.manufacturer, 'price': drug.price,
        'storage': drug.storage, 'legal_class': drug.legal_class,
        'description': drug.description,
    })


@app.route('/api/drugs/search')
@login_required
def api_drugs_search():
    q = request.args.get('q', '').strip()
    category = request.args.get('category', '').strip()
    categories_json = request.args.get('categories', '').strip()   # JSON array of exact category names
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))

    query = Medication.query.filter_by(is_active=True)
    if q:
        like = f'%{q}%'
        query = query.filter(
            (Medication.drug_name.ilike(like)) |
            (Medication.generic_name.ilike(like)) |
            (Medication.description.ilike(like)) |
            (Medication.drug_name_ar.ilike(like)) |
            (Medication.generic_name_ar.ilike(like)) |
            (Medication.indications.ilike(like)) |
            (Medication.indications_ar.ilike(like))
        )
    if categories_json:
        import json as _json
        try:
            cat_list = _json.loads(categories_json)
            if cat_list:
                query = query.filter(Medication.category.in_(cat_list))
        except Exception:
            pass
    elif category:
        query = query.filter(Medication.category.ilike(f'%{category}%'))

    paginated = query.order_by(Medication.drug_name).paginate(
        page=page, per_page=per_page, error_out=False)

    return jsonify({
        'drugs': [{
            'id': d.id, 'drug_name': d.drug_name,
            'drug_name_ar': d.drug_name_ar,
            'generic_name': d.generic_name,
            'generic_name_ar': d.generic_name_ar,
            'strength': d.strength_label,
            'dosage_form': d.dosage_form, 'category': d.category,
            'color': d.color, 'shape': d.shape, 'price': d.price,
            'manufacturer': d.manufacturer,
        } for d in paginated.items],
        'total': paginated.total,
        'pages': paginated.pages,
        'current_page': page,
    })


# ---------------------------------------------------------------------------
# History & Profile
# ---------------------------------------------------------------------------

@app.route('/history')
@login_required
def history():
    page     = int(request.args.get('page', 1))
    q        = request.args.get('q', '').strip()
    conf     = request.args.get('conf', '')       # high | medium | low
    has_match = request.args.get('match', '')     # 1 = matched, 0 = no match

    query = SearchHistory.query.filter_by(user_id=current_user.id)

    if q:
        like = f'%{q}%'
        query = query.filter(
            (SearchHistory.description.ilike(like)) |
            (SearchHistory.top_result.ilike(like))
        )
    if conf == 'high':
        query = query.filter(SearchHistory.top_confidence >= 0.8)
    elif conf == 'medium':
        query = query.filter(SearchHistory.top_confidence >= 0.5,
                             SearchHistory.top_confidence < 0.8)
    elif conf == 'low':
        query = query.filter(SearchHistory.top_confidence < 0.5,
                             SearchHistory.top_confidence > 0)
    if has_match == '1':
        query = query.filter(SearchHistory.top_result != '')
    elif has_match == '0':
        query = query.filter(
            (SearchHistory.top_result == '') | (SearchHistory.top_result.is_(None))
        )

    searches = query.order_by(SearchHistory.created_at.desc())\
        .paginate(page=page, per_page=15, error_out=False)

    return render_template('history.html', searches=searches,
                           q=q, conf=conf, has_match=has_match)


@app.route('/history/<int:search_id>/delete', methods=['POST'])
@login_required
def delete_history(search_id):
    record = SearchHistory.query.filter_by(id=search_id,
                                           user_id=current_user.id).first_or_404()
    db.session.delete(record)
    db.session.commit()
    flash('Search record deleted.', 'success')
    return redirect(url_for('history'))


@app.route('/medication/<int:med_id>')
@login_required
def medication_detail(med_id):
    drug = db.session.get(Medication, med_id)
    if not drug:
        abort(404)
    related = Medication.query.filter(
        Medication.category == drug.category,
        Medication.id != drug.id,
        Medication.is_active == True
    ).limit(4).all()

    # Resolve a medication image
    image_url = None
    img_path = _drug_vision_image(drug.drug_name)
    if not img_path and drug.image_path and os.path.isfile(drug.image_path):
        img_path = drug.image_path
    if img_path:
        img_stem = os.path.splitext(os.path.basename(img_path))[0]
        image_url = url_for('drug_vision_image', drug_name=img_stem)

    return render_template('medication_detail.html', drug=drug, related=related,
                           image_url=image_url)


@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        email = request.form.get('email', '').strip()
        hospital = request.form.get('hospital', '').strip()
        specialty = request.form.get('specialty', '').strip()
        new_password = request.form.get('new_password', '')
        confirm = request.form.get('confirm_password', '')

        # Check email uniqueness
        if email and email != current_user.email:
            existing = User.query.filter_by(email=email).first()
            if existing:
                flash('Email already in use.', 'danger')
                return render_template('profile.html')

        current_user.full_name = full_name
        if email:
            current_user.email = email
        current_user.hospital = hospital
        current_user.specialty = specialty

        if new_password:
            if len(new_password) < 6:
                flash('New password must be at least 6 characters.', 'danger')
                return render_template('profile.html')
            if new_password != confirm:
                flash('Passwords do not match.', 'danger')
                return render_template('profile.html')
            current_user.set_password(new_password)

        db.session.commit()
        log_action('update_profile')
        flash('Profile updated successfully.', 'success')
        return redirect(url_for('profile'))

    total_searches = SearchHistory.query.filter_by(user_id=current_user.id).count()
    return render_template('profile.html', total_searches=total_searches)


# ---------------------------------------------------------------------------
# Admin management routes
# ---------------------------------------------------------------------------

@app.route('/admin/user/<int:user_id>/toggle-role', methods=['POST'])
@login_required
@role_required('admin')
def toggle_user_role(user_id):
    user = db.session.get(User, user_id)
    if not user:
        abort(404)
    role = request.form.get('role', 'patient')
    if role in ('admin', 'doctor', 'pharmacist', 'patient'):
        user.role = role
        db.session.commit()
        log_action('change_role', f'User {user.username} → {role}')
        flash(f'{user.username} role changed to {role}.', 'success')
    return redirect(url_for('admin_dashboard') + '#users')


@app.route('/admin/user/<int:user_id>/toggle-active', methods=['POST'])
@login_required
@role_required('admin')
def toggle_user_active(user_id):
    user = db.session.get(User, user_id)
    if not user or user.id == current_user.id:
        abort(400)
    user.is_active = not user.is_active
    db.session.commit()
    status = 'activated' if user.is_active else 'deactivated'
    log_action('toggle_active', f'User {user.username} {status}')
    flash(f'{user.username} has been {status}.', 'success')
    return redirect(url_for('admin_dashboard') + '#users')


@app.route('/admin/user/<int:user_id>/delete', methods=['POST'])
@login_required
@role_required('admin')
def delete_user(user_id):
    user = db.session.get(User, user_id)
    if not user or user.id == current_user.id:
        abort(400)
    username = user.username
    db.session.delete(user)
    db.session.commit()
    log_action('delete_user', f'Deleted user {username}')
    flash(f'User {username} has been deleted.', 'success')
    return redirect(url_for('admin_dashboard') + '#users')


@app.route('/admin/drugs/add', methods=['POST'])
@login_required
@role_required('admin')
def admin_add_drug():
    drug_name = request.form.get('drug_name', '').strip()
    if not drug_name:
        flash('Medicine name is required.', 'danger')
        return redirect(url_for('admin_dashboard') + '#drugs')

    generic_name = request.form.get('generic_name', '').strip()
    strength = request.form.get('strength', '').strip()
    strength_unit = request.form.get('strength_unit', '').strip()
    admin_route = request.form.get('admin_route', '').strip()
    dosage_form = request.form.get('dosage_form', '').strip()
    category = request.form.get('category', '').strip()
    color = request.form.get('color', '').strip()
    shape = request.form.get('shape', '').strip()
    manufacturer = request.form.get('manufacturer', '').strip()
    price = request.form.get('price', '').strip()
    storage = request.form.get('storage', '').strip()
    legal_class = request.form.get('legal_class', '').strip()
    indications = request.form.get('indications', '').strip()
    side_effects = request.form.get('side_effects', '').strip()
    contraindications = request.form.get('contraindications', '').strip()
    description = request.form.get('description', '').strip()

    # Auto-generate description for AI identification if not provided
    if not description:
        parts = [f'drug {drug_name.lower()}']
        if generic_name:
            parts.append(f'generic {generic_name.lower()}')
        if dosage_form:
            parts.append(f'form {dosage_form.lower()}')
        if strength:
            parts.append(f'strength {strength}')
        if category:
            parts.append(category.lower())
        if color:
            parts.append(color.lower())
        if shape:
            parts.append(shape.lower())
        description = ' '.join(parts)

    drug = Medication(
        drug_name=drug_name,
        generic_name=generic_name,
        strength=strength,
        strength_unit=strength_unit,
        admin_route=admin_route,
        dosage_form=dosage_form,
        category=category,
        color=color,
        shape=shape,
        manufacturer=manufacturer,
        price=price,
        storage=storage,
        legal_class=legal_class,
        indications=indications,
        side_effects=side_effects,
        contraindications=contraindications,
        description=description,
    )
    db.session.add(drug)
    db.session.commit()
    log_action('add_drug', f'Added drug: {drug.drug_name}')
    flash(f'Drug "{drug.drug_name}" added successfully.', 'success')
    return redirect(url_for('admin_dashboard') + '#drugs')


@app.route('/admin/drug/<int:drug_id>/delete', methods=['POST'])
@login_required
@role_required('admin')
def admin_delete_drug(drug_id):
    drug = db.session.get(Medication, drug_id)
    if not drug:
        abort(404)
    drug.is_active = False
    db.session.commit()
    log_action('delete_drug', f'Removed drug: {drug.drug_name}')
    flash(f'Drug "{drug.drug_name}" removed from database.', 'success')
    return redirect(url_for('admin_dashboard') + '#drugs')


# ---------------------------------------------------------------------------
# Drug Vision image serving
# ---------------------------------------------------------------------------

_DRUG_VISION_BASE = os.path.join(BASE_DIR, 'DataSet', 'medication_images')


def _drug_vision_image(drug_name: str) -> str | None:
    """Return the path of the first matching image in DataSet/medication_images/."""
    if not os.path.isdir(_DRUG_VISION_BASE):
        return None
    name_lower = drug_name.lower().strip()
    name_under = name_lower.replace(' ', '_').replace('+', 'and').replace('/', '_').replace('%', '')
    # Also try first word only (e.g. "Humulin 70/30" -> "humulin")
    first_word = name_lower.split()[0] if name_lower else ''

    best = None
    for fname in os.listdir(_DRUG_VISION_BASE):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        stem = os.path.splitext(fname)[0].lower()
        # Exact match on underscore-normalised name
        if stem == name_under:
            return os.path.join(_DRUG_VISION_BASE, fname)
        # Prefix / contains match
        if stem.startswith(name_under) or name_under in stem:
            return os.path.join(_DRUG_VISION_BASE, fname)
        # First-word match (keep as fallback — less specific)
        if first_word and len(first_word) >= 4 and stem == first_word:
            best = os.path.join(_DRUG_VISION_BASE, fname)
    return best


def _inject_image_urls(results: list) -> None:
    """Inject image_url into results for drugs that have medication images."""
    for r in results:
        if r.get('image_url'):
            continue
        name = r.get('medication', '')
        if not name:
            continue

        # Primary: medication_images folder lookup
        img_path = _drug_vision_image(name)

        # Fallback: image_path from DB record
        if not img_path:
            db_id = r.get('drug_id')
            if db_id:
                med = Medication.query.get(db_id)
                if med and med.image_path and os.path.isfile(med.image_path):
                    img_path = med.image_path

        # Fallback: image_path already resolved by ai_model
        if not img_path:
            ai_path = r.get('image_path', '')
            if ai_path and os.path.isfile(str(ai_path)):
                img_path = str(ai_path)

        if img_path:
            # Use the filename (without ext) as the drug_name route param
            img_stem = os.path.splitext(os.path.basename(img_path))[0]
            r['image_url'] = url_for('drug_vision_image', drug_name=img_stem)


@app.route('/drug-image/<path:drug_name>')
@login_required
def drug_vision_image(drug_name):
    """Serve a medication image from DataSet/medication_images/."""
    from flask import send_file
    # Try direct file match first (drug_name may be a filename stem)
    for ext in ('.jpg', '.jpeg', '.png'):
        candidate = os.path.join(_DRUG_VISION_BASE, drug_name + ext)
        if os.path.isfile(candidate):
            return send_file(candidate, mimetype='image/jpeg')
    # Fallback to fuzzy match
    img_path = _drug_vision_image(drug_name)
    if not img_path or not os.path.isfile(img_path):
        return '', 404
    return send_file(img_path, mimetype='image/jpeg')


# ---------------------------------------------------------------------------
# Image upload helper (pill image analysis module removed)
# ---------------------------------------------------------------------------

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Patient routes
# ---------------------------------------------------------------------------

@app.route('/patient/dashboard')
@login_required
@role_required('patient', 'admin')
def patient_dashboard():
    # Current medications
    current_meds = PatientMedication.query.filter_by(
        patient_id=current_user.id, status='current').all()
    # Previous medications
    prev_meds = PatientMedication.query.filter_by(
        patient_id=current_user.id, status='previous').all()

    # Check interactions across current medications (expand with generic names)
    from drug_interaction_checker import DrugInteractionChecker
    checker = DrugInteractionChecker()
    med_names = [m.medication_name for m in current_meds]
    expanded = list(med_names)
    for mn in med_names:
        db_med = Medication.query.filter(Medication.drug_name.ilike(f'%{mn}%')).first()
        if db_med and db_med.generic_name:
            generic = db_med.generic_name.split('/')[0].strip()
            if generic.lower() not in [n.lower() for n in expanded]:
                expanded.append(generic)
    interactions = checker.check_by_names(expanded) if len(expanded) >= 2 else []

    # Collect long-term notices
    long_term_notices = [m for m in current_meds if m.long_term_flag]

    recent_searches = SearchHistory.query.filter_by(user_id=current_user.id)\
        .order_by(SearchHistory.created_at.desc()).limit(5).all()

    drugs = Medication.query.filter_by(is_active=True)\
        .with_entities(Medication.drug_name, Medication.generic_name)\
        .order_by(Medication.drug_name).all()

    return render_template('patient/dashboard.html',
                           current_meds=current_meds,
                           prev_meds=prev_meds,
                           interactions=interactions,
                           long_term_notices=long_term_notices,
                           recent_searches=recent_searches,
                           drugs=drugs)


@app.route('/patient/medications')
@login_required
@role_required('patient', 'admin')
def patient_medications():
    current_meds = PatientMedication.query.filter_by(
        patient_id=current_user.id, status='current')\
        .order_by(PatientMedication.added_at.desc()).all()
    prev_meds = PatientMedication.query.filter_by(
        patient_id=current_user.id, status='previous')\
        .order_by(PatientMedication.end_date.desc()).all()

    from drug_interaction_checker import DrugInteractionChecker
    checker = DrugInteractionChecker()
    med_names = [m.medication_name for m in current_meds]
    expanded = list(med_names)
    for mn in med_names:
        db_med = Medication.query.filter(Medication.drug_name.ilike(f'%{mn}%')).first()
        if db_med and db_med.generic_name:
            generic = db_med.generic_name.split('/')[0].strip()
            if generic.lower() not in [n.lower() for n in expanded]:
                expanded.append(generic)
    interactions = checker.check_by_names(expanded) if len(expanded) >= 2 else []

    return render_template('patient/medications.html',
                           current_meds=current_meds,
                           prev_meds=prev_meds,
                           interactions=interactions)


@app.route('/patient/medications/add', methods=['GET', 'POST'])
@login_required
@role_required('patient', 'admin')
def add_patient_medication():
    if request.method == 'POST':
        med_name = request.form.get('medication_name', '').strip()
        if not med_name:
            flash('Medication name is required.', 'danger')
            return redirect(url_for('add_patient_medication'))

        # Try to find matching drug in DB
        med_obj = Medication.query.filter(
            Medication.drug_name.ilike(f'%{med_name}%')
        ).first()

        start_str = request.form.get('start_date', '')
        end_str = request.form.get('end_date', '')
        try:
            start_d = datetime.strptime(start_str, '%Y-%m-%d').date() if start_str else None
        except ValueError:
            start_d = None
        try:
            end_d = datetime.strptime(end_str, '%Y-%m-%d').date() if end_str else None
        except ValueError:
            end_d = None

        pm = PatientMedication(
            patient_id=current_user.id,
            medication_id=med_obj.id if med_obj else None,
            medication_name=med_name,
            status=request.form.get('status', 'current'),
            start_date=start_d,
            end_date=end_d,
            dosage=request.form.get('dosage', '').strip(),
            prescribed_by=request.form.get('prescribed_by', '').strip(),
            notes=request.form.get('notes', '').strip(),
        )
        db.session.add(pm)
        db.session.commit()
        log_action('add_patient_medication', f'{current_user.username} added {med_name}')

        # Check for long-term effects and interactions to show on confirmation
        long_term = pm.long_term_info
        new_interactions = []
        if pm.status == 'current':
            current_meds = PatientMedication.query.filter_by(
                patient_id=current_user.id, status='current').all()
            med_names = [m.medication_name for m in current_meds]
            # Expand with generic names for better interaction matching
            expanded_names = list(med_names)
            for mn in med_names:
                db_med = Medication.query.filter(
                    Medication.drug_name.ilike(f'%{mn}%')
                ).first()
                if db_med and db_med.generic_name:
                    generic = db_med.generic_name.split('/')[0].strip()
                    if generic.lower() not in [n.lower() for n in expanded_names]:
                        expanded_names.append(generic)
            if len(expanded_names) >= 2:
                from drug_interaction_checker import DrugInteractionChecker
                checker = DrugInteractionChecker()
                all_interactions = checker.check_by_names(expanded_names)
                # Only show interactions that involve the newly added drug
                med_name_lower = med_name.lower()
                # Also check against the generic name of the new drug
                new_generic = ''
                new_med_obj = Medication.query.filter(
                    Medication.drug_name.ilike(f'%{med_name}%')
                ).first()
                if new_med_obj and new_med_obj.generic_name:
                    new_generic = new_med_obj.generic_name.split('/')[0].strip().lower()
                new_interactions = [
                    ia for ia in all_interactions
                    if med_name_lower in ia['drug_a'].lower() or
                       ia['drug_a'].lower() in med_name_lower or
                       med_name_lower in ia['drug_b'].lower() or
                       ia['drug_b'].lower() in med_name_lower or
                       (new_generic and (new_generic in ia['drug_a'].lower() or
                        ia['drug_a'].lower() in new_generic or
                        new_generic in ia['drug_b'].lower() or
                        ia['drug_b'].lower() in new_generic))
                ]

        return render_template('patient/medication_added.html',
                               med=pm,
                               long_term=long_term,
                               new_interactions=new_interactions)

    # GET — list drugs for autocomplete
    drugs = Medication.query.filter_by(is_active=True)\
        .with_entities(Medication.drug_name, Medication.generic_name)\
        .order_by(Medication.drug_name).all()
    return render_template('patient/add_medication.html', drugs=drugs, today=date.today())


@app.route('/patient/medications/<int:med_id>/update', methods=['POST'])
@login_required
def update_patient_medication(med_id):
    pm = PatientMedication.query.filter_by(
        id=med_id, patient_id=current_user.id).first_or_404()
    new_status = request.form.get('status', pm.status)
    if new_status in ('current', 'previous'):
        pm.status = new_status
        if new_status == 'previous' and not pm.end_date:
            pm.end_date = date.today()
    db.session.commit()
    flash('Medication status updated.', 'success')
    return redirect(url_for('patient_medications'))


@app.route('/patient/medications/<int:med_id>/delete', methods=['POST'])
@login_required
def delete_patient_medication(med_id):
    pm = PatientMedication.query.filter_by(
        id=med_id, patient_id=current_user.id).first_or_404()
    name = pm.medication_name
    db.session.delete(pm)
    db.session.commit()
    flash(f'"{name}" removed from your list.', 'success')
    return redirect(url_for('patient_medications'))

@app.route('/api/patient/confirm-medication', methods=['POST'])
@login_required
def api_patient_confirm_medication():
    from datetime import date

    data = request.get_json(silent=True) or {}
    medication_name = (data.get('medication_name') or '').strip()
    drug_id = data.get('drug_id')

    if not medication_name and not drug_id:
        return jsonify({
            'ok': False,
            'message': 'No medication selected.'
        }), 400

    medication = None

    if drug_id:
        try:
            medication = Medication.query.get(int(drug_id))
        except Exception:
            medication = None

    if medication is None and medication_name:
        medication = Medication.query.filter(
            db.func.lower(Medication.drug_name) == medication_name.lower()
        ).first()

    if medication is None and medication_name:
        medication = Medication.query.filter(
            Medication.drug_name.ilike(f'%{medication_name}%')
        ).first()

    if medication is None:
        return jsonify({
            'ok': False,
            'message': 'Medication was not found in the database.'
        }), 404

    # Prevent duplicate current medication for the same patient
    existing = PatientMedication.query.filter_by(
        patient_id=current_user.id,
        medication_id=medication.id,
        status='current'
    ).first()

    if existing:
        return jsonify({
            'ok': True,
            'message': 'Medication is already in your current list.',
            'already_exists': True
        })

    pm = PatientMedication(
    patient_id=current_user.id,
    medication_id=medication.id,
    medication_name=medication.drug_name,
    status='current',
    start_date=date.today(),
    end_date=None,
    dosage='',
    prescribed_by='',
    notes=''
)

    db.session.add(pm)
    db.session.commit()

    return jsonify({
        'ok': True,
        'message': 'Medication added successfully.',
        'medication': medication.drug_name
    })

@app.route('/api/patient/check-interactions', methods=['POST'])
@login_required
def api_check_interactions():
    data = request.get_json(silent=True) or {}
    names = data.get('medication_names', [])

    # Support single new_drug check against current user's medications
    new_drug = data.get('new_drug', '').strip()
    if new_drug:
        current_meds = PatientMedication.query.filter_by(
            patient_id=current_user.id, status='current').all()
        names = [m.medication_name for m in current_meds] + [new_drug]

    if not names or len(names) < 2:
        return jsonify({'interactions': [], 'count': 0, 'highest_risk': 'none'})

    # Expand names with generic equivalents so interactions are caught
    # regardless of whether brand or generic name was used.
    # Also expand generics → brand names for combination drugs (e.g. Fevadol → Ibuprofen → Brufen).
    import re as _re
    expanded_names = list(names)
    for name in names:
        med = Medication.query.filter(
            Medication.drug_name.ilike(f'%{name}%')
        ).first()
        if med and med.generic_name:
            parts = _re.split(r'[/+]', med.generic_name)
            for part in parts:
                generic = part.strip()
                if generic and generic.lower() not in [n.lower() for n in expanded_names]:
                    expanded_names.append(generic)
    # Second pass: for each generic name, find brand names that contain it
    for name in list(expanded_names):
        brand_meds = Medication.query.filter(
            Medication.generic_name.ilike(f'%{name}%'),
            Medication.is_active == True
        ).all()
        for bm in brand_meds:
            if bm.drug_name.lower() not in [n.lower() for n in expanded_names]:
                expanded_names.append(bm.drug_name)

    from drug_interaction_checker import DrugInteractionChecker
    checker = DrugInteractionChecker()
    interactions = checker.check_by_names(expanded_names)
    risks = [i['risk_level'] for i in interactions]
    highest = 'high' if 'high' in risks else ('medium' if 'medium' in risks else ('low' if risks else 'none'))
    return jsonify({'interactions': interactions, 'count': len(interactions), 'highest_risk': highest})


@app.route('/api/patient/long-term-info', methods=['POST'])
@login_required
def api_long_term_info():
    data = request.get_json(silent=True) or {}
    name = data.get('medication_name', '').strip()
    if not name:
        return jsonify({'has_long_term': False})
    try:
        from long_term_effects import get_long_term_warning
        info = get_long_term_warning(name)
        if info:
            # Normalize to template-expected keys
            return jsonify({
                'has_long_term': True,
                'found': True,
                'drug_name': info.get('drug_name', name),
                'effect_summary': info.get('effect_summary', ''),
                'description': info.get('description', ''),
                'patient_guidance': info.get('patient_guidance', ''),
                'warning': info.get('patient_guidance', ''),
                'severity': info.get('severity', 'low'),
            })
    except Exception:
        pass
    return jsonify({'has_long_term': False, 'found': False})


# ---------------------------------------------------------------------------
# Pharmacist dispense routes
# ---------------------------------------------------------------------------

@app.route('/pharmacist/dispense', methods=['GET', 'POST'])
@login_required
@role_required('pharmacist', 'admin')
def pharmacist_dispense():
    if request.method == 'POST':
        med_id_str = request.form.get('medication_id', '')
        med_name = request.form.get('medication_name', '').strip()
        patient_name = request.form.get('patient_name', '').strip()
        patient_informed = request.form.get('patient_informed') == '1'
        notes = request.form.get('notes', '').strip()

        med_obj = None
        if med_id_str:
            try:
                med_obj = db.session.get(Medication, int(med_id_str))
            except ValueError:
                pass

        if med_obj:
            med_name = med_obj.drug_name

        log_entry = DispenseLog(
            pharmacist_id=current_user.id,
            patient_name=patient_name,
            medication_id=med_obj.id if med_obj else None,
            medication_name=med_name,
            long_term_warning_shown=bool(request.form.get('long_term_warning_shown')),
            patient_informed=patient_informed,
            notes=notes,
        )
        db.session.add(log_entry)
        db.session.commit()
        log_action('dispense', f'{current_user.username} dispensed {med_name} to {patient_name}')
        flash(f'"{med_name}" dispensed successfully.', 'success')
        return redirect(url_for('pharmacist_dispense'))

    drugs = Medication.query.filter_by(is_active=True)\
        .order_by(Medication.drug_name).all()
    logs = DispenseLog.query.filter_by(pharmacist_id=current_user.id)\
        .order_by(DispenseLog.dispensed_at.desc()).limit(20).all()
    return render_template('pharmacist/dispense.html', drugs=drugs, logs=logs)


@app.route('/api/drug/<int:drug_id>/long-term')
@login_required
def api_drug_long_term(drug_id):
    drug = db.session.get(Medication, drug_id)
    if not drug:
        return jsonify({'has_long_term': False})
    if drug.long_term_effect:
        return jsonify({
            'has_long_term': True,
            'description': drug.long_term_description,
            'duration': drug.long_term_duration,
            'warning_text': drug.warning_text,
        })
    # Fallback to module lookup
    try:
        from long_term_effects import get_long_term_warning
        info = get_long_term_warning(drug.drug_name)
        if info:
            return jsonify({'has_long_term': True, **info})
    except Exception:
        pass
    return jsonify({'has_long_term': False})


# ---------------------------------------------------------------------------
# Pipeline status & evaluation  (Roadmap Steps 1–5)
# ---------------------------------------------------------------------------

# Ground-truth test queries covering all 10 Drug Vision drugs
_EVAL_QUERIES = [
    {'query': 'round white tablet for fever and headache pain',          'expected': 'Biogesic'},
    {'query': 'white round tablet colds flu fever nasal congestion',     'expected': 'Neozep'},
    {'query': 'white round tablet decongestant antihistamine colds congestion', 'expected': 'Decolgen'},
    {'query': 'yellow oval tablet flu cold fever runny nose sneezing',   'expected': 'Bioflu'},
    {'query': 'red white oval tablet ibuprofen paracetamol pain arthritis', 'expected': 'Alaxan'},
    {'query': 'red white capsule ibuprofen pain headache fever menstrual', 'expected': 'Medicol'},
    {'query': 'clear blue antiseptic mouthwash sore throat gargle oral', 'expected': 'Bactidol'},
    {'query': 'white green chewable antacid tablet heartburn stomach acid bloating', 'expected': 'Kremil S'},
    {'query': 'orange capsule zinc supplement immune support mineral daily', 'expected': 'DayZinc'},
    {'query': 'yellow gold softgel omega-3 heart health cardiovascular supplement', 'expected': 'Fish Oil'},
]


@app.route('/api/system/status')
@login_required
def api_system_status():
    """
    CAPSULE Pipeline Status — returns the completion state of all 5 roadmap steps.
    """
    model = get_ai_model()

    capsule_csv = os.path.join(BASE_DIR, 'datasets', 'capsule_dataset.csv')
    imap_csv    = os.path.join(BASE_DIR, 'datasets', 'capsule_image_map.csv')

    # ── Step 1: Data Preprocessing ───────────────────────────────────────────
    csv_exists = os.path.exists(capsule_csv)
    drug_count_csv = 0
    if csv_exists:
        try:
            drug_count_csv = len(pd.read_csv(capsule_csv))
        except Exception:
            pass

    step1 = {
        'step': 1, 'name': 'Data Preprocessing',
        'status': 'complete' if csv_exists else 'pending',
        'details': {
            'capsule_dataset_csv':  csv_exists,
            'capsule_image_map_csv': os.path.exists(imap_csv),
            'drug_records':  drug_count_csv,
            'note': 'Run the Jupyter notebook to generate capsule_dataset.csv' if not csv_exists else '',
        },
    }

    # ── Step 2: NLP Model ────────────────────────────────────────────────────
    tfidf_ok = model.tfidf_model is not None
    vocab_size = len(model.tfidf_model.vocabulary_) if tfidf_ok else 0
    step2 = {
        'step': 2, 'name': 'NLP Model (TF-IDF)',
        'status': 'complete' if tfidf_ok else 'pending',
        'details': {
            'tfidf_loaded':   tfidf_ok,
            'vocabulary_size': vocab_size,
            'ml_model':       model.ml_model is not None,
            'bilstm':         model.bilstm_model is not None,
            'drug_records':   len(model.drug_df) if model.drug_df is not None else 0,
        },
    }

    # ── Step 3: Image Database ───────────────────────────────────────────────
    img_drugs  = len(model.image_db)
    img_total  = sum(len(v) for v in model.image_db.values())
    step3 = {
        'step': 3, 'name': 'Drug Image Database',
        'status': 'complete' if img_drugs > 0 else 'pending',
        'details': {
            'drugs_with_images': img_drugs,
            'total_images':      img_total,
            'image_map_csv':     os.path.exists(imap_csv),
        },
    }

    # ── Step 4: Application ──────────────────────────────────────────────────
    db_drugs   = Medication.query.filter_by(is_active=True).count()
    db_users   = User.query.count()
    db_searches = SearchHistory.query.count()
    step4 = {
        'step': 4, 'name': 'Application',
        'status': 'complete',
        'details': {
            'drugs_in_database': db_drugs,
            'registered_users':  db_users,
            'total_searches':    db_searches,
            'roles_supported':   ['admin', 'doctor', 'pharmacist', 'patient'],
            'api_ready':         model.is_ready,
        },
    }

    # ── Step 5: Testing & Evaluation ─────────────────────────────────────────
    step5 = {
        'step': 5, 'name': 'Testing & Evaluation',
        'status': 'ready' if model.is_ready else 'pending',
        'details': {
            'evaluation_endpoint': '/api/evaluate',
            'test_queries':        len(_EVAL_QUERIES),
            'total_searches_logged': db_searches,
        },
    }

    pipeline = [step1, step2, step3, step4, step5]
    complete = sum(1 for s in pipeline if s['status'] == 'complete')
    overall  = 'complete' if complete == len(pipeline) else (
               'ready'    if model.is_ready else 'in_progress')

    return jsonify({
        'status':           overall,
        'steps_complete':   complete,
        'total_steps':      len(pipeline),
        'pipeline':         pipeline,
    })


@app.route('/api/evaluate', methods=['GET'])
@login_required
def api_evaluate():
    """
    CAPSULE Step 5 — evaluate NLP identification accuracy.
    Returns top-1 / top-5 accuracy, Precision, Recall, F1, and AUC proxy.
    """
    model = get_ai_model()
    rows  = []
    correct = 0

    # Per-class tracking for Precision / Recall
    tp: dict = {}   # true positives  per expected drug
    fp: dict = {}   # false positives per predicted drug
    fn: dict = {}   # false negatives per expected drug

    conf_correct: list[float] = []   # confidence of correct top-1 predictions
    conf_wrong:   list[float] = []   # confidence of wrong top-1 predictions

    for test in _EVAL_QUERIES:
        result    = model.identify_medication(test['query'], top_k=5)
        top       = result['results'][0] if result.get('results') else {}
        predicted = top.get('medication', '')
        conf_val  = float(top.get('confidence_score', top.get('confidence_pct', 0)) or 0)
        if conf_val > 1:          # pct → fraction
            conf_val = conf_val / 100.0
        expected  = test['expected']

        is_correct = (
            expected.lower() in predicted.lower() or
            predicted.lower() in expected.lower()
        )
        if is_correct:
            correct += 1
            tp[expected] = tp.get(expected, 0) + 1
            conf_correct.append(conf_val)
        else:
            fn[expected]  = fn.get(expected, 0) + 1
            fp[predicted] = fp.get(predicted, 0) + 1
            conf_wrong.append(conf_val)

        rows.append({
            'query':      test['query'],
            'expected':   expected,
            'predicted':  predicted,
            'confidence': round(conf_val * 100, 1),
            'correct':    is_correct,
            'top5':       [r.get('medication', '') for r in result.get('results', [])],
        })

    n = len(_EVAL_QUERIES)
    top1_acc = round(correct / n * 100, 1)

    # Top-5 accuracy
    top5_correct = sum(
        1 for i, r in enumerate(rows)
        if any(_EVAL_QUERIES[i]['expected'].lower() in m.lower()
               for m in r['top5'])
    )
    top5_acc = round(top5_correct / n * 100, 1)

    # Micro-averaged Precision, Recall, F1
    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    precision = round(total_tp / (total_tp + total_fp) * 100, 1) if (total_tp + total_fp) > 0 else 0.0
    recall    = round(total_tp / (total_tp + total_fn) * 100, 1) if (total_tp + total_fn) > 0 else 0.0
    f1        = round(2 * precision * recall / (precision + recall), 1) if (precision + recall) > 0 else 0.0

    # AUC proxy — mean confidence for correct minus mean for wrong
    avg_conf_correct = round(sum(conf_correct) / len(conf_correct) * 100, 1) if conf_correct else 0.0
    avg_conf_wrong   = round(sum(conf_wrong)   / len(conf_wrong)   * 100, 1) if conf_wrong   else 0.0
    auc_proxy = round(
        (avg_conf_correct / 100 + (1 - avg_conf_wrong / 100)) / 2, 3
    ) if (conf_correct or conf_wrong) else None

    log_action('evaluate',
               f'NLP eval: top1={top1_acc}% top5={top5_acc}% P={precision}% R={recall}% F1={f1}%')

    return jsonify({
        'top1_accuracy':      top1_acc,
        'top5_accuracy':      top5_acc,
        'precision':          precision,
        'recall':             recall,
        'f1_score':           f1,
        'auc_proxy':          auc_proxy,
        'avg_conf_correct':   avg_conf_correct,
        'avg_conf_wrong':     avg_conf_wrong,
        'correct':            correct,
        'top5_correct':       top5_correct,
        'total_queries':      n,
        'results':            rows,
        'summary': (
            f"Top-1: {correct}/{n} ({top1_acc}%)  |  "
            f"Top-5: {top5_correct}/{n} ({top5_acc}%)  |  "
            f"P={precision}%  R={recall}%  F1={f1}%  AUC≈{auc_proxy}"
        ),
        # Note 6: Evaluation scope metadata — clarifies what was tested and limitations
        'evaluation_scope': {
            'dataset':          'KSA medication database (210 medications, 42 classes)',
            'test_queries':     n,
            'evaluation_type':  'internal held-out test set (not independent external validation)',
            'models_evaluated': 'full ensemble pipeline (TF-IDF + ML + BiLSTM + keyword filters)',
            'limitations': [
                'Evaluation uses predefined queries, not real-world user input variability',
                'AUC proxy is confidence-separation metric, not true ROC-AUC',
                'No cross-validation or external dataset validation performed',
                'Results reflect ensemble performance, not individual model accuracy',
            ],
        },
    })


# ---------------------------------------------------------------------------
# Admin evaluation page
# ---------------------------------------------------------------------------

@app.route('/admin/evaluation')
@login_required
@role_required('admin')
def admin_evaluation():
    """Render the ML/DL evaluation metrics dashboard."""
    return render_template('admin_evaluation.html')


# ---------------------------------------------------------------------------
# Team page
# ---------------------------------------------------------------------------

@app.route('/team')
@login_required
def team():
    return render_template('team.html')


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def page_not_found(e):
    return render_template('errors/404.html'), 404


@app.errorhandler(500)
def internal_error(e):
    return render_template('errors/500.html'), 500


# ---------------------------------------------------------------------------
# Database initialisation with seed data
# ---------------------------------------------------------------------------

SEED_DRUGS = [
    # From trained CSV
    dict(reg_no='0809258186', drug_name='Quillivant XR', generic_name='Methylphenidate',
         strength='5', strength_unit='mg/ml', admin_route='Oral', dosage_form='Powder for Oral Suspension',
         package_type='Bottle', legal_class='prescription', drug_type='NCE', shelf_life=24,
         storage='Store below 25°C', price='1256.00', manufacturer='Tris Pharma Inc',
         color='', shape='', category='ADHD',
         description='drug quillivant xr generic methylphenidate form powder for oral suspension strength 5'),
    dict(reg_no='0809258183', drug_name='Zyglic MR', generic_name='Gliclazide',
         strength='30', strength_unit='mg', admin_route='Oral', dosage_form='Modified-Release Tablet',
         package_type='Blister', legal_class='prescription', drug_type='Generic', shelf_life=24,
         storage='Do not store above 30°C', price='20.40', manufacturer='Zydus Lifesciences Limited',
         color='', shape='round', category='Diabetes',
         description='drug zyglic mr generic gliclazide form modified-release tablet strength 30'),
    dict(reg_no='0309258149', drug_name='Emedius', generic_name='Palonosetron Hydrochloride',
         strength='0.05', strength_unit='mg/ml', admin_route='Intravenous', dosage_form='Solution for Injection',
         package_type='Ampoule', legal_class='prescription', drug_type='Generic', shelf_life=24,
         storage='Store below 30°C, protect from light', price='695.00',
         manufacturer='Jamjoom Pharmaceuticals', color='', shape='', category='Antiemetic',
         description='drug emedius generic palonosetron hydrochloride form solution for injection strength 0.05'),
    dict(reg_no='0409258160', drug_name='Ozanex', generic_name='Ozenoxacin',
         strength='1', strength_unit='%', admin_route='Cutaneous', dosage_form='Cream',
         package_type='Tube', legal_class='prescription', drug_type='NCE', shelf_life=36,
         storage='Do not store above 30°C', price='62.70', manufacturer='Ferrer International',
         color='white', shape='', category='Antibiotic/Antifungal',
         description='drug ozanex generic ozenoxacin form cream strength 1'),
    dict(reg_no='2808258125', drug_name='Vatowis', generic_name='Vasopressin',
         strength='20', strength_unit='U/ml', admin_route='Intravenous', dosage_form='Injection',
         package_type='Vial', legal_class='prescription', drug_type='Generic', shelf_life=24,
         storage='Store in refrigerator 2–8°C, do not freeze', price='2421.20',
         manufacturer='Eugia Pharma', color='', shape='', category='Vasopressor',
         description='drug vatowis generic vasopressin form injection strength 20'),
    dict(reg_no='2808258126', drug_name='Prevymis', generic_name='Letermovir',
         strength='480', strength_unit='mg', admin_route='Oral', dosage_form='Film-Coated Tablet',
         package_type='Blister', legal_class='prescription', drug_type='NCE', shelf_life=36,
         storage='Do not store above 30°C', price='40656.00', manufacturer='MSD International',
         color='orange', shape='oval', category='Antiviral',
         description='drug prevymis generic letermovir form film-coated tablet strength 480'),
    dict(reg_no='2608258114', drug_name='Olimel N12E', generic_name='Glucose/Amino Acid/Lipid Emulsion',
         strength='27.5/14.2/17.5', strength_unit='g/100ml', admin_route='Intravenous',
         dosage_form='Emulsion for Infusion', package_type='Bag', legal_class='prescription',
         drug_type='NCE', shelf_life=21, storage='Do not store above 30°C, do not freeze',
         price='765.00', manufacturer='Baxter', color='white', shape='', category='Parenteral Nutrition',
         description='drug olimel n12e generic amino acid glucose lipid form emulsion for infusion parenteral nutrition'),
    dict(reg_no='2608258109', drug_name='Alzil', generic_name='Donepezil Hydrochloride',
         strength='10', strength_unit='mg', admin_route='Oral', dosage_form='Orodispersible Tablet',
         package_type='Blister', legal_class='prescription', drug_type='Generic', shelf_life=36,
         storage='Store below 30°C', price='197.45', manufacturer='Al-Taqaddom Pharmaceutical',
         color='white', shape='round', category='Neurological/Alzheimer',
         description='drug alzil generic donepezil hydrochloride form orodispersible tablet strength 10'),
    dict(reg_no='2608258106', drug_name='Disflax', generic_name='Deflazacort',
         strength='6', strength_unit='mg', admin_route='Oral', dosage_form='Tablet',
         package_type='Blister', legal_class='prescription', drug_type='Generic', shelf_life=24,
         storage='Store below 30°C', price='11.40', manufacturer='Faes Farma',
         color='white', shape='round', category='Anti-Inflammatory/Corticosteroid',
         description='drug disflax generic deflazacort form tablet strength 6 anti-inflammatory corticosteroid'),
    # Additional common KSA drugs
    dict(drug_name='Glucophage', generic_name='Metformin', strength='500', strength_unit='mg',
         admin_route='Oral', dosage_form='Film-Coated Tablet', package_type='Blister',
         legal_class='prescription', shelf_life=36, storage='Store below 30°C', price='15.00',
         manufacturer='Merck', color='white', shape='oval', category='Diabetes',
         description='drug glucophage generic metformin form film-coated tablet strength 500 diabetes blood sugar glucose'),
    dict(drug_name='Norvasc', generic_name='Amlodipine', strength='5', strength_unit='mg',
         admin_route='Oral', dosage_form='Tablet', package_type='Blister',
         legal_class='prescription', shelf_life=36, storage='Store below 30°C', price='22.00',
         manufacturer='Pfizer', color='white', shape='round', category='Hypertension',
         description='drug norvasc generic amlodipine form tablet strength 5 hypertension blood pressure'),
    dict(drug_name='Zocor', generic_name='Simvastatin', strength='20', strength_unit='mg',
         admin_route='Oral', dosage_form='Tablet', package_type='Blister',
         legal_class='prescription', shelf_life=36, storage='Store below 25°C', price='18.00',
         manufacturer='MSD', color='tan', shape='oval', category='Cholesterol',
         description='drug zocor generic simvastatin form tablet strength 20 cholesterol statin'),
    dict(drug_name='Lipiget', generic_name='Atorvastatin', strength='20', strength_unit='mg',
         admin_route='Oral', dosage_form='Film-Coated Tablet', package_type='Blister',
         legal_class='prescription', shelf_life=36, storage='Store below 30°C', price='25.00',
         manufacturer='Pfizer', color='white', shape='oval', category='Cholesterol',
         description='drug lipiget generic atorvastatin form film-coated tablet strength 20 cholesterol'),
    dict(drug_name='Augmentin', generic_name='Amoxicillin/Clavulanate', strength='625', strength_unit='mg',
         admin_route='Oral', dosage_form='Film-Coated Tablet', package_type='Blister',
         legal_class='prescription', shelf_life=24, storage='Store below 25°C', price='45.00',
         manufacturer='GSK', color='white', shape='oval', category='Antibiotic',
         description='drug augmentin generic amoxicillin clavulanate form film-coated tablet strength 625 antibiotic infection'),
    dict(drug_name='Azithral', generic_name='Azithromycin', strength='500', strength_unit='mg',
         admin_route='Oral', dosage_form='Film-Coated Tablet', package_type='Blister',
         legal_class='prescription', shelf_life=24, storage='Store below 30°C', price='35.00',
         manufacturer='Alkem', color='pink', shape='oval', category='Antibiotic',
         description='drug azithral generic azithromycin form film-coated tablet strength 500 antibiotic infection'),
    dict(drug_name='Panadol', generic_name='Paracetamol', strength='500', strength_unit='mg',
         admin_route='Oral', dosage_form='Tablet', package_type='Blister',
         legal_class='OTC', shelf_life=36, storage='Store below 25°C', price='8.50',
         manufacturer='GSK', color='white', shape='caplet', category='Pain Relief/Antipyretic',
         description='drug panadol generic paracetamol form tablet strength 500 pain relief headache fever antipyretic'),
    dict(drug_name='Brufen', generic_name='Ibuprofen', strength='400', strength_unit='mg',
         admin_route='Oral', dosage_form='Film-Coated Tablet', package_type='Blister',
         legal_class='OTC', shelf_life=36, storage='Store below 30°C', price='12.00',
         manufacturer='Abbott', color='orange', shape='round', category='Pain Relief/Anti-Inflammatory',
         description='drug brufen generic ibuprofen form film-coated tablet strength 400 pain relief anti-inflammatory'),
    dict(drug_name='Voltaren', generic_name='Diclofenac', strength='50', strength_unit='mg',
         admin_route='Oral', dosage_form='Enteric-Coated Tablet', package_type='Blister',
         legal_class='prescription', shelf_life=36, storage='Store below 30°C', price='20.00',
         manufacturer='Novartis', color='yellow', shape='round', category='Pain Relief/Anti-Inflammatory',
         description='drug voltaren generic diclofenac form enteric-coated tablet strength 50 pain relief anti-inflammatory'),
    dict(drug_name='Losartan Actavis', generic_name='Losartan Potassium', strength='50', strength_unit='mg',
         admin_route='Oral', dosage_form='Film-Coated Tablet', package_type='Blister',
         legal_class='prescription', shelf_life=36, storage='Store below 30°C', price='30.00',
         manufacturer='Actavis', color='white', shape='oval', category='Hypertension',
         description='drug losartan generic losartan potassium form film-coated tablet strength 50 hypertension blood pressure'),
    dict(drug_name='Zantac', generic_name='Ranitidine', strength='150', strength_unit='mg',
         admin_route='Oral', dosage_form='Film-Coated Tablet', package_type='Blister',
         legal_class='OTC', shelf_life=36, storage='Store below 30°C', price='16.00',
         manufacturer='GSK', color='white', shape='round', category='Gastrointestinal',
         description='drug zantac generic ranitidine form film-coated tablet strength 150 gastric acid stomach'),
    dict(drug_name='Omeprazole Capsule', generic_name='Omeprazole', strength='20', strength_unit='mg',
         admin_route='Oral', dosage_form='Gastro-Resistant Capsule', package_type='Blister',
         legal_class='OTC', shelf_life=24, storage='Store below 25°C', price='18.00',
         manufacturer='AstraZeneca', color='purple', shape='capsule', category='Gastrointestinal',
         description='drug omeprazole capsule generic omeprazole form gastro-resistant capsule strength 20 stomach acid'),
    dict(drug_name='Concor', generic_name='Bisoprolol', strength='5', strength_unit='mg',
         admin_route='Oral', dosage_form='Film-Coated Tablet', package_type='Blister',
         legal_class='prescription', shelf_life=36, storage='Store below 30°C', price='28.00',
         manufacturer='Merck', color='yellow', shape='oval', category='Hypertension/Cardiac',
         description='drug concor generic bisoprolol form film-coated tablet strength 5 hypertension heart beta blocker'),
    dict(drug_name='Clexane', generic_name='Enoxaparin Sodium', strength='40', strength_unit='mg',
         admin_route='Subcutaneous', dosage_form='Solution for Injection',
         package_type='Syringe', legal_class='prescription', shelf_life=36,
         storage='Store below 25°C', price='85.00', manufacturer='Sanofi',
         color='', shape='', category='Anticoagulant',
         description='drug clexane generic enoxaparin sodium form solution for injection strength 40 anticoagulant blood clot'),
    dict(drug_name='Prednisolone', generic_name='Prednisolone', strength='5', strength_unit='mg',
         admin_route='Oral', dosage_form='Tablet', package_type='Blister',
         legal_class='prescription', shelf_life=36, storage='Store below 25°C', price='10.00',
         manufacturer='Nycomed', color='white', shape='round', category='Corticosteroid/Anti-Inflammatory',
         description='drug prednisolone form tablet strength 5 corticosteroid anti-inflammatory immune'),
    dict(drug_name='Ciprobay', generic_name='Ciprofloxacin', strength='500', strength_unit='mg',
         admin_route='Oral', dosage_form='Film-Coated Tablet', package_type='Blister',
         legal_class='prescription', shelf_life=36, storage='Store below 25°C', price='38.00',
         manufacturer='Bayer', color='white', shape='oblong', category='Antibiotic',
         description='drug ciprobay generic ciprofloxacin form film-coated tablet strength 500 antibiotic bacteria infection'),
    dict(drug_name='Glucovance', generic_name='Metformin/Glibenclamide', strength='500/2.5', strength_unit='mg',
         admin_route='Oral', dosage_form='Film-Coated Tablet', package_type='Blister',
         legal_class='prescription', shelf_life=36, storage='Store below 30°C', price='35.00',
         manufacturer='Merck', color='yellow', shape='oval', category='Diabetes',
         description='drug glucovance generic metformin glibenclamide form film-coated tablet strength 500 2.5 diabetes'),
    dict(drug_name='Januvia', generic_name='Sitagliptin', strength='100', strength_unit='mg',
         admin_route='Oral', dosage_form='Film-Coated Tablet', package_type='Blister',
         legal_class='prescription', shelf_life=36, storage='Store below 30°C', price='120.00',
         manufacturer='MSD', color='beige', shape='round', category='Diabetes',
         description='drug januvia generic sitagliptin form film-coated tablet strength 100 diabetes blood sugar'),
    dict(drug_name='Amoxil', generic_name='Amoxicillin', strength='500', strength_unit='mg',
         admin_route='Oral', dosage_form='Capsule', package_type='Blister',
         legal_class='prescription', shelf_life=36, storage='Store below 25°C', price='14.00',
         manufacturer='GSK', color='red/yellow', shape='capsule', category='Antibiotic',
         description='drug amoxil generic amoxicillin form capsule strength 500 antibiotic infection'),
    dict(drug_name='Ventolin', generic_name='Salbutamol', strength='100', strength_unit='mcg',
         admin_route='Inhalation', dosage_form='Pressurised Inhalation Suspension',
         package_type='Inhaler', legal_class='prescription', shelf_life=24,
         storage='Store below 30°C, protect from frost', price='25.00',
         manufacturer='GSK', color='blue', shape='', category='Respiratory/Asthma',
         description='drug ventolin generic salbutamol form inhaler strength 100 mcg asthma bronchodilator respiratory'),
    dict(drug_name='Xanax', generic_name='Alprazolam', strength='0.5', strength_unit='mg',
         admin_route='Oral', dosage_form='Tablet', package_type='Blister',
         legal_class='prescription', drug_type='controlled', shelf_life=36,
         storage='Store below 25°C', price='30.00', manufacturer='Pfizer',
         color='white', shape='oval', category='Anxiolytic/Neurological',
         description='drug xanax generic alprazolam form tablet strength 0.5 anxiety neurological'),
    dict(drug_name='Crestor', generic_name='Rosuvastatin', strength='10', strength_unit='mg',
         admin_route='Oral', dosage_form='Film-Coated Tablet', package_type='Blister',
         legal_class='prescription', shelf_life=36, storage='Store below 30°C', price='65.00',
         manufacturer='AstraZeneca', color='pink', shape='round', category='Cholesterol',
         description='drug crestor generic rosuvastatin form film-coated tablet strength 10 cholesterol statin cardiovascular'),
    # ── ADHD drugs ───────────────────────────────────────────────────────────────
    dict(drug_name='Fevadol', generic_name='Paracetamol + Ibuprofen',
         strength='250/200', strength_unit='mg', admin_route='Oral', dosage_form='Tablet',
         package_type='Blister', legal_class='OTC', shelf_life=36,
         storage='Store below 30°C', price='18.00',
         manufacturer='Riyadh Pharma', color='white', shape='round', category='Pain Relief / Antipyretic',
         description='drug fevadol فيفادول generic paracetamol ibuprofen باراسيتامول إيبوبروفين '
                     'tablet قرص pain relief مسكن للألم fever حمى headache صداع '
                     'toothache ألم الأسنان muscle pain آلام العضلات menstrual pain آلام الدورة الشهرية '
                     'analgesic antipyretic خافض للحرارة white أبيض round دائري tablet dual-action'),
    dict(drug_name='Ritalin', generic_name='Methylphenidate',
         strength='10', strength_unit='mg', admin_route='Oral', dosage_form='Tablet',
         package_type='Blister', legal_class='prescription', drug_type='controlled', shelf_life=36,
         storage='Store below 25°C', price='320.00',
         manufacturer='Novartis', color='white', shape='round', category='ADHD',
         description='drug ritalin generic methylphenidate tablet ADHD attention deficit hyperactivity disorder '
                     'stimulant white round tablet concentration focus'),
    dict(drug_name='Strattera', generic_name='Atomoxetine',
         strength='25', strength_unit='mg', admin_route='Oral', dosage_form='Capsule',
         package_type='Blister', legal_class='prescription', shelf_life=36,
         storage='Store below 25°C', price='560.00',
         manufacturer='Lilly', color='gold white', shape='capsule', category='ADHD',
         description='drug strattera generic atomoxetine capsule ADHD attention deficit non-stimulant '
                     'gold white capsule focus hyperactivity'),
    # ── Drug Vision dataset — 10 Philippine OTC drugs ─────────────────────────
    dict(drug_name='Alaxan', generic_name='Ibuprofen + Paracetamol',
         strength='200/325', strength_unit='mg', admin_route='Oral', dosage_form='Tablet',
         package_type='Blister', legal_class='OTC', shelf_life=36,
         storage='Store below 30°C', price='12.00',
         manufacturer='United American Pharmaceuticals',
         color='red white', shape='oval', category='Pain Relief / NSAID',
         long_term_effect=False,
         description='Alaxan red white oval tablet ibuprofen paracetamol dual-action relief '
                     'mild moderate pain headache toothache muscle pain back pain arthritis '
                     'fever NSAID analgesic antipyretic combination'),
    dict(drug_name='Bactidol', generic_name='Hexetidine',
         strength='0.1', strength_unit='%', admin_route='Topical/Oral', dosage_form='Mouthwash Solution',
         package_type='Bottle', legal_class='OTC', shelf_life=36,
         storage='Store below 30°C', price='8.00',
         manufacturer='Johnson and Johnson',
         color='clear blue', shape='liquid', category='Antiseptic / Oral Hygiene',
         long_term_effect=False,
         description='Bactidol clear blue antiseptic mouthwash solution hexetidine '
                     'sore throat mouth infection oral hygiene gargle tonsillitis '
                     'antibacterial oral care liquid'),
    dict(drug_name='Bioflu', generic_name='Phenylephrine + Chlorphenamine + Paracetamol',
         strength='10/2/500', strength_unit='mg', admin_route='Oral', dosage_form='Tablet',
         package_type='Blister', legal_class='OTC', shelf_life=36,
         storage='Store below 30°C', price='7.00',
         manufacturer='Pascual Laboratories',
         color='yellow', shape='oval', category='Cold and Flu Remedy',
         long_term_effect=False,
         description='Bioflu yellow oval tablet phenylephrine chlorphenamine paracetamol '
                     'flu cold fever runny nose nasal congestion sneezing chills '
                     'decongestant antihistamine antipyretic cold flu remedy'),
    dict(drug_name='Biogesic', generic_name='Paracetamol',
         strength='500', strength_unit='mg', admin_route='Oral', dosage_form='Tablet',
         package_type='Blister', legal_class='OTC', shelf_life=36,
         storage='Store below 25°C', price='5.00',
         manufacturer='United American Pharmaceuticals',
         color='white', shape='round', category='Pain Relief / Antipyretic',
         long_term_effect=False,
         description='Biogesic white round tablet paracetamol 500mg fever headache body pain '
                     'mild moderate pain relief antipyretic analgesic safe common'),
    dict(drug_name='DayZinc', generic_name='Zinc Sulfate',
         strength='20', strength_unit='mg', admin_route='Oral', dosage_form='Capsule',
         package_type='Blister', legal_class='OTC', shelf_life=36,
         storage='Store below 30°C', price='6.00',
         manufacturer='Pascual Laboratories',
         color='orange', shape='capsule', category='Vitamin / Mineral Supplement',
         long_term_effect=False,
         description='DayZinc orange capsule zinc sulfate supplement immune support '
                     'mineral deficiency immunity boost daily vitamin mineral supplement'),
    dict(drug_name='Decolgen', generic_name='Phenylpropanolamine + Chlorphenamine',
         strength='25/2', strength_unit='mg', admin_route='Oral', dosage_form='Tablet',
         package_type='Blister', legal_class='OTC', shelf_life=36,
         storage='Store below 30°C', price='5.00',
         manufacturer='United American Pharmaceuticals',
         color='white', shape='round', category='Cold / Decongestant / Antihistamine',
         long_term_effect=False,
         description='Decolgen white round tablet phenylpropanolamine chlorphenamine '
                     'colds nasal congestion runny nose decongestant antihistamine '
                     'sneezing allergy rhinitis cold'),
    dict(drug_name='Fish Oil', generic_name='Omega-3 Fatty Acids (EPA + DHA)',
         strength='1000', strength_unit='mg', admin_route='Oral', dosage_form='Softgel Capsule',
         package_type='Bottle', legal_class='OTC', shelf_life=36,
         storage='Store below 30°C', price='15.00',
         manufacturer='Pascual Laboratories',
         color='yellow gold', shape='oval', category='Nutritional / Omega-3 Supplement',
         long_term_effect=False,
         description='Fish Oil yellow gold oval softgel capsule omega-3 fatty acids EPA DHA '
                     'heart health supplement triglycerides cardiovascular brain function nutritional'),
    dict(drug_name='Kremil S', generic_name='Aluminum Hydroxide + Magnesium Hydroxide + Simethicone',
         strength='225/200/25', strength_unit='mg', admin_route='Oral', dosage_form='Chewable Tablet',
         package_type='Blister', legal_class='OTC', shelf_life=36,
         storage='Store below 30°C', price='7.00',
         manufacturer='Pascual Laboratories',
         color='white green', shape='round', category='Antacid / Gastrointestinal',
         long_term_effect=False,
         description='Kremil S white green chewable tablet aluminum hydroxide magnesium hydroxide '
                     'simethicone antacid heartburn stomach acid indigestion ulcer gas bloating hyperacidity'),
    dict(drug_name='Medicol', generic_name='Ibuprofen',
         strength='200', strength_unit='mg', admin_route='Oral', dosage_form='Capsule',
         package_type='Blister', legal_class='OTC', shelf_life=36,
         storage='Store below 30°C', price='8.00',
         manufacturer='Pascual Laboratories',
         color='red white', shape='capsule', category='Pain Relief / NSAID / Antipyretic',
         long_term_effect=False,
         description='Medicol red white capsule ibuprofen 200mg NSAID analgesic antipyretic '
                     'mild moderate pain headache muscle pain menstrual dysmenorrhea toothache fever'),
    dict(drug_name='Neozep', generic_name='Phenylephrine + Chlorphenamine + Paracetamol',
         strength='10/2/325', strength_unit='mg', admin_route='Oral', dosage_form='Tablet',
         package_type='Blister', legal_class='OTC', shelf_life=36,
         storage='Store below 30°C', price='5.00',
         manufacturer='United American Pharmaceuticals',
         color='white', shape='round', category='Cold and Flu Remedy',
         long_term_effect=False,
         description='Neozep white round tablet phenylephrine chlorphenamine paracetamol '
                     'colds flu fever nasal congestion runny nose allergy sneezing body pain'),
]


def init_db():
    """Initialize database, create tables, and seed with demo data."""
    with app.app_context():
        db.create_all()

        # Seed admin user
        if not User.query.filter_by(username='admin').first():
            admin = User(username='admin', email='admin@capsule.sa',
                         role='admin', full_name='System Administrator',
                         hospital='Taif University Hospital')
            admin.set_password('admin123')
            db.session.add(admin)

        if not User.query.filter_by(username='doctor').first():
            doctor = User(username='doctor', email='doctor@capsule.sa',
                          role='doctor', full_name='Dr. Ahmad Al-Zahrani',
                          hospital='King Faisal Hospital', specialty='Internal Medicine')
            doctor.set_password('doctor123')
            db.session.add(doctor)

        if not User.query.filter_by(username='pharmacist').first():
            pharm = User(username='pharmacist', email='pharmacist@capsule.sa',
                         role='pharmacist', full_name='Sara Al-Otaibi',
                         hospital='National Guard Hospital', specialty='Clinical Pharmacy')
            pharm.set_password('pharma123')
            db.session.add(pharm)

        if not User.query.filter_by(username='testuser').first():
            user = User(username='testuser', email='user@capsule.sa',
                        role='patient', full_name='Test User')
            user.set_password('test123')
            db.session.add(user)

        # Seed patient demo user
        if not User.query.filter_by(username='patient').first():
            patient = User(username='patient', email='patient@capsule.sa',
                           role='patient', full_name='Fatima Al-Otaibi')
            patient.set_password('patient123')
            db.session.add(patient)

        # Seed medications — upsert so new drugs added to SEED_DRUGS are always present
        for d in SEED_DRUGS:
            if not Medication.query.filter(
                Medication.drug_name.ilike(d['drug_name'])
            ).first():
                db.session.add(Medication(**d))

        # Seed all 210 medications from capsule_dataset.csv (bilingual KSA dataset)
        csv_path = os.path.join(BASE_DIR, 'DataSet', 'capsule_dataset.csv')
        if os.path.isfile(csv_path):
            try:
                csv_df = pd.read_csv(csv_path)
                images_dir = os.path.join(BASE_DIR, 'DataSet', 'medication_images')
                added = 0
                for _, row in csv_df.iterrows():
                    name_en = str(row.get('drug_name_en', '')).strip()
                    if not name_en:
                        continue
                    # Skip if already seeded (case-insensitive)
                    if Medication.query.filter(
                        Medication.drug_name.ilike(name_en)
                    ).first():
                        continue
                    # Resolve image path
                    img_rel = str(row.get('image_url', '')).strip()
                    img_abs = ''
                    if img_rel:
                        candidate = os.path.join(BASE_DIR, 'DataSet', img_rel)
                        if os.path.isfile(candidate):
                            img_abs = candidate
                    # Parse strength (e.g. "5 mg" → strength="5", unit="mg")
                    raw_str = str(row.get('strength', '')).strip()
                    parts = raw_str.split(None, 1)
                    str_val = parts[0] if parts else raw_str
                    str_unit = parts[1] if len(parts) > 1 else ''
                    med = Medication(
                        drug_name=name_en,
                        drug_name_ar=str(row.get('drug_name_ar', '')).strip(),
                        generic_name=str(row.get('generic_name_en', '')).strip(),
                        generic_name_ar=str(row.get('generic_name_ar', '')).strip(),
                        salt_composition=str(row.get('salt_composition', '')).strip(),
                        dosage_form=str(row.get('dosage_form_en', '')).strip(),
                        strength=str_val,
                        strength_unit=str_unit,
                        indications=str(row.get('indications_en', '')).strip(),
                        indications_ar=str(row.get('indications_ar', '')).strip(),
                        manufacturer=str(row.get('manufacturer_en', '')).strip(),
                        side_effects=str(row.get('side_effects_en', '')).strip(),
                        side_effects_ar=str(row.get('side_effects_ar', '')).strip(),
                        contraindications=str(row.get('contraindications_en', '')).strip(),
                        contraindications_ar=str(row.get('contraindications_ar', '')).strip(),
                        category=str(row.get('drug_class_en', '')).strip(),
                        legal_class=str(row.get('rx_otc_status_en', '')).strip(),
                        color=str(row.get('color_en', '')).strip(),
                        shape=str(row.get('shape_en', '')).strip(),
                        imprint_code=str(row.get('imprint_code', '')).strip(),
                        image_path=img_abs,
                        description=(
                            f"drug {name_en.lower()} generic {str(row.get('generic_name_en','')).lower()} "
                            f"form {str(row.get('dosage_form_en','')).lower()} "
                            f"color {str(row.get('color_en','')).lower()} "
                            f"shape {str(row.get('shape_en','')).lower()} "
                            f"category {str(row.get('drug_class_en','')).lower()} "
                            f"{str(row.get('indications_en','')).lower()} "
                            f"{str(row.get('drug_name_ar',''))} "
                            f"{str(row.get('generic_name_ar',''))} "
                            f"{str(row.get('indications_ar',''))}"
                        ),
                        is_active=True,
                    )
                    db.session.add(med)
                    added += 1
                if added > 0:
                    db.session.flush()
                    print(f"[CAPSULE] Seeded {added} medications from capsule_dataset.csv")
            except Exception as e:
                print(f"[CAPSULE] CSV seed error: {e}")

        # Seed long-term effect data onto existing/new drugs
        try:
            from long_term_effects import LONG_TERM_DRUG_EFFECTS
            for name_key, info in LONG_TERM_DRUG_EFFECTS.items():
                existing = Medication.query.filter(
                    Medication.drug_name.ilike(f'%{name_key.split("(")[0].strip()}%')
                ).first()
                if existing and not existing.long_term_effect:
                    existing.long_term_effect = True
                    existing.long_term_description = info.get('description', '')
                    months = info.get('min_duration_months', 0)
                    existing.long_term_duration = f'{months} months+' if months else ''
                    existing.requires_warning = True
                    existing.warning_text = info.get('patient_guidance', '')
        except Exception:
            pass

        # Seed drug interactions
        if DrugInteraction.query.count() == 0:
            INTERACTIONS = [
                ('Warfarin', 'Aspirin', 'high',
                 'Increased risk of serious bleeding. Aspirin inhibits platelet aggregation and displaces warfarin from protein binding.',
                 'Additive anticoagulant and antiplatelet effects',
                 'Stop immediately and contact your doctor — serious bleeding risk.',
                 'Monitor INR closely; consider alternative analgesic'),
                ('Warfarin', 'Ibuprofen', 'high',
                 'NSAIDs inhibit platelet function and can cause GI bleeding while increasing warfarin levels.',
                 'CYP2C9 inhibition + antiplatelet effect',
                 'Do NOT take together — dangerous bleeding risk. Contact your doctor.',
                 'Avoid combination; if needed, use paracetamol and monitor INR'),
                ('Warfarin', 'Naproxen', 'high',
                 'Increased anticoagulant effect and GI bleeding risk.',
                 'Same as other NSAIDs',
                 'Increased bleeding risk — contact your doctor before taking.',
                 'Avoid; monitor INR if unavoidable'),
                ('Warfarin', 'Fluconazole', 'high',
                 'Fluconazole significantly inhibits CYP2C9, greatly increasing warfarin plasma levels.',
                 'CYP2C9 inhibition',
                 'This combination greatly increases your bleeding risk. Tell your doctor immediately.',
                 'Reduce warfarin dose by ~50%; monitor INR daily'),
                ('Warfarin', 'Vitamin K', 'medium',
                 'Vitamin K supplements reduce warfarin effectiveness.',
                 'Competitive reversal of warfarin action',
                 'Keep your vitamin K intake consistent and inform your doctor about supplements.',
                 'Monitor INR; adjust warfarin dose accordingly'),
                ('Methotrexate', 'Ibuprofen', 'high',
                 'NSAIDs reduce methotrexate renal clearance, causing toxic accumulation.',
                 'Reduced renal tubular secretion',
                 'Do NOT take ibuprofen with methotrexate — risk of serious toxicity. Contact your doctor.',
                 'Avoid NSAIDs; use paracetamol if analgesia needed'),
                ('Lithium', 'Ibuprofen', 'high',
                 'NSAIDs reduce lithium renal clearance, causing lithium toxicity.',
                 'Reduced renal clearance of lithium',
                 'Do NOT take ibuprofen while on lithium — contact your doctor.',
                 'Monitor serum lithium levels; avoid NSAIDs'),
                ('SSRIs', 'Tramadol', 'high',
                 'Both increase serotonin. Combined use risks life-threatening serotonin syndrome.',
                 'Additive serotonergic effect',
                 'Dangerous combination — risk of serotonin syndrome. Contact your doctor immediately.',
                 'Avoid combination; if pain control needed, use alternative opioid'),
                ('Statins', 'Gemfibrozil', 'high',
                 'Gemfibrozil inhibits statin metabolism, greatly increasing risk of severe muscle damage (rhabdomyolysis).',
                 'CYP inhibition + OATP1B1 inhibition',
                 'Do NOT take these together — severe muscle damage risk. Inform your doctor.',
                 'Contraindicated; use fenofibrate instead if combination lipid therapy needed'),
                ('Amiodarone', 'Digoxin', 'high',
                 'Amiodarone increases digoxin plasma levels, causing toxicity.',
                 'P-glycoprotein inhibition reduces digoxin excretion',
                 'This combination can be toxic — do not change doses without your doctor.',
                 'Reduce digoxin dose by 50% when starting amiodarone; monitor digoxin levels'),
                ('ACE Inhibitor', 'Spironolactone', 'high',
                 'Both increase potassium levels, leading to dangerous hyperkalemia.',
                 'Additive potassium-sparing effects',
                 'Dangerous potassium build-up possible — contact your doctor before taking together.',
                 'Monitor serum potassium closely; use lowest effective doses'),
                ('SSRIs', 'Ibuprofen', 'medium',
                 'NSAIDs plus SSRIs increase risk of GI bleeding.',
                 'Antiplatelet effect of SSRIs + NSAIDs',
                 'Increased bleeding risk — discuss with your doctor or pharmacist.',
                 'Use gastroprotective agent if combination unavoidable'),
                ('Metformin', 'Alcohol', 'medium',
                 'Alcohol increases risk of lactic acidosis with metformin.',
                 'Inhibition of lactate metabolism',
                 'Avoid alcohol while taking metformin — risk of serious side effects.',
                 'Counsel patient to avoid excessive alcohol'),
                ('ACE Inhibitor', 'Potassium', 'medium',
                 'ACE inhibitors retain potassium; supplements can cause dangerous levels.',
                 'Additive hyperkalemia risk',
                 'Excess potassium may build up — mention any supplements to your doctor.',
                 'Monitor serum potassium; avoid routine potassium supplements'),
                ('Corticosteroids', 'Ibuprofen', 'medium',
                 'Both can cause GI bleeding; combination greatly increases risk.',
                 'Additive GI mucosal damage',
                 'Avoid taking ibuprofen with steroids unless your doctor advises.',
                 'Use PPI cover if combination unavoidable'),
                ('Statins', 'CYP3A4 Inhibitor', 'medium',
                 'CYP3A4 inhibitors increase statin blood levels, raising rhabdomyolysis risk.',
                 'CYP3A4 inhibition increases statin exposure',
                 'Monitor for unexplained muscle pain or weakness — contact your doctor.',
                 'Use lowest effective statin dose; consider statin not metabolised by CYP3A4'),
                ('Metformin', 'Contrast Dye', 'medium',
                 'Contrast agents can impair renal function, causing metformin accumulation and lactic acidosis.',
                 'Renal impairment from contrast nephropathy',
                 'Always inform your radiologist and doctor that you take metformin before any scan.',
                 'Hold metformin 48h before and after contrast procedures; check renal function'),
                ('Antacids', 'Ciprofloxacin', 'low',
                 'Antacids chelate ciprofloxacin, reducing its absorption by up to 90%.',
                 'Chelation of fluoroquinolone',
                 'Take your antibiotic at least 2 hours before or 6 hours after antacids.',
                 'Separate doses by ≥2h; counsel patient'),
                ('Antacids', 'Levothyroxine', 'low',
                 'Antacids reduce levothyroxine absorption.',
                 'Adsorption/chelation',
                 'Take levothyroxine on an empty stomach, 4 hours apart from antacids.',
                 'Separate administration by ≥4h; check TSH levels'),
                ('Digoxin', 'Amiodarone', 'high',
                 'Duplicate entry — same as Amiodarone/Digoxin above.',
                 'P-glycoprotein inhibition',
                 'Digoxin toxicity risk — do not change doses without your doctor.',
                 'Reduce digoxin dose; monitor levels'),
            ]
            for row in INTERACTIONS:
                di = DrugInteraction(
                    drug_a_name=row[0], drug_b_name=row[1], risk_level=row[2],
                    interaction_description=row[3], mechanism=row[4],
                    patient_guidance=row[5], clinical_note=row[6],
                )
                db.session.add(di)

        db.session.commit()
        print("[CAPSULE] Database initialized and seeded.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
    # Flask debug reloader spawns a child process with WERKZEUG_RUN_MAIN='true'.
    # The parent must NOT touch the DB, otherwise its connection pool locks SQLite
    # and the child hangs. Only init DB in the child (or when reloader is disabled).
    _use_debug = True
    if not _use_debug or os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        init_db()
    print("\n" + "="*60)
    print("  CAPSULE - AI Medication Identification System")
    print("  Taif University · Vision 2030")
    print("="*60)
    print("  URL: http://127.0.0.1:5000")
    print("  Demo logins:")
    print("    admin / admin123")
    print("    doctor / doctor123")
    print("    pharmacist / pharma123")
    print("="*60 + "\n")
    app.run(debug=_use_debug, host='0.0.0.0', port=5001, use_reloader=False)
