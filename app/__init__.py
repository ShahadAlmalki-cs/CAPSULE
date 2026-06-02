# app/__init__.py
# ---------------------------------------------------------------------------
# The monolithic Flask application lives in app.py (project root).
# Python treats the app/ directory as a package, which shadows app.py.
# This file loads app.py explicitly so that:
#   from app import app, db, User, Medication, ...
# continues to work from any entry point.
# ---------------------------------------------------------------------------
import importlib.util
import os
import sys

# If app.py was already loaded as __main__ (i.e. `python app.py`), reuse it
# to avoid creating duplicate Flask/SQLAlchemy instances.
_mod = None
if hasattr(sys.modules.get('__main__', None), 'app') and hasattr(sys.modules['__main__'], 'db'):
    _main = sys.modules['__main__']
    # Verify it's actually our app.py
    main_file = getattr(_main, '__file__', '')
    if main_file and os.path.basename(main_file) == 'app.py':
        _mod = _main

if _mod is None and '_capsule_legacy' in sys.modules:
    _mod = sys.modules['_capsule_legacy']

if _mod is None:
    _app_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app.py')
    _spec = importlib.util.spec_from_file_location('_capsule_legacy', _app_file)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules['_capsule_legacy'] = _mod
    _spec.loader.exec_module(_mod)

# Re-export all public names from app.py
app = _mod.app
db = _mod.db
init_db = _mod.init_db

# Database models
User = _mod.User
Medication = _mod.Medication
SearchHistory = _mod.SearchHistory
SystemLog = _mod.SystemLog
PatientMedication = _mod.PatientMedication
DrugInteraction = _mod.DrugInteraction
DispenseLog = _mod.DispenseLog
