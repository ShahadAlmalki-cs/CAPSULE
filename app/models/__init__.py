# app/models/__init__.py
# Re-export database models from the monolithic app.py
from app import Medication, User, SearchHistory, SystemLog, PatientMedication, DrugInteraction, DispenseLog
