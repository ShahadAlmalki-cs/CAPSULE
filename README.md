# CAPSULE

CAPSULE is an AI-powered medication identification and medication safety support system developed as a Senior Capstone Project at Taif University.

The system helps patients, doctors, and pharmacists identify medications from simple text descriptions when the medication name is forgotten or unclear. Users can describe a medication using details such as color, shape, dosage form, and intended use. CAPSULE analyzes the description and returns ranked medication suggestions with relative matching scores.

CAPSULE currently focuses on text-based medication identification.

---

## Project Overview

Medication identification can be difficult when patients cannot remember the name of a medicine. This may lead to confusion, delayed decisions, repeated prescriptions, or medication safety risks.

CAPSULE supports this process by allowing users to enter a natural language medication description. The system processes the text, extracts useful attributes, compares the input with a curated medication database, and displays the most relevant medication matches.

The project also supports medication safety through drug interaction checking, long-term medication warnings, medication history, and role-based dashboards.

---

## Main Features

- Text-based medication identification
- Natural language processing for medication descriptions
- Attribute extraction such as color, shape, dosage form, and intended use
- Ranked medication suggestions with relative matching scores
- Drug interaction checker with risk-level alerts
- Long-term medication warnings
- Patient medication list and medication history
- Search history for identification queries
- Role-based dashboards for patients, doctors, pharmacists, and administrators
- Secure login and role-based access control

---

## User Roles

### Patient

Patients can:

- Search for medications using text descriptions
- View ranked medication suggestions
- Add confirmed medications to their personal medication list
- Manage current and previous medications
- View medication history
- Receive interaction and safety alerts

### Doctor

Doctors can:

- Identify medications from patient descriptions
- Review ranked medication suggestions
- Open detailed medication information
- Use search history to review previous identification attempts
- Support clinical decision-making using medication details and matching results

### Pharmacist

Pharmacists can:

- Search and review medication records
- Identify medications from descriptions
- Check drug interactions between medications
- Review medication details
- Record dispense information
- Support safe medication use and patient counseling

### Admin

Administrators can:

- Manage system records
- Review users and system activity
- Support system maintenance and data management

---

## How CAPSULE Works

1. The user enters a medication description.

Example:

```text
yellow oval tablet used for cold flu fever runny nose and nasal congestion
```

2. The system processes the text.

CAPSULE normalizes the input and extracts useful attributes such as color, shape, dosage form, symptoms, and intended use.

3. The system applies AI-based matching.

CAPSULE uses NLP processing, TF-IDF similarity matching, attribute matching, and supporting machine learning components to compare the description with medication records.

4. The system displays ranked results.

The user receives possible medication matches with relative matching scores.

5. The user reviews details.

Users can open medication detail pages to review drug name, generic name, dosage form, strength, category, and related information.

6. Medication safety features support decision-making.

The system can show interaction alerts and long-term warning information when relevant.

---

## Matching Score Explanation

The percentage shown in the results is a relative matching score.

It represents how strongly the user description matches the medication record in the database. It is not a medical diagnosis and it is not a guaranteed probability of correctness.

The matching score may consider:

- Medication name match
- Color match
- Shape match
- Dosage form match
- Intended use or symptom match
- Category similarity
- Text similarity between the query and medication description

Examples:

- A single word such as `red` or `fever` returns low-confidence possible matches.
- A physical description such as `yellow oval tablet` returns possible matches with limited confidence.
- A detailed description with color, shape, dosage form, and use returns stronger matches.
- An exact medication name returns a very high matching score.

---

## Example Test Cases

### Medication identification

```text
small round white tablet used for fever headache and pain relief
```

Expected result: Biogesic or a similar paracetamol-based medication.

```text
yellow oval tablet used for cold flu fever runny nose and nasal congestion
```

Expected result: Bioflu or a similar cold and flu medication.

```text
clear blue antiseptic mouthwash solution used for sore throat mouth infection oral hygiene
```

Expected result: Bactidol or a similar antiseptic mouthwash.

```text
orange capsule used as zinc supplement
```

Expected result: DayZinc or a similar zinc supplement.

### Low-information descriptions

```text
red
fever
yellow oval tablet
white round tablet
```

Expected result: possible matches with lower confidence.

### Invalid or unclear inputs

```text
asdfghjkl
تينيمصمرثنيني
```

Expected result: no match or an unclear input response.

### Contradictory dosage forms

```text
cream tablet
capsule syrup
```

Expected result: the system should detect conflicting dosage form descriptions.

### Drug interaction examples

```text
Warfarin + Ibuprofen
Warfarin + Naproxen
Fevadol + Warfarin
Aspirin + Ibuprofen
```

Expected result: interaction warning if the interaction exists in the system database.

---

## Technology Stack

### Backend

- Python
- Flask
- SQLAlchemy
- SQLite
- Flask-Login
- Werkzeug

### AI and NLP

- Natural language processing
- TF-IDF similarity matching
- Attribute extraction
- scikit-learn
- PyTorch support for deep learning components

### Frontend

- HTML5
- CSS3
- JavaScript
- Font Awesome
- Responsive web design

### Database

- SQLite medication database
- Drug interaction records
- Long-term warning records
- User and role-based account data
- Patient medication history
- Search history records

---

## Project Structure

```text
CAPSULE_final/
│
├── app.py
├── capsule_database.db
├── drug_interaction_checker.py
├── long_term_effects.py
├── requirements.txt
├── README.md
│
├── AI_models/
│   ├── ai_model.py
│   ├── nlp_processor.py
│   └── trained_models/
│
├── templates/
│   ├── base.html
│   ├── landing.html
│   ├── about.html
│   ├── identify.html
│   ├── medication_detail.html
│   ├── history.html
│   ├── login.html
│   ├── register.html
│   ├── doctor_dashboard.html
│   ├── pharmacist_dashboard.html
│   ├── dashboard.html
│   ├── patient/
│   ├── pharmacist/
│   └── errors/
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
└── DataSet/
```

---

## Installation and Local Run

### 1. Download or clone the project

Place the project folder on your device.

### 2. Create a virtual environment

Using Anaconda:

```bash
conda create -n capsule_env python=3.11
conda activate capsule_env
```

Or using Python venv:

```bash
python -m venv capsule_env
capsule_env\Scripts\activate
```

### 3. Install requirements

```bash
pip install -r requirements.txt
```

### 4. Run the project

```bash
python app.py
```

Then open the local link shown in the terminal, such as:

```text
http://127.0.0.1:5001/
```

The port may be different depending on the configuration used in `app.py`.

---

## Replit Deployment Notes

CAPSULE can be deployed on Replit after ensuring that:

- `app.py` is the main application file
- `requirements.txt` includes all required packages
- `templates/` and `static/` folders are included
- `capsule_database.db` is included if SQLite is used directly
- The run command is correctly configured

Typical run command:

```bash
python app.py
```

For production deployment, additional configuration may be required depending on the hosting environment.

---

## Important Notes

- CAPSULE is a capstone prototype and is not intended to replace professional medical judgment.
- The system provides decision-support suggestions based on available database records.
- Match scores are relative similarity scores, not guaranteed probabilities.
- Medication suggestions should be reviewed by qualified healthcare professionals before use in real medical decisions.
- The current working scope is text-based medication identification.
- Image-based medication recognition is not included in the current working implementation.

---

## Project Context

CAPSULE supports the digital transformation of healthcare by using AI and data-driven methods to improve medication identification and medication safety.

The project aligns with Saudi Vision 2030 by contributing to smart healthcare solutions, technology-based healthcare services, and improved digital support for healthcare providers and patients.

---

## Developed By

CAPSULE was developed as a Senior Capstone Project by Computer Science students at Taif University.

Supervisor: Dr. Sabah Alzahrani  
College of Computers and Information Technology  
Taif University, Saudi Arabia