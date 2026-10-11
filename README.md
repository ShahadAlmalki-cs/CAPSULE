# CAPSULE

CAPSULE is an AI-powered medication identification and medication safety support system developed as a Senior Capstone Project at Taif University.

The project helps patients, doctors, and pharmacists identify medicines from simple text descriptions.

## Security and configuration

Before running the app, create a local `.env` file from `.env.example` and set your own values.

```bash
cp .env.example .env
```

Do not commit `.env` files or real secrets. The project must not contain hardcoded production credentials or fixed secret keys in source code.

## Local development

### 1. Clone the project

```bash
git clone https://github.com/ShahadAlmalki-cs/CAPSULE.git
cd CAPSULE
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install requirements

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Update `.env` with real values for your local environment. Example keys are listed in `.env.example`.

### 5. Run the project

```bash
python app.py
```

Then open the local URL reported in the terminal, usually:

```text
http://127.0.0.1:5001/
```

## Security notes

- The application must use environment variables for secret configuration.
- Demo credentials should only be enabled in local development when explicitly configured.
- `SECRET_KEY`, database credentials, and admin passwords must never be committed into source control.

## Project overview

CAPSULE supports medication identification from descriptions, ranking candidate matches, checking medication interactions, and helping clinicians and patients make better medication decisions.
