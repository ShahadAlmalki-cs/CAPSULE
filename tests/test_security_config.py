from pathlib import Path


def test_env_example_exists():
    assert Path('.env.example').exists()


def test_gitignore_keeps_env_out():
    gitignore = Path('.gitignore').read_text(encoding='utf-8')
    assert '.env' in gitignore
    assert '.env.*' in gitignore
    assert '!.env.example' in gitignore


def test_no_hardcoded_demo_credentials_in_source():
    source = Path('app.py').read_text(encoding='utf-8', errors='ignore')
    assert 'capsule-taif-university-2025-secret' not in source
    assert 'admin123' not in source
    assert 'doctor123' not in source
    assert 'pharma123' not in source
    assert 'patient123' not in source
