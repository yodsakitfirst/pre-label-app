from pathlib import Path
import subprocess
import sys


def test_cli_doctor_reports_environment_without_models():
    result = subprocess.run([sys.executable, '-m', 'prelabel', 'doctor'], capture_output=True, text=True)
    assert result.returncode == 0
    assert 'python' in result.stdout.lower()
    assert 'platform' in result.stdout.lower()


def test_cli_help_exposes_batch_and_model_setup():
    result = subprocess.run([sys.executable, '-m', 'prelabel', '--help'], capture_output=True, text=True)
    assert result.returncode == 0
    assert 'serve' in result.stdout and 'download-model' in result.stdout and 'batch' in result.stdout
