from setuptools import setup
from pathlib import Path
import re

# Extract version without importing the module (avoids dependency issues during build)
def get_version():
    version_file = Path(__file__).parent / 'ai_ops_cli.py'
    with open(version_file, 'r') as f:
        content = f.read()
        match = re.search(r'^VERSION\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
        if match:
            return match.group(1)
        raise RuntimeError("Unable to find version string in ai_ops_cli.py")

CON = {}
DIR = Path(__file__).parent
with open(str(DIR / 'requirements-cli.txt'), 'r') as fp:
    lines = fp.read().splitlines()
    CON['requirements-cli'] = lines if lines else []

setup(
    name='ai-ops-cli',
    version=get_version(),
    author='@antoninoLorenzo',
    url='https://github.com/antoninoLorenzo/AI-OPS',
    install_requires=CON['requirements-cli'],
    py_modules=['ai_ops_cli'],
    entry_points={
        'console_scripts': ['ai_ops_cli=ai_ops_cli:main']
    },
)
