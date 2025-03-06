# save as check_versions.py
import pkg_resources

required = {
    "numpy": "2.0.2",
    "pandas": "2.2.3",
    "yfinance": "0.2.54",
    "transformers": "4.49.0",
    "torch": "2.6.0",
    "spacy": "3.8.3",
    "PyPDF2": "3.0.1",
    "scikit-learn": "1.6.1",
    "matplotlib": "3.9.4",
    "seaborn": "0.13.2",
    "nltk": "3.9.1"
}

for pkg, version in required.items():
    installed = pkg_resources.get_distribution(pkg).version
    print(f"{pkg}: required {version}, installed {installed}")