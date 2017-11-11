import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # project root
DATA_DIR = os.path.join(ROOT_DIR, 'data')
DATA_LOCAL_DIR = os.path.join(ROOT_DIR, 'data_local')
SENTIMENT_ANALYSIS_DATA_DIR = os.path.join(DATA_DIR, 'text_analysis', 'sentiment_analysis')
REPORTS_LOCAL_DIR = os.path.join(DATA_LOCAL_DIR, 'reports')
POLIMORF_DB_PATH = os.path.join(DATA_DIR, 'lemmatizers', 'morfologik', 'polish.db')
