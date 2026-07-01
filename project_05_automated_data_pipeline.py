"""
automated_data_pipeline.py
by Muhammad Ejaz
Date: 24-11-2025

End-to-end reproducible pipeline that:
- ingests CSV files from a folder
- preprocesses and generates features
- retrains and validates a scikit-learn model
- saves/version models with joblib
- archives processed files and artifacts
- supports scheduling with `schedule` or one-off runs

How it works (quick):
- Place incoming CSV(s) in `data/incoming/`. Processed files move to `data/processed/`.
- Models are written to `models/` with timestamped filenames and a metadata JSON.
- Best model info is kept in `models/best_model.json`.

Dependencies:
- pandas, numpy, scikit-learn, joblib, schedule
- optional: sqlalchemy (for recording runs in sqlite)

Run examples:
- One run now: python automated_data_pipeline.py --run_once
- Start scheduler (runs every day at midnight by default): python automated_data_pipeline.py --schedule

"""

import argparse
import json
import os
import shutil
import time
from datetime import datetime
from glob import glob

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

try:
    import schedule
except Exception:
    schedule = None

# Optional: record metadata in sqlite (simple)
try:
    from sqlalchemy import (Column, DateTime, Float, Integer, MetaData, String,
                            Table, create_engine)
    SQLALCHEMY_AVAILABLE = True
except Exception:
    SQLALCHEMY_AVAILABLE = False

# ---------------------------
# Configuration
# ---------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INCOMING_DIR = os.path.join(DATA_DIR, "incoming")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(INCOMING_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

BEST_MODEL_METAFILE = os.path.join(MODELS_DIR, "best_model.json")
RUNS_DB = os.path.join(BASE_DIR, "runs.db")

# ---------------------------
# Simple logger
# ---------------------------
import logging

logger = logging.getLogger("auto_pipeline")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join(LOGS_DIR, "pipeline.log"))
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(fh)
    logger.addHandler(ch)

# ---------------------------
# Helper utilities
# ---------------------------

def read_csvs_from_incoming():
    """Read all CSV files from INCOMING_DIR and return concatenated DataFrame and list of file paths."""
    pattern = os.path.join(INCOMING_DIR, "*.csv")
    files = sorted(glob(pattern))
    if not files:
        logger.info("No CSV files found in %s", INCOMING_DIR)
        return pd.DataFrame(), []

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df['_source_file'] = os.path.basename(f)
            dfs.append(df)
            logger.info("Loaded %s rows from %s", len(df), f)
        except Exception as e:
            logger.exception("Failed reading %s: %s", f, e)
    if not dfs:
        return pd.DataFrame(), []
    combined = pd.concat(dfs, ignore_index=True, sort=False)
    return combined, files


def archive_files(files):
    """Move processed files to processed dir with timestamp suffix."""
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    for f in files:
        try:
            dest = os.path.join(PROCESSED_DIR, os.path.basename(f) + "." + ts)
            shutil.move(f, dest)
            logger.info("Archived %s -> %s", f, dest)
        except Exception:
            logger.exception("Failed to archive %s", f)


# ---------------------------
# Preprocessing & features
# ---------------------------

def basic_preprocess(df):
    """Simple preprocessing:
    - drop rows with all-NA
    - fill numeric NAs with median
    - fill categorical NAs with 'missing'
    - automatically encode simple categorical columns (low cardinality)

    The function returns (X, y) where y is a column named 'target' if present.
    If 'target' not present, we will generate a synthetic target for demo purposes.
    """
    if df.empty:
        return pd.DataFrame(), pd.Series(dtype=int)

    df = df.copy()
    df.dropna(how='all', inplace=True)

    # If 'target' not present, create a synthetic one for demo
    synthetic_target = False
    if 'target' not in df.columns:
        synthetic_target = True
        # create synthetic target based on hash of first column or row index
        if df.shape[1] >= 1:
            first_col = df.columns[0]
            df['target'] = (pd.util.hash_pandas_object(df[first_col], index=False) % 2).astype(int)
        else:
            df['target'] = (np.arange(len(df)) % 2).astype(int)
        logger.info("No 'target' column found — created synthetic target for demo")

    # Separate y
    y = df['target'].copy()
    X = df.drop(columns=['target', '_source_file'], errors='ignore')

    # Identify numeric vs categorical
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # Fill missing numeric with median
    for c in numeric_cols:
        med = X[c].median()
        X[c].fillna(med, inplace=True)

    # Fill categorical
    for c in categorical_cols:
        X[c].fillna('missing', inplace=True)
        # simple low-cardinality encoding
        if X[c].nunique() <= 20:
            X[c] = X[c].astype(str)
            X[c] = X[c].astype('category').cat.codes
        else:
            # drop very high cardinality categorical column
            X.drop(columns=[c], inplace=True)
            logger.info("Dropped high-card col %s (cardinality=%d)", c, X.get(c, pd.Series()).nunique() if c in X else 0)

    return X, y


# ---------------------------
# Model train / validation
# ---------------------------

def train_and_validate(X, y, random_state=42):
    if X.empty:
        logger.warning("Empty feature matrix; skipping training")
        return None, {"accuracy": 0.0}

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y if len(np.unique(y))>1 else None)

    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    metrics = {"accuracy": float(acc)}
    logger.info("Validation accuracy: %.4f", acc)
    return model, metrics


def load_best_model_metadata():
    if os.path.exists(BEST_MODEL_METAFILE):
        try:
            with open(BEST_MODEL_METAFILE, 'r') as f:
                return json.load(f)
        except Exception:
            logger.exception("Failed to read best model metadata")
    return None


def save_model_with_version(model, metrics):
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    # read existing best
    best = load_best_model_metadata()
    version = 1
    if best and 'version' in best:
        version = best['version'] + 1

    model_name = f"model_v{version}_{ts}.joblib"
    model_path = os.path.join(MODELS_DIR, model_name)

    try:
        joblib.dump(model, model_path)
        logger.info("Saved model to %s", model_path)
    except Exception:
        logger.exception("Failed saving model to %s", model_path)

    # Save metadata
    meta = {
        'version': version,
        'timestamp': ts,
        'model_path': model_path,
        'metrics': metrics
    }
    try:
        with open(os.path.join(MODELS_DIR, f"meta_v{version}_{ts}.json"), 'w') as f:
            json.dump(meta, f, indent=2)
    except Exception:
        logger.exception("Failed writing meta for version %s", version)

    # if this is the best model, update best_model.json
    updated_best = False
    if not best or metrics.get('accuracy', 0.0) >= best.get('metrics', {}).get('accuracy', 0.0):
        try:
            with open(BEST_MODEL_METAFILE, 'w') as f:
                json.dump(meta, f, indent=2)
            updated_best = True
            logger.info("Updated best model metadata (v%d)" % version)
        except Exception:
            logger.exception("Failed updating best model file")

    return model_path, updated_best


# ---------------------------
# Optional: record runs in sqlite using sqlalchemy
# ---------------------------

def init_db():
    if not SQLALCHEMY_AVAILABLE:
        return None
    engine = create_engine(f'sqlite:///{RUNS_DB}')
    metadata = MetaData()
    runs = Table('runs', metadata,
                 Column('id', Integer, primary_key=True, autoincrement=True),
                 Column('timestamp', DateTime),
                 Column('model_path', String),
                 Column('accuracy', Float),
                 Column('notes', String)
                 )
    metadata.create_all(engine)
    return engine, runs


def record_run(engine_runs_tuple, model_path, metrics, notes=''):
    if not engine_runs_tuple:
        return
    engine, runs = engine_runs_tuple
    ins = runs.insert().values(timestamp=datetime.utcnow(), model_path=model_path, accuracy=metrics.get('accuracy', None), notes=notes)
    conn = engine.connect()
    conn.execute(ins)
    conn.close()
    logger.info("Recorded run into sqlite")


# ---------------------------
# Orchestration: a single pipeline run
# ---------------------------

def pipeline_run(prompt_notes=None):
    """One full pipeline execution."""
    start = datetime.utcnow()
    logger.info("Pipeline run started")

    df, files = read_csvs_from_incoming()
    if df.empty:
        logger.info("No data to process — exiting run")
        return {'status': 'no_data', 'timestamp': start.isoformat()}

    X, y = basic_preprocess(df)
    model, metrics = train_and_validate(X, y)

    model_path = None
    updated_best = False
    if model is not None:
        model_path, updated_best = save_model_with_version(model, metrics)

    # archive processed files
    if files:
        archive_files(files)

    # copy artifacts
    try:
        artifact_name = f"artifacts_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.zip"
        artifact_path = os.path.join(ARTIFACTS_DIR, artifact_name)
        shutil.make_archive(artifact_path.replace('.zip',''), 'zip', MODELS_DIR)
        logger.info("Created artifact bundle %s", artifact_path)
    except Exception:
        logger.exception("Failed creating artifact bundle")

    # Optionally record run in DB
    engine_runs = None
    if SQLALCHEMY_AVAILABLE:
        engine_runs = init_db()
    record_run(engine_runs, model_path, metrics, notes=prompt_notes or '')

    end = datetime.utcnow()
    elapsed = (end - start).total_seconds()
    result = {
        'status': 'success',
        'model_path': model_path,
        'metrics': metrics,
        'updated_best': updated_best,
        'duration_seconds': elapsed,
        'timestamp': end.isoformat()
    }
    logger.info("Pipeline run finished in %.1f sec", elapsed)
    return result


# ---------------------------
# Scheduler wrapper
# ---------------------------

def schedule_runner(interval_cron: str = None):
    """Simple scheduler using schedule library.

    By default it runs once per day at 00:00. The user may run with a cron-like spec
    but this function only accepts a few simple human schedules.
    """
    if schedule is None:
        raise RuntimeError("schedule library not installed. pip install schedule to use scheduling")

    # default: daily at midnight
    def job():
        logger.info("Scheduled job triggered")
        pipeline_run(prompt_notes='scheduled run')

    # schedule every day at 00:00
    schedule.every().day.at("00:00").do(job)
    logger.info("Scheduler set to run every day at 00:00 UTC (change in code if you need different schedule)")

    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")


# ---------------------------
# CLI
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description='Automated Data Pipeline')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--run_once', action='store_true', help='Run the pipeline once and exit')
    group.add_argument('--schedule', action='store_true', help='Run pipeline on schedule (requires schedule lib)')
    args = parser.parse_args()

    if args.run_once:
        res = pipeline_run(prompt_notes='manual run')
        print(json.dumps(res, indent=2))
    elif args.schedule:
        if schedule is None:
            logger.error("schedule library missing. install with: pip install schedule")
            return
        schedule_runner()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
