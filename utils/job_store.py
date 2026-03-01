"""
job_store.py
Simple file-based job persistence for Render deployment.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import threading

# Thread-safe lock for file operations
_lock = threading.Lock()

# Jobs directory
JOBS_DIR = Path(os.getenv("JOBS_DIR", "job_store"))


def _ensure_dir():
    """Ensure jobs directory exists."""
    JOBS_DIR.mkdir(parents=True, exist_ok=True)


def _job_path(job_id: str) -> Path:
    """Get path for job metadata file."""
    return JOBS_DIR / f"{job_id}.json"


def _zip_path(job_id: str) -> Path:
    """Get path for job ZIP file."""
    return JOBS_DIR / f"{job_id}.zip"


def save_job(job_id: str, job_data: Dict[str, Any]) -> None:
    """Save job to disk."""
    with _lock:
        _ensure_dir()
        
        # Make a copy to avoid modifying original
        data_copy = dict(job_data)
        
        # Extract zip_bytes if present (store separately)
        zip_bytes = data_copy.pop("zip_bytes", None)
        
        # Save metadata
        with open(_job_path(job_id), "w", encoding="utf-8") as f:
            json.dump(data_copy, f)
        
        # Save ZIP if present
        if zip_bytes:
            with open(_zip_path(job_id), "wb") as f:
                f.write(zip_bytes)


def load_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Load job from disk."""
    meta_path = _job_path(job_id)
    
    if not meta_path.exists():
        return None
    
    with _lock:
        with open(meta_path, "r", encoding="utf-8") as f:
            job_data = json.load(f)
        
        # Load ZIP if exists
        zip_file = _zip_path(job_id)
        if zip_file.exists():
            with open(zip_file, "rb") as f:
                job_data["zip_bytes"] = f.read()
        
        return job_data


def update_job(job_id: str, updates: Dict[str, Any]) -> None:
    """Update existing job on disk."""
    job = load_job(job_id) or {}
    job.update(updates)
    save_job(job_id, job)


def delete_old_jobs(max_age_hours: int = 24) -> int:
    """Delete jobs older than max_age_hours. Returns count deleted."""
    _ensure_dir()
    deleted = 0
    cutoff = datetime.utcnow().timestamp() - (max_age_hours * 3600)
    
    for meta_file in JOBS_DIR.glob("*.json"):
        try:
            if meta_file.stat().st_mtime < cutoff:
                job_id = meta_file.stem
                meta_file.unlink(missing_ok=True)
                _zip_path(job_id).unlink(missing_ok=True)
                deleted += 1
        except Exception:
            pass
    
    return deleted
