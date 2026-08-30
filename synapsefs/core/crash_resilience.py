"""
Write-Ahead Logging (WAL) and Crash-Recovery Subsystem.
Ensures repository remains consistent and uncorrupted under unexpected SIGKILL / power crashes.
"""

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from synapsefs.core.cas import ContentAddressableStore


class TransactionJournal:
    """
    Manages write-ahead logging (WAL) for atomic multi-stage operations (e.g. commits).
    """

    def __init__(self, synapse_dir: Path):
        self.synapse_dir = Path(synapse_dir)
        self.wal_dir = self.synapse_dir / "wal"
        self.tmp_dir = self.synapse_dir / "tmp"
        self.wal_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def begin_transaction(self, tx_type: str, details: Optional[Dict[str, Any]] = None) -> str:
        """Starts a new transaction log."""
        tx_id = f"tx_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        tx_file = self.wal_dir / f"{tx_id}.json"
        
        data = {
            "tx_id": tx_id,
            "type": tx_type,
            "status": "PREPARING",
            "created_at": time.time(),
            "details": details or {},
            "created_blobs": [],
        }
        
        with open(tx_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
            
        return tx_id

    def record_blob(self, tx_id: str, blob_hash: str) -> None:
        """Records a newly written blob hash as part of the transaction."""
        tx_file = self.wal_dir / f"{tx_id}.json"
        if not tx_file.exists():
            return
        try:
            with open(tx_file, "r+", encoding="utf-8") as f:
                data = json.load(f)
                data["created_blobs"].append(blob_hash)
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            pass

    def mark_committed(self, tx_id: str, final_commit_id: str) -> None:
        """Marks the transaction as fully committed."""
        tx_file = self.wal_dir / f"{tx_id}.json"
        if not tx_file.exists():
            return
        with open(tx_file, "r+", encoding="utf-8") as f:
            data = json.load(f)
            data["status"] = "COMMITTED"
            data["final_commit_id"] = final_commit_id
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()
            f.flush()
            os.fsync(f.fileno())

    def close_transaction(self, tx_id: str) -> None:
        """Removes the WAL file upon successful transaction completion."""
        tx_file = self.wal_dir / f"{tx_id}.json"
        if tx_file.exists():
            try:
                tx_file.unlink()
            except OSError:
                pass

    def recover(self, cas: ContentAddressableStore) -> Dict[str, Any]:
        """
        Scans for interrupted transactions and temporary files and restores consistency.
        """
        recovery_report = {
            "cleaned_temp_files": 0,
            "aborted_transactions": 0,
            "completed_transactions": 0,
        }

        # 1. Clean stale temporary files in .synapsefs/tmp
        if self.tmp_dir.exists():
            for tmp_file in self.tmp_dir.iterdir():
                if tmp_file.is_file():
                    try:
                        tmp_file.unlink()
                        recovery_report["cleaned_temp_files"] += 1
                    except OSError:
                        pass

        # 2. Reconcile WAL logs
        if self.wal_dir.exists():
            for wal_file in self.wal_dir.glob("*.json"):
                try:
                    with open(wal_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    status = data.get("status")
                    if status != "COMMITTED":
                        # Incomplete transaction: clean up any orphaned blobs if desired
                        recovery_report["aborted_transactions"] += 1
                        wal_file.unlink()
                    else:
                        # Transaction was committed, safe to remove WAL
                        recovery_report["completed_transactions"] += 1
                        wal_file.unlink()
                except Exception:
                    try:
                        wal_file.unlink()
                    except OSError:
                        pass

        return recovery_report
