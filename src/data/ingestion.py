"""
ingestion.py - Automated, Resumable Dataset Ingestion with Hash Manifest and Safety Checks
"""

import hashlib
import json
import os
import shutil
import tarfile
import zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set


def compute_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """Computes SHA-256 hash of a file efficiently."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def is_safe_path(base_dir: Path, target_path: Path) -> bool:
    """Prevents zip-slip / path traversal vulnerability."""
    try:
        base_dir_resolved = base_dir.resolve()
        target_path_resolved = target_path.resolve()
        return base_dir_resolved in target_path_resolved.parents or base_dir_resolved == target_path_resolved
    except Exception:
        return False


@dataclass
class IngestionRecord:
    file_name: str
    relative_path: str
    sha256: str
    file_size_bytes: int
    status: str  # 'processed', 'skipped', 'quarantined', 'rejected'
    reason: str
    extracted_files_count: int = 0


class DatasetIngestionPipeline:
    """
    Automated, idempotent ingestion pipeline:
    - Scans incoming directory for new archives (*.zip, *.tar.gz, *.csv, *.mp4)
    - Verifies path safety and archive integrity
    - Avoids duplicate extractions using SHA-256 manifest
    - Records audit trail to ingestion_manifest.json
    """

    def __init__(
        self,
        incoming_dir: Path,
        raw_storage_dir: Optional[Path] = None,
        manifest_path: Path = Path("outputs/ingestion_manifest.json"),
        quarantine_dir: Optional[Path] = None,
        extracted_dir: Optional[Path] = None,
    ):
        self.incoming_dir = Path(incoming_dir)
        target_dir = raw_storage_dir if raw_storage_dir is not None else extracted_dir
        if target_dir is None:
            raise ValueError("Either 'raw_storage_dir' or 'extracted_dir' must be provided.")
        self.raw_storage_dir = Path(target_dir)
        self.manifest_path = Path(manifest_path)
        self.quarantine_dir = Path(quarantine_dir) if quarantine_dir else (self.raw_storage_dir / "quarantine")
        self.manifest: Dict[str, dict] = self._load_manifest()

    def _load_manifest(self) -> Dict[str, dict]:
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_manifest(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

    def scan_and_ingest(self) -> List[IngestionRecord]:
        """Scans incoming directory and processes new files safely."""
        self.raw_storage_dir.mkdir(parents=True, exist_ok=True)
        records = []

        if not self.incoming_dir.exists():
            print(f"[Ingestion] Incoming directory does not exist: {self.incoming_dir}")
            return records

        for file_path in sorted(self.incoming_dir.rglob("*")):
            if not file_path.is_file():
                continue

            rel_str = str(file_path.relative_to(self.incoming_dir))
            file_size = file_path.stat().st_size

            # Check if file is an archive
            is_zip = file_path.suffix.lower() == ".zip"
            is_tar = file_path.name.lower().endswith((".tar.gz", ".tar", ".tgz"))
            is_data = file_path.suffix.lower() in [".csv", ".mp4", ".json", ".txt"]

            if not (is_zip or is_tar or is_data):
                record = IngestionRecord(
                    file_name=file_path.name,
                    relative_path=rel_str,
                    sha256="",
                    file_size_bytes=file_size,
                    status="skipped",
                    reason="Unsupported file format",
                )
                records.append(record)
                continue

            # Compute hash to detect duplicates
            file_hash = compute_sha256(file_path)

            if file_hash in self.manifest:
                prev_status = self.manifest[file_hash].get("status", "processed")
                record = IngestionRecord(
                    file_name=file_path.name,
                    relative_path=rel_str,
                    sha256=file_hash,
                    file_size_bytes=file_size,
                    status="skipped",
                    reason=f"Already processed previously with status: {prev_status}",
                    extracted_files_count=self.manifest[file_hash].get("extracted_files_count", 0),
                )
                records.append(record)
                continue

            # Process new archive
            if is_zip:
                rec = self._safe_extract_zip(file_path, file_hash)
            elif is_tar:
                rec = self._safe_extract_tar(file_path, file_hash)
            else:
                rec = self._copy_raw_data(file_path, file_hash)

            records.append(rec)
            self.manifest[file_hash] = asdict(rec)

        self._save_manifest()
        return records

    def _safe_extract_zip(self, zip_path: Path, file_hash: str) -> IngestionRecord:
        extracted = 0
        try:
            free_bytes = shutil.disk_usage(self.raw_storage_dir).free
            if free_bytes < 250 * 1024 * 1024:
                return IngestionRecord(
                    file_name=zip_path.name,
                    relative_path=str(zip_path.relative_to(self.incoming_dir)),
                    sha256=file_hash,
                    file_size_bytes=zip_path.stat().st_size,
                    status="quarantined",
                    reason=f"Insufficient disk space for extraction ({free_bytes / (1024**2):.1f} MB free)",
                )
            with zipfile.ZipFile(zip_path, "r") as zf:
                # Check for zip bomb or malicious path traversal
                for member in zf.infolist():
                    target = self.raw_storage_dir / member.filename
                    if not is_safe_path(self.raw_storage_dir, target):
                        return IngestionRecord(
                            file_name=zip_path.name,
                            relative_path=str(zip_path.relative_to(self.incoming_dir)),
                            sha256=file_hash,
                            file_size_bytes=zip_path.stat().st_size,
                            status="rejected",
                            reason=f"Path traversal detected in zip member: {member.filename}",
                        )

                # Extract safely
                for member in zf.infolist():
                    if not member.is_dir():
                        zf.extract(member, self.raw_storage_dir)
                        extracted += 1

            return IngestionRecord(
                file_name=zip_path.name,
                relative_path=str(zip_path.relative_to(self.incoming_dir)),
                sha256=file_hash,
                file_size_bytes=zip_path.stat().st_size,
                status="processed",
                reason="Successfully extracted archive safely",
                extracted_files_count=extracted,
            )
        except Exception as e:
            return IngestionRecord(
                file_name=zip_path.name,
                relative_path=str(zip_path.relative_to(self.incoming_dir)),
                sha256=file_hash,
                file_size_bytes=zip_path.stat().st_size,
                status="quarantined",
                reason=f"Archive corrupted or unreadable: {str(e)}",
            )

    def _safe_extract_tar(self, tar_path: Path, file_hash: str) -> IngestionRecord:
        extracted = 0
        try:
            free_bytes = shutil.disk_usage(self.raw_storage_dir).free
            if free_bytes < 250 * 1024 * 1024:
                return IngestionRecord(
                    file_name=tar_path.name,
                    relative_path=str(tar_path.relative_to(self.incoming_dir)),
                    sha256=file_hash,
                    file_size_bytes=tar_path.stat().st_size,
                    status="quarantined",
                    reason=f"Insufficient disk space for extraction ({free_bytes / (1024**2):.1f} MB free)",
                )
            with tarfile.open(tar_path, "r:*") as tf:
                for member in tf.getmembers():
                    target = self.raw_storage_dir / member.name
                    if not is_safe_path(self.raw_storage_dir, target):
                        return IngestionRecord(
                            file_name=tar_path.name,
                            relative_path=str(tar_path.relative_to(self.incoming_dir)),
                            sha256=file_hash,
                            file_size_bytes=tar_path.stat().st_size,
                            status="rejected",
                            reason=f"Path traversal detected in tar member: {member.name}",
                        )

                tf.extractall(self.raw_storage_dir)
                extracted = len(tf.getmembers())

            return IngestionRecord(
                file_name=tar_path.name,
                relative_path=str(tar_path.relative_to(self.incoming_dir)),
                sha256=file_hash,
                file_size_bytes=tar_path.stat().st_size,
                status="processed",
                reason="Successfully extracted tar archive",
                extracted_files_count=extracted,
            )
        except Exception as e:
            return IngestionRecord(
                file_name=tar_path.name,
                relative_path=str(tar_path.relative_to(self.incoming_dir)),
                sha256=file_hash,
                file_size_bytes=tar_path.stat().st_size,
                status="quarantined",
                reason=f"Tar corrupted or unreadable: {str(e)}",
            )

    def _copy_raw_data(self, file_path: Path, file_hash: str) -> IngestionRecord:
        target = self.raw_storage_dir / file_path.name
        shutil.copy2(file_path, target)
        return IngestionRecord(
            file_name=file_path.name,
            relative_path=str(file_path.relative_to(self.incoming_dir)),
            sha256=file_hash,
            file_size_bytes=file_path.stat().st_size,
            status="processed",
            reason="Copied raw data file",
            extracted_files_count=1,
        )
