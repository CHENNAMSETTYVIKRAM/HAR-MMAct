"""
extract_genuine_data.py - Selective Extractor for Genuine MMAct Option 2 Archives

Extracts only the 10 verified target classes from downloaded subject archives:
- trimmed_sensor.zip
- trimmed_splits.zip
- trimmed_x-scene_subject16.zip
- trimmed_x-scene_subject8.zip
Preserves directory layout under data/genuine_mmact/
"""

import os
import sys
import zipfile
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import VERIFIED_TARGET_CLASSES

DOWNLOADS_DIR = Path("data/downloads")
TARGET_DIR = Path("data/genuine_mmact")

REQUIRED_ARCHIVES = [
    "trimmed_sensor.zip",
    "trimmed_splits.zip",
    "trimmed_x-scene_subject16.zip",
    "trimmed_x-scene_subject8.zip",
]


def check_downloads():
    print("=" * 70)
    print("Checking Downloaded Archives in:", DOWNLOADS_DIR.resolve())
    print("=" * 70)
    missing = []
    found = []
    for archive in REQUIRED_ARCHIVES:
        p = DOWNLOADS_DIR / archive
        if p.exists():
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  [FOUND] {archive:<30} ({size_mb:.2f} MB)")
            found.append(p)
        else:
            print(f"  [MISSING] {archive:<30}")
            missing.append(archive)

    return found, missing


def is_target_file(filename: str) -> bool:
    """Checks if the file belongs to the 10 verified target classes or is a split/metadata file."""
    if not filename.endswith((".mp4", ".csv", ".txt", ".json")):
        return False

    name_lower = Path(filename).stem.lower().strip()
    # Check if target action name is directly in filename
    for action in VERIFIED_TARGET_CLASSES:
        if action == name_lower or action in name_lower:
            return True

    # Keep split and metadata files
    if "split" in name_lower or "gt" in name_lower:
        return True

    return False


def extract_archive(zip_path: Path):
    print(f"\nProcessing {zip_path.name}...")
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    extracted_count = 0
    skipped_count = 0

    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue

            # For video archives, selectively extract target classes
            if "subject" in zip_path.name:
                if is_target_file(member.filename):
                    zf.extract(member, TARGET_DIR)
                    extracted_count += 1
                else:
                    skipped_count += 1
            else:
                # Sensors and splits: extract all
                zf.extract(member, TARGET_DIR)
                extracted_count += 1

    print(f"  --> Extracted {extracted_count} files (filtered out {skipped_count} non-target clips)")


def main():
    found, missing = check_downloads()
    if missing:
        print("\n[ERROR] Cannot proceed with extraction. Missing required archives:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    print("\nBeginning selective extraction to:", TARGET_DIR.resolve())
    for archive_path in found:
        extract_archive(archive_path)

    print("\nExtraction complete! Directory structure:")
    for root, dirs, files in os.walk(TARGET_DIR):
        depth = len(Path(root).relative_to(TARGET_DIR).parts)
        if depth <= 3:
            indent = "  " * depth
            print(f"{indent}- {Path(root).name}/ ({len(files)} files)")


if __name__ == "__main__":
    main()
