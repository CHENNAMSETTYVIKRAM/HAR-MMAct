"""
generate_synthetic_data.py - CLI tool to generate synthetic smoke-test fixture
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ProjectConfig
from src.data.synthetic import generate_synthetic_fixture


def main():
    cfg = ProjectConfig()
    print("=" * 70)
    print("Generating Synthetic MMAct Smoke-Test Fixture...")
    print(f"Target Directory: {cfg.synthetic_root}")
    print("=" * 70)

    fixture_path = generate_synthetic_fixture(
        output_root=cfg.synthetic_root,
        num_subjects=3,
        num_cams=2,
        include_test_anomalies=True,
    )

    print(f"[SUCCESS] Synthetic smoke-test fixture created at: {fixture_path}")
    print("Metadata: SYNTHETIC_METADATA.json written.")


if __name__ == "__main__":
    main()
