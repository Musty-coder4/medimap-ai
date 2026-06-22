"""
MediMap AI — Project Setup Script
===================================
Scaffolds all required directories and drops placeholder README files.

Run once after cloning:
    python setup_project.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent


def main() -> None:
    """Create directory tree and placeholder files."""
    dirs = [
        "data/raw/tabular",
        "data/raw/images/xray",
        "data/raw/images/skin",
        "data/processed",
        "data/processed/splits",
        "models/saved",
        "models/checkpoints",
        "app",
        "utils",
        "tests",
        "logs",
        "mlruns",
    ]

    for d in dirs:
        target = ROOT / d
        target.mkdir(parents=True, exist_ok=True)
        gitkeep = target / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()
        print(f"  ✔  {d}/")

    # Copy .env.example → .env if not present
    env_example = ROOT / ".env.example"
    env_target = ROOT / ".env"
    if env_example.exists() and not env_target.exists():
        env_target.write_text(env_example.read_text())
        print("  ✔  .env created from .env.example")

    print("\n✅ MediMap AI project scaffolding complete.")
    print("Next steps:")
    print("  1. pip install -r requirements.txt")
    print("  2. Place dataset.csv in data/raw/tabular/")
    print("  3. python models/train.py --epochs 30")
    print("  4. streamlit run app/main.py")


if __name__ == "__main__":
    main()
