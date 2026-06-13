"""
CLI script to ingest race sessions into the vector store.

Usage:
  python scripts/ingest_race.py --year 2024 --gp Bahrain --session R
  python scripts/ingest_race.py --year 2024 --gp Monaco --session Q
  python scripts/ingest_race.py --bulk  # ingest a default set of memorable races
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from dotenv import load_dotenv
load_dotenv()

from src.data.ingestion import ingest_race_session

BULK_RACES = [
    (2024, "Bahrain", "R"),
    (2024, "Monaco", "R"),
    (2024, "Great Britain", "R"),
    (2024, "Singapore", "R"),
    (2023, "Monaco", "R"),
    (2023, "Bahrain", "R"),
    (2023, "Singapore", "R"),
    (2023, "Abu Dhabi", "R"),
]


def main():
    parser = argparse.ArgumentParser(description="Ingest F1 race data into ChromaDB")
    parser.add_argument("--year", type=int)
    parser.add_argument("--gp", type=str)
    parser.add_argument("--session", type=str, default="R")
    parser.add_argument("--bulk", action="store_true", help="Ingest a curated set of races")
    args = parser.parse_args()

    if args.bulk:
        print(f"Bulk ingesting {len(BULK_RACES)} races...")
        for year, gp, sess in BULK_RACES:
            try:
                ingest_race_session(year, gp, sess)
            except Exception as e:
                print(f"  Failed {year} {gp} {sess}: {e}")
    elif args.year and args.gp:
        ingest_race_session(args.year, args.gp, args.session)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
