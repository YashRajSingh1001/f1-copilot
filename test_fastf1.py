"""Quick terminal test for FastF1 session loading."""

import traceback
from pathlib import Path

print("Step 1: Importing FastF1...")
import fastf1
print(f"  FastF1 version: {fastf1.__version__}")

print("\nStep 2: Setting up cache...")
cache_path = Path("./f1_cache")
cache_path.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(cache_path))
print(f"  Cache at: {cache_path.resolve()}")

print("\nStep 3: Loading 2024 Bahrain Race session...")
try:
    session = fastf1.get_session(2024, "Bahrain", "R")
    print("  Session object created.")

    print("  Calling session.load() — this downloads data, may take 1-2 min...")
    session.load(laps=True, telemetry=False, weather=True, messages=False)
    print("  session.load() complete.")

    print("\nStep 4: Accessing session data...")
    print(f"  Results rows: {len(session.results)}")
    print(f"  Laps rows: {len(session.laps)}")
    print(f"  Winner: {session.results.iloc[0]['FullName']}")
    print("\nAll good! FastF1 is working correctly.")

except Exception as e:
    print(f"\nERROR: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
