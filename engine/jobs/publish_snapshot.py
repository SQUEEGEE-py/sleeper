"""Copy what the player model reads into ``snapshot/``, which is committed, so the nightly job
can run in GitHub Actions.

    uv run python -m engine.jobs.ingest_history     # LOCAL: stats.nba.com blocks cloud IPs
    uv run python -m engine.jobs.build_board
    uv run python -m engine.jobs.publish_snapshot   # then commit and push snapshot/

``data/`` stays gitignored; ``snapshot/`` is the deliberate exception, about 1.2 MB: the valued
pool and the game logs ``PlayerModel`` is built from. It is a copy as of the day it was
published, so the cloud model learns this season's roles only when it is refreshed (weekly is
plenty early in the season) -- ``snapshot/manifest.json`` records when, and the nightly report
says how old it is.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from engine.ingest.nba_stats import DATA_DIR, previous_seasons
from engine.jobs.build_board import HISTORY_SEASONS
from engine.jobs.runtime import utf8_output

CONFIG_PATH = Path("config/league.json")
POOL_PATH = Path("data/board/pool.csv")
SNAPSHOT_DIR = Path("snapshot")
MANIFEST = "manifest.json"


def snapshot_files(season: str, data_dir: Path = DATA_DIR) -> list[tuple[Path, bool]]:
    """(source, required) for everything the model reads. The current season's logs and the
    preseason logs are optional, as ``load_logs`` and ``nightly`` already treat them."""
    files = [(POOL_PATH, True)]
    files += [(data_dir / f"game_logs_{past}.parquet", True) for past in previous_seasons(season, HISTORY_SEASONS)]
    files += [(data_dir / f"game_logs_{season}.parquet", False), (data_dir / f"preseason_logs_{season}.parquet", False)]
    return files


def main() -> int:
    utf8_output()
    season = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["season"]
    files = snapshot_files(season)
    missing = [str(path) for path, required in files if required and not path.exists()]
    if missing:
        raise SystemExit(f"missing {', '.join(missing)}. Run ingest_history then build_board first.")

    SNAPSHOT_DIR.mkdir(exist_ok=True)
    copied = {}
    for path, _ in files:
        if path.exists():
            shutil.copy2(path, SNAPSHOT_DIR / path.name)
            copied[path.name] = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
    manifest = {"published": datetime.now(timezone.utc).isoformat(timespec="seconds"), "season": season,
                "files": copied}
    (SNAPSHOT_DIR / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    total = sum((SNAPSHOT_DIR / name).stat().st_size for name in copied)
    print(f"snapshot/: {len(copied)} files, {total / 1e6:.2f} MB. Now commit and push snapshot/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
