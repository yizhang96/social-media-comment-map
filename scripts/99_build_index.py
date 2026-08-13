import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "data" / "datasets"
WEB_DATASETS = ROOT / "web" / "public" / "datasets"


def main():
    WEB_DATASETS.mkdir(parents=True, exist_ok=True)
    index = []

    for ds in DATASETS.iterdir():
        if not ds.is_dir():
            continue

        meta = ds / "processed" / "dataset.json"
        processed = ds / "processed"
        map_files = sorted(processed.glob("comments_map_*.json"))

        # Require meta + at least one map
        if not meta.exists() or not map_files:
            continue

        out_dir = WEB_DATASETS / ds.name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Copy meta
        shutil.copy2(meta, out_dir / "dataset.json")

        # Copy all maps
        for mf in map_files:
            shutil.copy2(mf, out_dir / mf.name)

        # Copy optional generated cluster descriptions used by the UI.
        for artifact in processed.glob("cluster_metadata_*.json"):
            shutil.copy2(artifact, out_dir / artifact.name)

        # Add to index
        with open(meta, "r", encoding="utf-8") as f:
            index.append(json.load(f))

    (WEB_DATASETS / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"✅ Dataset index written ({len(index)} datasets)")


if __name__ == "__main__":
    main()
