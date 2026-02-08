from dotenv import load_dotenv
load_dotenv()
import os
import json
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
import umap
import hdbscan



ROOT = Path(__file__).resolve().parents[1]


def embed_tfidf(texts: list[str]) -> np.ndarray:
    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        min_df=2
    )
    X = vectorizer.fit_transform(texts)
    return X.toarray().astype(np.float32)


def embed_openai(texts, model):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if not client.api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    def sanitize_one(x):
        # Force string, remove weird nulls, normalize whitespace
        if x is None:
            s = ""
        else:
            s = str(x)
        s = s.replace("\x00", " ").strip()
        # Avoid empty inputs
        if not s:
            s = "(empty)"
        # Conservative truncation to avoid request limits
        # (embeddings are fine with much longer, but this prevents rare outliers)
        if len(s) > 6000:
            s = s[:6000]
        return s

    embeddings = []
    batch_size = 64

    clean_texts = [sanitize_one(t) for t in texts]

    for i in tqdm(range(0, len(clean_texts), batch_size), desc="Embedding"):
        batch = clean_texts[i:i+batch_size]
        try:
            resp = client.embeddings.create(model=model, input=batch)
        except Exception as e:
            # Print minimal debug info to find the offending sample
            print(f"\n❌ Embedding request failed at batch starting index {i}")
            # show per-item lengths and first 30 chars only
            for j, s in enumerate(batch):
                preview = s[:30].replace("\n", " ")
                print(f"  idx={i+j} len={len(s)} preview={preview!r}")
            raise
        embeddings.extend([d.embedding for d in resp.data])

    return np.array(embeddings, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mode", choices=["tfidf", "openai"], default="tfidf")
    parser.add_argument("--openai_model", default=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"))
    parser.add_argument("--min_cluster_size", type=int, default=6)
    args = parser.parse_args()

    ds_root = ROOT / "data" / "datasets" / args.dataset
    in_xlsx = ds_root / "processed" / "comments_cleaned.xlsx"
    in_csv = ds_root / "processed" / "comments_cleaned.csv"
    suffix = "openai" if args.mode == "openai" else "tfidf"
    out_json = ds_root / "processed" / f"comments_map_{suffix}.json"

    if in_csv.exists():
        df = pd.read_csv(in_csv).copy()
    elif in_xlsx.exists():
        df = pd.read_excel(in_xlsx).copy()
    else:
        raise FileNotFoundError(
            f"Missing cleaned file: {in_csv} or {in_xlsx}"
        )

    # Stable id field
    if "seq" in df.columns and df["seq"].notna().any():
        # pandas new versions: use ffill()
        df["id"] = df["seq"].ffill().astype(int)
    else:
        df["id"] = np.arange(1, len(df) + 1)

    # Texts for embedding
    if "comment_text" not in df.columns:
        raise KeyError("comments_cleaned.* must contain a 'comment_text' column")
    texts = df["comment_text"].fillna("").astype(str).tolist()

    # Embedding
    if args.mode == "openai":
        X = embed_openai(texts, args.openai_model)
    else:
        X = embed_tfidf(texts)

    # UMAP to 2D
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.05,
        n_components=2,
        metric="cosine",
        random_state=42
    )
    xy = reducer.fit_transform(X)
    df["x"] = xy[:, 0]
    df["y"] = xy[:, 1]

    # Cluster in 2D
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        metric="euclidean"
    )
    labels = clusterer.fit_predict(xy)
    df["cluster_id"] = labels

    # Build strict JSON records (no NaN)
    records = []
    for _, r in df.iterrows():
        text_val = r.get("comment_text")
        text = "" if pd.isna(text_val) else str(text_val)

        x = float(r.get("x", 0.0))
        y = float(r.get("y", 0.0))
        if not math.isfinite(x):
            x = 0.0
        if not math.isfinite(y):
            y = 0.0

        likes_val = r.get("likes")
        likes = None if pd.isna(likes_val) else int(likes_val)

        time_val = r.get("time_raw")
        time = None if pd.isna(time_val) else str(time_val)

        loc_val = r.get("location_raw")
        location = None if pd.isna(loc_val) else str(loc_val)

        user_val = r.get("user")
        user = None if pd.isna(user_val) else str(user_val)

        cluster_val = r.get("cluster_id")
        cluster_id = -1 if pd.isna(cluster_val) else int(cluster_val)

        records.append({
            "id": int(r["id"]),
            "text": text,
            "likes": likes,
            "time": time,
            "location": location,
            "user": user,
            "x": x,
            "y": y,
            "cluster_id": cluster_id,
        })

    out_json.write_text(
        json.dumps(records, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8"
    )

    n_noise = sum(1 for rec in records if rec["cluster_id"] == -1)
    n_clusters = len(set(rec["cluster_id"] for rec in records if rec["cluster_id"] >= 0))
    print(f"✅ Map written to {out_json} ({len(records)} points; clusters={n_clusters}; noise={n_noise})")


if __name__ == "__main__":
    main()
