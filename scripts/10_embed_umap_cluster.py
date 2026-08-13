from dotenv import load_dotenv
load_dotenv()
import os
import json
import math
import hashlib
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
import umap
import hdbscan
from sklearn.metrics import silhouette_score



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
    parser.add_argument(
        "--min_samples",
        type=int,
        default=None,
        help="HDBSCAN min_samples (defaults to min_cluster_size)",
    )
    parser.add_argument(
        "--cluster_dimensions",
        type=int,
        default=None,
        help="UMAP dimensions used for clustering (TF-IDF defaults to 2; OpenAI to 15)",
    )
    parser.add_argument(
        "--min_clusters",
        type=int,
        default=None,
        help="Required cluster-count floor (TF-IDF defaults to 5; OpenAI has no floor)",
    )
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument(
        "--refresh_embeddings",
        action="store_true",
        help="Ignore a compatible embedding cache and recompute embeddings",
    )
    args = parser.parse_args()

    ds_root = ROOT / "data" / "datasets" / args.dataset
    in_xlsx = ds_root / "processed" / "comments_cleaned.xlsx"
    in_csv = ds_root / "processed" / "comments_cleaned.csv"
    suffix = "openai" if args.mode == "openai" else "tfidf"
    out_json = ds_root / "processed" / f"comments_map_{suffix}.json"
    embedding_cache = ds_root / "processed" / f"comments_embeddings_{suffix}.npz"
    diagnostics_json = ds_root / "processed" / f"cluster_diagnostics_{suffix}.json"

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
        candidate_ids = df["seq"].ffill().astype(int)
        if candidate_ids.is_unique:
            df["id"] = candidate_ids
        else:
            print("Warning: seq contains duplicate ids; using stable row ids instead")
            df["id"] = np.arange(1, len(df) + 1)
    else:
        df["id"] = np.arange(1, len(df) + 1)

    # Texts for embedding
    if "comment_text" not in df.columns:
        raise KeyError("comments_cleaned.* must contain a 'comment_text' column")
    texts = df["comment_text"].fillna("").astype(str).tolist()
    text_hash = hashlib.sha256(
        json.dumps(texts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    # Embeddings are cached so cluster candidates and labels can be evaluated
    # without repeating paid embedding calls.
    cached = None
    if embedding_cache.exists() and not args.refresh_embeddings:
        cached = np.load(embedding_cache, allow_pickle=False)
        cached_ids = cached["ids"].astype(int)
        cached_hash = str(cached["text_hash"].item()) if "text_hash" in cached else ""
        current_ids = df["id"].to_numpy(dtype=int)
        if np.array_equal(cached_ids, current_ids) and cached_hash == text_hash:
            X = cached["embeddings"].astype(np.float32)
            print(f"Using cached embeddings: {embedding_cache}")
        else:
            cached.close()
            cached = None

    if cached is None:
        if args.mode == "openai":
            X = embed_openai(texts, args.openai_model)
        else:
            X = embed_tfidf(texts)
        np.savez_compressed(
            embedding_cache,
            embeddings=X.astype(np.float32),
            ids=df["id"].to_numpy(dtype=int),
            text_hash=np.array(text_hash),
        )
        print(f"Embedding cache written to {embedding_cache}")
    else:
        cached.close()

    # The TF-IDF two-dimensional option clusters the visible lexical
    # neighborhoods directly. Higher-dimensional and OpenAI configurations use
    # a separate semantic projection for clustering.
    requested_dimensions = args.cluster_dimensions or (2 if args.mode == "tfidf" else 15)
    max_components = max(2, min(requested_dimensions, len(texts) - 2))
    display_reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.05,
        n_components=2,
        metric="cosine",
        random_state=args.random_state,
    )
    xy = display_reducer.fit_transform(X)
    df["x"] = xy[:, 0]
    df["y"] = xy[:, 1]

    if args.mode == "tfidf" and max_components == 2:
        cluster_space = xy
        clustering_space = "umap_display"
    else:
        cluster_reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.0,
            n_components=max_components,
            metric="cosine",
            random_state=args.random_state,
        )
        cluster_space = cluster_reducer.fit_transform(X)
        clustering_space = "umap_semantic"

    # Cluster in the selected projection.
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(cluster_space)
    df["cluster_id"] = labels

    n_clusters = len(set(labels) - {-1})
    min_clusters = args.min_clusters if args.min_clusters is not None else (5 if args.mode == "tfidf" else 0)
    if n_clusters < min_clusters:
        raise RuntimeError(
            f"{args.mode} clustering produced {n_clusters} clusters, below the "
            f"required minimum of {min_clusters}. Choose an eligible candidate "
            "from scripts/11_evaluate_cluster_candidates.py."
        )

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
    clustered_mask = labels >= 0
    silhouette = None
    if n_clusters >= 2 and clustered_mask.sum() > n_clusters:
        silhouette = float(
            silhouette_score(cluster_space[clustered_mask], labels[clustered_mask])
        )
    diagnostics = {
        "dataset_id": args.dataset,
        "map_type": suffix,
        "embedding_model": args.openai_model if args.mode == "openai" else "tfidf",
        "clustering": {
            "algorithm": "hdbscan",
            "space": clustering_space,
            "dimensions": max_components,
            "min_cluster_size": args.min_cluster_size,
            "min_samples": args.min_samples,
            "min_clusters": min_clusters,
            "random_state": args.random_state,
        },
        "n_points": len(records),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_fraction": n_noise / len(records) if records else 0.0,
        "silhouette_clustered": silhouette,
        "mean_cluster_persistence": (
            float(np.mean(clusterer.cluster_persistence_))
            if len(clusterer.cluster_persistence_)
            else None
        ),
        "cluster_sizes": {
            str(cluster_id): int(np.sum(labels == cluster_id))
            for cluster_id in sorted(set(labels))
            if cluster_id >= 0
        },
    }
    diagnostics_json.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"✅ Map written to {out_json} ({len(records)} points; clusters={n_clusters}; noise={n_noise})")
    print(f"✅ Diagnostics written to {diagnostics_json}")


if __name__ == "__main__":
    main()
