"""Compare HDBSCAN configurations without recomputing text embeddings."""

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import hdbscan
import numpy as np
import umap
from hdbscan.validity import validity_index
from sklearn.metrics import adjusted_rand_score, silhouette_score


ROOT = Path(__file__).resolve().parents[1]


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mode", choices=["tfidf", "openai"], default="openai")
    parser.add_argument(
        "--cluster_dimensions",
        default=None,
        help="Comma-separated UMAP dimensions (TF-IDF defaults to 2,5,10,15,20)",
    )
    parser.add_argument("--min_cluster_sizes", default="6,8,10,12,15")
    parser.add_argument("--min_samples", default=None)
    parser.add_argument(
        "--min_clusters",
        type=int,
        default=None,
        help="Required cluster-count floor (TF-IDF defaults to 5; OpenAI has no floor)",
    )
    parser.add_argument("--seeds", default="41,42,43")
    parser.add_argument(
        "--selection_seed",
        type=int,
        default=42,
        help="Seed used for the published map and minimum-cluster constraint",
    )
    parser.add_argument("--resamples", type=int, default=3)
    parser.add_argument("--sample_fraction", type=float, default=0.9)
    parser.add_argument("--resample_seed", type=int, default=2026)
    args = parser.parse_args()

    processed = ROOT / "data" / "datasets" / args.dataset / "processed"
    cache_path = processed / f"comments_embeddings_{args.mode}.npz"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing {cache_path}. Run scripts/10_embed_umap_cluster.py first."
        )

    with np.load(cache_path, allow_pickle=False) as cached:
        embeddings = cached["embeddings"].astype(np.float32)

    min_clusters = args.min_clusters if args.min_clusters is not None else (5 if args.mode == "tfidf" else 0)
    dimensions = parse_ints(
        args.cluster_dimensions
        or ("2,5,10,15,20" if args.mode == "tfidf" else "10,15,20")
    )
    cluster_sizes = parse_ints(args.min_cluster_sizes)
    min_samples_values = parse_ints(
        args.min_samples or ("3,5,6,8" if args.mode == "tfidf" else "3,5,8")
    )
    seeds = parse_ints(args.seeds)
    if args.selection_seed not in seeds:
        raise ValueError("--selection_seed must be included in --seeds")
    runs = []
    labels_by_config = defaultdict(list)
    baseline_labels = {}
    subsample_aris = defaultdict(list)

    for dimensions_value in dimensions:
        components = max(2, min(dimensions_value, len(embeddings) - 2))
        for seed in seeds:
            semantic = umap.UMAP(
                n_neighbors=15,
                min_dist=0.05 if components == 2 else 0.0,
                n_components=components,
                metric="cosine",
                random_state=seed,
            ).fit_transform(embeddings)
            for min_cluster_size in cluster_sizes:
                for min_samples in min_samples_values:
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                        metric="euclidean",
                    )
                    labels = clusterer.fit_predict(semantic)
                    cluster_ids = sorted(set(labels) - {-1})
                    mask = labels >= 0
                    silhouette = None
                    if len(cluster_ids) >= 2 and mask.sum() > len(cluster_ids):
                        silhouette = float(silhouette_score(semantic[mask], labels[mask]))
                    dbcv = None
                    if len(cluster_ids) >= 2:
                        try:
                            dbcv = float(
                                validity_index(
                                    semantic.astype(np.float64), labels.astype(int)
                                )
                            )
                        except ValueError:
                            pass

                    key = (components, min_cluster_size, min_samples)
                    labels_by_config[key].append(labels)
                    if seed == seeds[0]:
                        baseline_labels[key] = labels
                    runs.append(
                        {
                            "dimensions": components,
                            "min_cluster_size": min_cluster_size,
                            "min_samples": min_samples,
                            "seed": seed,
                            "n_clusters": len(cluster_ids),
                            "noise_fraction": float(np.mean(labels == -1)),
                            "silhouette_clustered": silhouette,
                            "dbcv": dbcv,
                            "mean_cluster_persistence": (
                                float(np.mean(clusterer.cluster_persistence_))
                                if len(clusterer.cluster_persistence_)
                                else None
                            ),
                        }
                    )

        # Test sensitivity to removing a small fraction of comments. The semantic
        # projection is refit for every subsample, so this includes UMAP as well
        # as HDBSCAN instability.
        rng = np.random.default_rng(args.resample_seed + components)
        sample_size = max(3, int(round(len(embeddings) * args.sample_fraction)))
        for resample in range(args.resamples):
            sample_indices = np.sort(
                rng.choice(len(embeddings), size=sample_size, replace=False)
            )
            sample_semantic = umap.UMAP(
                n_neighbors=min(15, sample_size - 1),
                min_dist=0.05 if components == 2 else 0.0,
                n_components=min(components, sample_size - 2),
                metric="cosine",
                random_state=seeds[0] + resample,
            ).fit_transform(embeddings[sample_indices])
            for min_cluster_size in cluster_sizes:
                for min_samples in min_samples_values:
                    key = (components, min_cluster_size, min_samples)
                    sample_labels = hdbscan.HDBSCAN(
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                        metric="euclidean",
                    ).fit_predict(sample_semantic)
                    subsample_aris[key].append(
                        adjusted_rand_score(
                            baseline_labels[key][sample_indices], sample_labels
                        )
                    )

    summaries = []
    for key, label_runs in labels_by_config.items():
        matching = [
            run
            for run in runs
            if (run["dimensions"], run["min_cluster_size"], run["min_samples"])
            == key
        ]
        aris = [adjusted_rand_score(a, b) for a, b in combinations(label_runs, 2)]

        def mean_present(field):
            values = [run[field] for run in matching if run[field] is not None]
            return float(np.mean(values)) if values else None

        cluster_counts = [r["n_clusters"] for r in matching]
        selected_run = next(r for r in matching if r["seed"] == args.selection_seed)
        summaries.append(
            {
                "dimensions": key[0],
                "min_cluster_size": key[1],
                "min_samples": key[2],
                "mean_n_clusters": float(np.mean(cluster_counts)),
                "min_n_clusters": int(min(cluster_counts)),
                "selection_seed": args.selection_seed,
                "selection_n_clusters": selected_run["n_clusters"],
                "meets_min_clusters": selected_run["n_clusters"] >= min_clusters,
                "mean_noise_fraction": mean_present("noise_fraction"),
                "mean_silhouette_clustered": mean_present("silhouette_clustered"),
                "mean_dbcv": mean_present("dbcv"),
                "mean_cluster_persistence": mean_present("mean_cluster_persistence"),
                "seed_stability_ari": float(np.mean(aris)) if aris else None,
                "subsample_stability_ari": (
                    float(np.mean(subsample_aris[key])) if subsample_aris[key] else None
                ),
            }
        )

    summaries.sort(
        key=lambda item: (
            item["meets_min_clusters"],
            item["seed_stability_ari"] or -1,
            item["subsample_stability_ari"] or -1,
            item["mean_dbcv"] or -1,
            item["mean_silhouette_clustered"] or -1,
        ),
        reverse=True,
    )
    output = processed / f"cluster_candidates_{args.mode}.json"
    eligible = [item for item in summaries if item["meets_min_clusters"]]
    output.write_text(
        json.dumps(
            {
                "dataset_id": args.dataset,
                "map_type": args.mode,
                "min_clusters": min_clusters,
                "selection_seed": args.selection_seed,
                "recommended": eligible[0] if eligible else None,
                "candidates": summaries,
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    print(f"✅ Candidate report written to {output} ({len(summaries)} configurations)")
    if min_clusters and not eligible:
        raise RuntimeError(
            f"No {args.mode} candidate produced at least {min_clusters} clusters "
            f"at selection seed {args.selection_seed}"
        )
    if eligible:
        choice = eligible[0]
        print(
            "✅ Recommended eligible candidate: "
            f"dimensions={choice['dimensions']}, "
            f"min_cluster_size={choice['min_cluster_size']}, "
            f"min_samples={choice['min_samples']}"
        )


if __name__ == "__main__":
    main()
