#!/usr/bin/env python3
"""
Speed KMeans clustering: classify speed into low/mid/high (3 clusters).
Removes zero-speed samples before clustering.
Outputs: cluster centers, boundary thresholds, and cluster labels.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Project paths
PROJECT_ROOT = Path("/root/code/ev_soc_predict")
DATA_PATH = PROJECT_ROOT / "data" / "aligned" / "aligned_data_refined_soc.csv"
OUTPUT_DIR = PROJECT_ROOT / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_speed_data(csv_path=None):
    """Load speed column from CSV, remove zeros, return array and full data."""
    if csv_path is None:
        csv_path = DATA_PATH
    df = pd.read_csv(csv_path)
    speed = df["speed"].values
    # Remove zero speed
    mask = speed > 0
    speed_nonzero = speed[mask]
    print(f"[speed_kmeans] Total samples: {len(speed)}, Non-zero: {len(speed_nonzero)}, "
          f"Removed zeros: {len(speed) - len(speed_nonzero)}")
    return speed_nonzero, df


def kmeans_cluster(data, n_clusters=3, random_state=42):
    """Run KMeans on 1D data, return sorted centers, labels, and model."""
    X = data.reshape(-1, 1)
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = model.fit_predict(X)
    # Sort clusters by center value (low -> mid -> high)
    centers_sorted_idx = np.argsort(model.cluster_centers_.flatten())
    # Remap labels so 0=low, 1=mid, 2=high
    label_mapping = {old: new for new, old in enumerate(centers_sorted_idx)}
    labels_mapped = np.array([label_mapping[l] for l in labels])
    centers = model.cluster_centers_.flatten()[centers_sorted_idx]
    return centers, labels_mapped, model


def compute_thresholds(centers):
    """Compute boundary thresholds between adjacent clusters (midpoints)."""
    thresholds = []
    for i in range(len(centers) - 1):
        t = (centers[i] + centers[i + 1]) / 2.0
        thresholds.append(t)
    return np.array(thresholds)


def get_cluster_stats(speed, labels):
    """Get per-cluster statistics: count, min, max, mean, std."""
    stats = {}
    for cluster_id, name in enumerate(["Low", "Mid", "High"]):
        mask = labels == cluster_id
        cluster_data = speed[mask]
        stats[name] = {
            "count": int(len(cluster_data)),
            "min": float(cluster_data.min()),
            "max": float(cluster_data.max()),
            "mean": float(cluster_data.mean()),
            "std": float(cluster_data.std()),
        }
    return stats


def run(csv_path=None):
    """Run full pipeline: load -> cluster -> compute thresholds -> report."""
    speed, df_full = load_speed_data(csv_path)
    centers, labels, model = kmeans_cluster(speed)
    thresholds = compute_thresholds(centers)
    stats = get_cluster_stats(speed, labels)

    print(f"\n=== Cluster Centers (km/h) ===")
    for name, center in zip(["Low", "Mid", "High"], centers):
        print(f"  {name:6s}: {center:.2f}")

    print(f"\n=== Boundary Thresholds (km/h) ===")
    bound_names = ["Low->Mid", "Mid->High"]
    for name, t in zip(bound_names, thresholds):
        print(f"  {name:10s}: {t:.2f}")

    print(f"\n=== Per-Cluster Stats ===")
    for name, s in stats.items():
        print(f"  {name:6s}: count={s['count']:>6d}, "
              f"range=[{s['min']:6.2f}, {s['max']:6.2f}], "
              f"mean={s['mean']:6.2f}, std={s['std']:5.2f}")

    return {
        "centers": centers,
        "thresholds": thresholds,
        "labels": labels,
        "speed_nonzero": speed,
        "stats": stats,
        "model": model,
    }


def main():
    result = run()
    print("\n[speed_kmeans] Done.")


if __name__ == "__main__":
    main()
