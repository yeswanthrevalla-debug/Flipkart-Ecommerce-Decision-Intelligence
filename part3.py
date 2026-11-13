"""Part 3: Segmentation and dimensionality reduction from scratch.
Run from the repository root: python part3.py
Only Python's standard library and NumPy are used.
"""
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "records.csv"
OUTPUT_DIR = ROOT / "outputs"
RESULTS_PATH = OUTPUT_DIR / "part3_results.txt"
SEGMENTS_PATH = OUTPUT_DIR / "part3_segments.csv"
HIERARCHICAL_PATH = OUTPUT_DIR / "part3_hierarchical.csv"

FEATURE_NAMES = [
    "customer_tenure_months",
    "complaint_count_90d",
    "discount_order_share",
    "avg_order_value_inr_hundreds",
    "loyalty_score",
]


def load_data(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    x = np.array(
        [[float(row[name]) for name in FEATURE_NAMES] for row in rows],
        dtype=float,
    )
    outcome = np.array([int(row["churned"]) for row in rows], dtype=int)
    value = np.array(
        [float(row["order_value_inr_hundreds"]) for row in rows],
        dtype=float,
    )
    return x, outcome, value


def standardise(x):
    mean = np.mean(x, axis=0)
    sd = np.std(x, axis=0, ddof=1)
    return (x - mean) / sd, mean, sd


def run_kmeans(z, k):
    """K-means with the exact starting rows and convergence rule in the brief."""
    n = len(z)
    step = n // k
    starting_indices = [j * step for j in range(k)]
    centroids = z[starting_indices].copy()
    labels = np.full(n, -1, dtype=int)
    assignment_passes = 0

    while True:
        # Squared Euclidean distances; argmin sends ties to lower cluster index.
        distances = np.sum(
            (z[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2,
            axis=2,
        )
        new_labels = np.argmin(distances, axis=1)
        assignment_passes += 1
        unchanged = np.array_equal(new_labels, labels)
        labels = new_labels

        new_centroids = centroids.copy()
        for cluster in range(k):
            members = z[labels == cluster]
            if len(members) > 0:
                new_centroids[cluster] = np.mean(members, axis=0)
        centroids = new_centroids

        if unchanged:
            break

    final_distances = np.sum((z - centroids[labels]) ** 2, axis=1)
    wcss = float(np.sum(final_distances))
    return labels, centroids, wcss, assignment_passes


def cluster_names(labels, x):
    """Map raw K-means labels to A/B/C by ascending mean original X3."""
    raw_clusters = []
    for cluster in range(3):
        members = np.flatnonzero(labels == cluster)
        mean_x3 = float(np.mean(x[members, 2]))
        raw_clusters.append((mean_x3, cluster, members))

    raw_clusters.sort(key=lambda item: item[0])
    ordered = {}
    for name, (_, raw_label, members) in zip(("A", "B", "C"), raw_clusters):
        ordered[name] = members
    return ordered


def pairwise_cluster_linkage(cluster_a, cluster_b, distances, method):
    values = [
        distances[i, j]
        for i in cluster_a
        for j in cluster_b
    ]
    if method == "single":
        return float(min(values))
    if method == "complete":
        return float(max(values))
    if method == "average":
        return float(sum(values) / len(values))
    raise ValueError(f"Unknown linkage method: {method}")


def hierarchical_clustering():
    """Agglomerative clustering on the six-cohort distance table."""
    names = ["E1", "E2", "E3", "E4", "E5", "E6"]
    distances = np.array(
        [
            [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
            [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
            [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
            [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
            [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
            [6.1, 2.0, 1.3, 4.5, 3.4, 0.0],
        ],
        dtype=float,
    )

    all_rows = []
    clusters_at_cut = {}
    for method in ("single", "complete", "average"):
        clusters = [{i} for i in range(6)]
        heights = []

        for merge_no in range(1, 6):
            best = None
            # The brief states that no linkage-distance ties occur.
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    distance = pairwise_cluster_linkage(
                        clusters[i], clusters[j], distances, method
                    )
                    candidate = (distance, i, j)
                    if best is None or candidate[0] < best[0]:
                        best = candidate

            height, i, j = best
            left = clusters[i]
            right = clusters[j]

            # cluster_a is the one with the smaller smallest-member number.
            if min(left) <= min(right):
                cluster_a, cluster_b = left, right
            else:
                cluster_a, cluster_b = right, left

            def display(cluster):
                return "+".join(names[index] for index in sorted(cluster))

            all_rows.append(
                [
                    method,
                    merge_no,
                    display(cluster_a),
                    display(cluster_b),
                    f"{height:.4f}",
                ]
            )
            heights.append(height)

            merged = left | right
            clusters = [
                cluster
                for index, cluster in enumerate(clusters)
                if index not in (i, j)
            ]
            clusters.append(merged)
            clusters.sort(key=lambda cluster: min(cluster))

        gaps = [
            heights[index + 1] - heights[index]
            for index in range(4)
        ]
        largest_gap_index = int(np.argmax(gaps))
        merges_below_cut = largest_gap_index + 1
        clusters_at_cut[method] = 6 - merges_below_cut

    return all_rows, clusters_at_cut


def fmt(value):
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.4f}"


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run 'python generate_data.py' from the project folder first."
        )

    x, outcome, value = load_data(DATA_PATH)
    z, feature_mean, feature_sd = standardise(x)

    results = {}

    # Task 3.1: pinned K-means for K = 2, 3, 4, 5, 6.
    kmeans_results = {}
    for k in range(2, 7):
        labels, centroids, wcss, passes = run_kmeans(z, k)
        kmeans_results[k] = (labels, centroids, wcss, passes)
        results[f"wcss_k{k}"] = wcss

    # Task 3.2: elbow rule; use K=3 for all remaining tasks regardless.
    elbow_k = None
    for k in (3, 4, 5):
        previous_drop = (
            kmeans_results[k - 1][2] - kmeans_results[k][2]
        )
        current_drop = (
            kmeans_results[k][2] - kmeans_results[k + 1][2]
        )
        if current_drop < 0.5 * previous_drop:
            elbow_k = k
            break
    if elbow_k is None:
        # The task defines the first K satisfying the rule; this fallback
        # makes absence explicit rather than silently selecting another K.
        elbow_k = 0
    results["elbow_k"] = elbow_k

    # Task 3.3: profile K=3 clusters, named by ascending original X3 mean.
    labels3 = kmeans_results[3][0]
    segments = cluster_names(labels3, x)
    segment_rows = []
    for name in ("A", "B", "C"):
        indices = segments[name]
        results[f"size_segment_{name}"] = len(indices)
        means = np.mean(x[indices], axis=0)
        outcome_rate = float(np.mean(outcome[indices]))
        mean_value = float(np.mean(value[indices]))
        segment_rows.append(
            [
                name,
                len(indices),
                *[f"{float(v):.4f}" for v in means],
                f"{outcome_rate:.4f}",
                f"{mean_value:.4f}",
            ]
        )

    # Task 3.4: hierarchical clustering on the supplied six-cohort table.
    hierarchical_rows, cut_counts = hierarchical_clustering()
    results["single_clusters_at_cut"] = cut_counts["single"]
    results["complete_clusters_at_cut"] = cut_counts["complete"]
    results["average_clusters_at_cut"] = cut_counts["average"]

    # Task 3.5: PCA, with deterministic eigenvector sign convention.
    centered = z - np.mean(z, axis=0)
    covariance = (centered.T @ centered) / (len(centered) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    for component in range(eigenvectors.shape[1]):
        vector = eigenvectors[:, component]
        largest_index = int(np.argmax(np.abs(vector)))
        if vector[largest_index] < 0:
            eigenvectors[:, component] *= -1.0

    eigenvalue_sum = float(np.sum(eigenvalues))
    explained = eigenvalues / eigenvalue_sum
    cumulative = np.cumsum(explained)
    components_for_80pct = int(np.searchsorted(cumulative, 0.8) + 1)

    results["eigenvalue_sum"] = eigenvalue_sum
    for i, eigenvalue in enumerate(eigenvalues, start=1):
        results[f"eigenvalue_{i}"] = float(eigenvalue)
        results[f"explained_ratio_{i}"] = float(explained[i - 1])
        results[f"cumulative_ratio_{i}"] = float(cumulative[i - 1])
    results["components_for_80pct"] = components_for_80pct

    pc1 = eigenvectors[:, 0]
    pc2 = eigenvectors[:, 1]
    for j, name in enumerate(("X1", "X2", "X3", "X4", "X5")):
        results[f"pc1_loading_{name}"] = float(pc1[j])
        results[f"pc2_loading_{name}"] = float(pc2[j])

    projections = centered @ eigenvectors[:, :2]
    for row_index in range(3):
        row_number = row_index + 1
        results[f"row{row_number}_pc1"] = float(projections[row_index, 0])
        results[f"row{row_number}_pc2"] = float(projections[row_index, 1])

    for name in ("A", "B", "C"):
        results[f"mean_pc1_segment_{name}"] = float(
            np.mean(projections[segments[name], 0])
        )

    OUTPUT_DIR.mkdir(exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8", newline="") as handle:
        for key, value_out in results.items():
            handle.write(f"{key} = {fmt(value_out)}\n")

    write_csv(
        SEGMENTS_PATH,
        [
            "segment", "size", "mean_X1", "mean_X2", "mean_X3",
            "mean_X4", "mean_X5", "outcome_rate", "mean_value",
        ],
        segment_rows,
    )
    write_csv(
        HIERARCHICAL_PATH,
        ["linkage", "merge_no", "cluster_a", "cluster_b", "height"],
        hierarchical_rows,
    )

    print("Part 3 completed.")
    print(f"Rows analysed: {len(x)}")
    print(f"K=3 segment sizes: {len(segments['A'])}, "
          f"{len(segments['B'])}, {len(segments['C'])}")
    print(f"Elbow K: {elbow_k}")
    print(f"Eigenvalue sum: {eigenvalue_sum:.4f}")
    print(f"Results saved to: {RESULTS_PATH}")
    print(f"CSV files saved to: {SEGMENTS_PATH} and {HIERARCHICAL_PATH}")


if __name__ == "__main__":
    main()
