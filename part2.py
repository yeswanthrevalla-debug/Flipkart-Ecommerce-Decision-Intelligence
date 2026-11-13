"""Part 2: Classification and evaluation from scratch.
Run from the repository root: python part2.py
Only Python's standard library and NumPy are used.
"""
import csv
import math
from pathlib import Path

import numpy as np

from common import stable_argsort

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "records.csv"
OUTPUT_DIR = ROOT / "outputs"
RESULTS_PATH = OUTPUT_DIR / "part2_results.txt"
THRESHOLDS_PATH = OUTPUT_DIR / "part2_thresholds.csv"
KNN_CV_PATH = OUTPUT_DIR / "part2_knn_cv.csv"


def load_dataset(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    feature_names = [
        "customer_tenure_months",
        "complaint_count_90d",
        "discount_order_share",
        "avg_order_value_inr_hundreds",
        "loyalty_score",
        "cod_flag",
        "plus_member_flag",
    ]
    x = np.array(
        [[float(row[name]) for name in feature_names] for row in rows],
        dtype=float,
    )
    y = np.array([int(row["churned"]) for row in rows], dtype=int)
    return x, y


def stratified_split(x, y):
    """Use the first floor(70%) rows of each class for training, preserving order."""
    train_indices = []
    test_indices = []
    for label in (0, 1):
        indices = np.flatnonzero(y == label)
        cut = (7 * len(indices)) // 10
        train_indices.extend(indices[:cut].tolist())
        test_indices.extend(indices[cut:].tolist())

    # The task says both sets keep file order.
    train_indices.sort()
    test_indices.sort()
    train_indices = np.asarray(train_indices, dtype=int)
    test_indices = np.asarray(test_indices, dtype=int)
    return x[train_indices], y[train_indices], x[test_indices], y[test_indices]


def standardise(train_x, test_x):
    mean = np.mean(train_x, axis=0)
    sd = np.std(train_x, axis=0, ddof=1)
    train_z = (train_x - mean) / sd
    test_z = (test_x - mean) / sd
    return train_z, test_z, mean, sd


def sigmoid(z):
    # Clipping prevents numerical overflow while leaving ordinary values unchanged.
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y, probabilities):
    p = np.clip(probabilities, 1e-15, 1.0 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1.0 - p)))


def fit_logistic_gradient_descent(x, y, learning_rate, updates, capture_epochs=()):
    n, p = x.shape
    weights = np.zeros(p, dtype=float)
    bias = 0.0

    def current_loss():
        return binary_cross_entropy(y, sigmoid(x @ weights + bias))

    captured = {}
    if 0 in capture_epochs:
        captured[0] = current_loss()

    for epoch in range(1, updates + 1):
        probabilities = sigmoid(x @ weights + bias)
        errors = probabilities - y
        grad_w = (x.T @ errors) / n
        grad_b = float(np.mean(errors))
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

        if epoch in capture_epochs:
            captured[epoch] = current_loss()

    return weights, bias, captured


def confusion(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, fp, tn, fn


def safe_divide(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def metrics_from_cells(tp, fp, tn, fn):
    accuracy = safe_divide(tp + tn, tp + fp + tn + fn)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    fpr = safe_divide(fp, fp + tn)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "fpr": fpr,
        "tpr": recall,
    }


def evaluate_at_threshold(y_true, probabilities, threshold):
    predicted = (probabilities >= threshold).astype(int)
    cells = confusion(y_true, predicted)
    return cells, metrics_from_cells(*cells)


def knn_predict(reference_x, reference_y, query_x, k):
    predictions = np.zeros(len(query_x), dtype=int)
    probabilities = np.zeros(len(query_x), dtype=float)
    for i, query in enumerate(query_x):
        differences = reference_x - query
        distances_squared = np.sum(differences * differences, axis=1)
        # Stable sorting preserves reference-row order when distances tie.
        nearest = stable_argsort(distances_squared)[:k]
        positives = int(np.sum(reference_y[nearest]))
        probabilities[i] = positives / k
        predictions[i] = int(positives >= (k + 1) // 2)
    return predictions, probabilities


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def fmt(value):
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.4f}"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run 'python generate_data.py' from the project folder first."
        )

    x, y = load_dataset(DATA_PATH)
    train_x, train_y, test_x, test_y = stratified_split(x, y)
    train_z, test_z, train_mean, train_sd = standardise(train_x, test_x)

    # Task 2.1: split counts and training-only standardisation.
    results = {
        "train_rows": len(train_y),
        "train_positives": int(np.sum(train_y)),
        "test_rows": len(test_y),
        "test_positives": int(np.sum(test_y)),
    }

    # Task 2.2: logistic regression, zero initialisation, 1,000 full-batch updates.
    capture = (0, 1, 10, 100, 1000)
    weights, bias, losses = fit_logistic_gradient_descent(
        train_z, train_y, 0.1, 1000, capture
    )
    for epoch in capture:
        results[f"loss_epoch_{epoch}"] = losses[epoch]
    for j, weight in enumerate(weights, start=1):
        results[f"weight_X{j}"] = float(weight)
    results["bias"] = bias

    test_probabilities = sigmoid(test_z @ weights + bias)
    results["test_bce"] = binary_cross_entropy(test_y, test_probabilities)

    # Task 2.3: independent restarts at each learning rate.
    for rate in (0.01, 0.1, 1.0):
        _, _, rate_losses = fit_logistic_gradient_descent(
            train_z, train_y, rate, 200, (200,)
        )
        results[f"loss_epoch_200_eta_{rate}"] = rate_losses[200]

    # Task 2.4: odds ratios for one training-standard-deviation increase.
    for j, weight in enumerate(weights, start=1):
        results[f"odds_ratio_X{j}"] = math.exp(float(weight))

    # Task 2.5: logistic test metrics at thresholds 0.5 and 0.3.
    for threshold in (0.5, 0.3):
        cells, metrics = evaluate_at_threshold(test_y, test_probabilities, threshold)
        label = f"tau_{threshold:.1f}"
        for cell_name, cell_value in zip(("tp", "fp", "tn", "fn"), cells):
            results[f"{label}_{cell_name}"] = cell_value
        for metric_name in (
            "accuracy", "precision", "recall", "f1", "specificity"
        ):
            results[f"{label}_{metric_name}"] = metrics[metric_name]

    blind_predictions = np.zeros_like(test_y)
    results["blind_accuracy"] = float(np.mean(blind_predictions == test_y))

    # Task 2.6: ROC threshold table.
    threshold_rows = []
    for threshold in np.arange(0.1, 1.0, 0.1):
        threshold = round(float(threshold), 1)
        cells, metrics = evaluate_at_threshold(test_y, test_probabilities, threshold)
        tp, fp, tn, fn = cells
        threshold_rows.append(
            [f"{threshold:.1f}", tp, fp, tn, fn,
             f"{metrics['tpr']:.4f}", f"{metrics['fpr']:.4f}"]
        )
    write_csv(
        THRESHOLDS_PATH,
        ["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"],
        threshold_rows,
    )

    # Task 2.7: five-fold CV; fold assignment is training position modulo 5.
    fold_ids = np.arange(len(train_y)) % 5
    k_values = list(range(1, 22, 2))
    cv_rows = []
    mean_f1_by_k = {}
    for k in k_values:
        fold_f1_scores = []
        for fold in range(5):
            validation_mask = fold_ids == fold
            reference_mask = ~validation_mask
            fold_predictions, _ = knn_predict(
                train_z[reference_mask],
                train_y[reference_mask],
                train_z[validation_mask],
                k,
            )
            fold_cells = confusion(train_y[validation_mask], fold_predictions)
            fold_metrics = metrics_from_cells(*fold_cells)
            fold_f1_scores.append(fold_metrics["f1"])
        mean_f1 = float(np.mean(fold_f1_scores))
        mean_f1_by_k[k] = mean_f1
        cv_rows.append([k, f"{mean_f1:.4f}"])

    write_csv(KNN_CV_PATH, ["K", "mean_cv_f1"], cv_rows)
    # max() returns the first item in insertion order when scores tie.
    best_k = max(k_values, key=lambda k: mean_f1_by_k[k])
    results["knn_best_k"] = best_k

    knn_test_predictions, _ = knn_predict(train_z, train_y, test_z, best_k)
    knn_cells = confusion(test_y, knn_test_predictions)
    knn_metrics = metrics_from_cells(*knn_cells)
    for cell_name, cell_value in zip(("tp", "fp", "tn", "fn"), knn_cells):
        results[f"knn_{cell_name}"] = cell_value
    for metric_name in ("accuracy", "precision", "recall", "f1"):
        results[f"knn_{metric_name}"] = knn_metrics[metric_name]

    # Task 2.8: select cost threshold on training data only.
    train_probabilities = sigmoid(train_z @ weights + bias)
    best_tau = None
    best_train_cost = None
    for k in range(1, 20):
        threshold = k / 20.0
        train_cells, _ = evaluate_at_threshold(
            train_y, train_probabilities, threshold
        )
        _, fp, _, fn = train_cells
        cost = 50 * fp + 400 * fn
        if best_train_cost is None or cost < best_train_cost:
            best_tau = threshold
            best_train_cost = cost

    results["cost_tau"] = best_tau
    results["cost_train_at_tau"] = best_train_cost

    test_cells_at_cost_tau, _ = evaluate_at_threshold(
        test_y, test_probabilities, best_tau
    )
    _, fp_tau, _, fn_tau = test_cells_at_cost_tau
    results["cost_test_at_tau"] = 50 * fp_tau + 400 * fn_tau

    test_cells_at_half, _ = evaluate_at_threshold(
        test_y, test_probabilities, 0.5
    )
    _, fp_half, _, fn_half = test_cells_at_half
    results["cost_test_at_0.5"] = 50 * fp_half + 400 * fn_half
    results["cost_test_flag_nobody"] = 400 * int(np.sum(test_y))

    OUTPUT_DIR.mkdir(exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8", newline="") as handle:
        for key, value in results.items():
            handle.write(f"{key} = {fmt(value)}\n")

    print("Part 2 completed.")
    print(f"Training rows: {len(train_y)} (positive: {int(np.sum(train_y))})")
    print(f"Test rows: {len(test_y)} (positive: {int(np.sum(test_y))})")
    print(f"Best K: {best_k}")
    print(f"Cost-selected threshold: {best_tau:.2f}")
    print(f"Results saved to: {RESULTS_PATH}")
    print(f"CSV files saved to: {THRESHOLDS_PATH} and {KNN_CV_PATH}")


if __name__ == "__main__":
    main()
