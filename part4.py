"""Part 4: Model choice and architecture calculations.
Run from the repository root: python part4.py
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
RESULTS_PATH = OUTPUT_DIR / "part4_results.txt"
SPLITS_PATH = OUTPUT_DIR / "part4_splits.csv"
VOTING_PATH = OUTPUT_DIR / "part4_voting.csv"
ADABOOST_PATH = OUTPUT_DIR / "part4_adaboost.csv"
ARCHITECTURE_PATH = OUTPUT_DIR / "part4_architecture.csv"

FEATURE_NAMES = [
    "customer_tenure_months",
    "complaint_count_90d",
    "discount_order_share",
    "avg_order_value_inr_hundreds",
    "loyalty_score",
    "cod_flag",
    "plus_member_flag",
]


def load_data(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    x = np.array(
        [[float(row[name]) for name in FEATURE_NAMES] for row in rows],
        dtype=float,
    )
    y = np.array([int(row["churned"]) for row in rows], dtype=int)
    return x, y


def stratified_split(x, y):
    train_indices = []
    test_indices = []
    for label in (0, 1):
        indices = np.flatnonzero(y == label)
        cut = (7 * len(indices)) // 10
        train_indices.extend(indices[:cut].tolist())
        test_indices.extend(indices[cut:].tolist())
    train_indices.sort()
    test_indices.sort()
    return (
        x[train_indices], y[train_indices],
        x[test_indices], y[test_indices],
    )


def standardise(train_x, test_x):
    mean = np.mean(train_x, axis=0)
    sd = np.std(train_x, axis=0, ddof=1)
    return (train_x - mean) / sd, (test_x - mean) / sd


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def fit_logistic(x, y):
    n = len(y)
    weights = np.zeros(x.shape[1], dtype=float)
    bias = 0.0
    for _ in range(1000):
        probabilities = sigmoid(x @ weights + bias)
        errors = probabilities - y
        weights -= 0.1 * (x.T @ errors) / n
        bias -= 0.1 * float(np.mean(errors))
    return weights, bias


def knn_predict(reference_x, reference_y, query_x, k):
    predictions = np.zeros(len(query_x), dtype=int)
    probabilities = np.zeros(len(query_x), dtype=float)
    for i, query in enumerate(query_x):
        distances_squared = np.sum((reference_x - query) ** 2, axis=1)
        nearest = stable_argsort(distances_squared)[:k]
        positives = int(np.sum(reference_y[nearest]))
        probabilities[i] = positives / k
        predictions[i] = int(positives >= (k + 1) // 2)
    return predictions, probabilities


def confusion(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, fp, tn, fn


def safe_divide(a, b):
    return float(a / b) if b else 0.0


def metric_row(y_true, y_pred):
    tp, fp, tn, fn = confusion(y_true, y_pred)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return tp, fp, tn, fn, precision, recall, f1


def entropy(labels):
    if len(labels) == 0:
        return 0.0
    positives = float(np.sum(labels))
    negatives = len(labels) - positives
    result = 0.0
    for count in (positives, negatives):
        if count:
            probability = count / len(labels)
            result -= probability * math.log2(probability)
    return result


def gini(labels):
    if len(labels) == 0:
        return 0.0
    p = float(np.mean(labels))
    return 1.0 - p * p - (1.0 - p) * (1.0 - p)


def split_gain(y, condition):
    yes = y[condition]
    no = y[~condition]
    parent_entropy = entropy(y)
    parent_gini = gini(y)
    weighted_entropy = (
        len(yes) / len(y) * entropy(yes)
        + len(no) / len(y) * entropy(no)
    )
    weighted_gini = (
        len(yes) / len(y) * gini(yes)
        + len(no) / len(y) * gini(no)
    )
    return (
        len(yes), int(np.sum(yes)),
        len(no), int(np.sum(no)),
        parent_entropy - weighted_entropy,
        parent_gini - weighted_gini,
    )


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
    if isinstance(value, str):
        return value
    return f"{float(value):.4f}"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run 'python generate_data.py' from the project folder first."
        )

    x, y = load_data(DATA_PATH)
    train_x, train_y, test_x, test_y = stratified_split(x, y)
    train_z, test_z = standardise(train_x, test_x)

    results = {}

    # Task 4.1: root impurity and candidate one-question splits.
    results["root_entropy"] = entropy(train_y)
    results["root_gini"] = gini(train_y)

    conditions = {
        "S1": train_x[:, 5] == 1,
        "S2": train_x[:, 6] == 1,
        "S3": train_x[:, 2] > 0.5,
        "S4": train_x[:, 0] <= 24,
    }
    split_rows = []
    info_gains = {}
    gini_gains = {}
    split_rates = {}
    for name in ("S1", "S2", "S3", "S4"):
        condition = conditions[name]
        n_yes, pos_yes, n_no, pos_no, info_gain, gini_gain = split_gain(
            train_y, condition
        )
        split_rows.append(
            [name, n_yes, pos_yes, n_no, pos_no,
             f"{info_gain:.4f}", f"{gini_gain:.4f}"]
        )
        info_gains[name] = info_gain
        gini_gains[name] = gini_gain
        split_rates[name] = (
            safe_divide(pos_yes, n_yes),
            safe_divide(pos_no, n_no),
        )

    best_info_split = max(("S1", "S2", "S3", "S4"),
                          key=lambda name: info_gains[name])
    best_gini_split = max(("S1", "S2", "S3", "S4"),
                          key=lambda name: gini_gains[name])
    results["best_split_by_info_gain"] = best_info_split
    results["best_split_by_gini_gain"] = best_gini_split

    # Task 4.2: reproduce Part 2 logistic model and best KNN model.
    logistic_weights, logistic_bias = fit_logistic(train_z, train_y)
    logistic_prob = sigmoid(test_z @ logistic_weights + logistic_bias)
    logistic_pred = (logistic_prob >= 0.3).astype(int)

    # The CV fold definition and stable distance tie-break match Part 2.
    fold_ids = np.arange(len(train_y)) % 5
    best_k = None
    best_cv_f1 = -1.0
    for k in range(1, 22, 2):
        scores = []
        for fold in range(5):
            validation_mask = fold_ids == fold
            reference_mask = ~validation_mask
            pred, _ = knn_predict(
                train_z[reference_mask], train_y[reference_mask],
                train_z[validation_mask], k,
            )
            tp, fp, tn, fn = confusion(train_y[validation_mask], pred)
            precision = safe_divide(tp, tp + fp)
            recall = safe_divide(tp, tp + fn)
            scores.append(safe_divide(2 * precision * recall,
                                      precision + recall))
        mean_score = float(np.mean(scores))
        if mean_score > best_cv_f1:
            best_cv_f1 = mean_score
            best_k = k

    knn_pred, knn_prob = knn_predict(train_z, train_y, test_z, best_k)

    # Stump probability is the training positive share on the matching side.
    stump_rate_yes, stump_rate_no = split_rates[best_info_split]
    stump_condition_test = {
        "S1": test_x[:, 5] == 1,
        "S2": test_x[:, 6] == 1,
        "S3": test_x[:, 2] > 0.5,
        "S4": test_x[:, 0] <= 24,
    }[best_info_split]
    stump_prob = np.where(
        stump_condition_test, stump_rate_yes, stump_rate_no
    )
    stump_pred = (stump_prob >= 0.3).astype(int)

    hard_vote = (
        logistic_pred.astype(int)
        + knn_pred.astype(int)
        + stump_pred.astype(int)
        >= 2
    ).astype(int)
    soft_probability = (logistic_prob + knn_prob + stump_prob) / 3.0
    soft_vote = (soft_probability >= 0.3).astype(int)

    voting_rows = []
    for name, prediction in (
        ("logistic", logistic_pred),
        ("knn", knn_pred),
        ("stump", stump_pred),
        ("hard_vote", hard_vote),
        ("soft_vote", soft_vote),
    ):
        tp, fp, tn, fn, precision, recall, f1 = metric_row(test_y, prediction)
        voting_rows.append(
            [name, tp, fp, tn, fn, f"{precision:.4f}",
             f"{recall:.4f}", f"{f1:.4f}"]
        )

    results["stump_rate_yes"] = stump_rate_yes
    results["stump_rate_no"] = stump_rate_no

    # Task 4.3: two rounds of AdaBoost with the supplied eight observations.
    true_y = np.array([1, 1, 1, -1, -1, -1, 1, -1], dtype=int)
    h1 = np.array([1, 1, -1, -1, -1, -1, -1, -1], dtype=int)
    h2 = np.array([-1, 1, 1, -1, 1, -1, 1, -1], dtype=int)
    weights = np.full(8, 1.0 / 8.0)
    adaboost_rows = []
    alphas = []
    for round_no, hypothesis in ((1, h1), (2, h2)):
        before = weights.copy()
        error = float(np.sum(before[hypothesis != true_y]))
        alpha = 0.5 * math.log((1.0 - error) / error)
        unnormalised = before * np.exp(-alpha * true_y * hypothesis)
        weights = unnormalised / np.sum(unnormalised)
        alphas.append(alpha)
        for point in range(8):
            adaboost_rows.append(
                [round_no, point + 1, f"{before[point]:.4f}",
                 f"{unnormalised[point]:.4f}", f"{weights[point]:.4f}"]
            )
        results[f"round{round_no}_error"] = error
        results[f"round{round_no}_alpha"] = alpha

    ensemble_score = alphas[0] * h1 + alphas[1] * h2
    ensemble_prediction = np.where(ensemble_score >= 0, 1, -1)
    results["ensemble_accuracy"] = float(np.mean(ensemble_prediction == true_y))

    # Task 4.4: expected out-of-bag fraction.
    results["oob_fraction"] = (1.0 - 1.0 / 560.0) ** 560

    # Task 4.5: parameter counts for MLP, CNN, convolution comparison, RNN.
    architecture_rows = [
        ["mlp_hidden1", 784 * 16 + 16],
        ["mlp_hidden2", 16 * 16 + 16],
        ["mlp_output", 16 * 10 + 10],
        ["cnn_conv1", (5 * 5 * 1 + 1) * 6],
        ["cnn_conv2", (5 * 5 * 6 + 1) * 16],
        ["cnn_dense1", (5 * 5 * 16) * 32 + 32],
        ["cnn_dense2", 32 * 10 + 10],
    ]
    mlp_total = sum(row[1] for row in architecture_rows[:3])
    cnn_total = sum(row[1] for row in architecture_rows[3:])
    results["mlp_total"] = mlp_total
    results["cnn_total"] = cnn_total

    results["fc_layer_params"] = (128 * 128 * 3) * 1024 + 1024
    results["conv_layer_params"] = (3 * 3 * 3 + 1) * 64

    wxh = 128 * 64
    whh = 128 * 128
    bh = 128
    why = 16 * 128
    by = 16
    results["rnn_total"] = wxh + whh + bh + why + by

    OUTPUT_DIR.mkdir(exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8", newline="") as handle:
        for key, value in results.items():
            handle.write(f"{key} = {fmt(value)}\n")

    write_csv(
        SPLITS_PATH,
        ["split", "n_yes", "positives_yes", "n_no", "positives_no",
         "info_gain", "gini_gain"],
        split_rows,
    )
    write_csv(
        VOTING_PATH,
        ["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"],
        voting_rows,
    )
    write_csv(
        ADABOOST_PATH,
        ["round", "point", "weight_before", "weight_unnormalised",
         "weight_after"],
        adaboost_rows,
    )
    write_csv(
        ARCHITECTURE_PATH,
        ["layer", "parameters"],
        architecture_rows,
    )

    print("Part 4 completed.")
    print(f"Training rows: {len(train_y)}; test rows: {len(test_y)}")
    print(f"Best split by information gain: {best_info_split}")
    print(f"Best split by Gini gain: {best_gini_split}")
    print(f"Round 1 AdaBoost error: {results['round1_error']:.4f}")
    print(f"MLP parameters: {mlp_total}; CNN parameters: {cnn_total}")
    print(f"RNN parameters: {results['rnn_total']}")
    print(f"Results saved to: {RESULTS_PATH}")
    print("CSV files saved to outputs/.")


if __name__ == "__main__":
    main()
