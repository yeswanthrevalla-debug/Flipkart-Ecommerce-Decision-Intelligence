"""Part 1: Estimation, inference and regression baseline.
Run from the repository root: python part1.py
Only Python's standard library and NumPy are used.
"""
import csv
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "records.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_PATH = OUTPUT_DIR / "part1_results.txt"


def read_data(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    # The assignment defines X1..X7 as columns 2..8,
    # outcome as column 9, and value as column 10.
    x4 = np.array(
        [float(row["avg_order_value_inr_hundreds"]) for row in rows],
        dtype=float,
    )
    value = np.array(
        [float(row["order_value_inr_hundreds"]) for row in rows],
        dtype=float,
    )
    x1 = np.array([float(row["customer_tenure_months"]) for row in rows])
    x2 = np.array([float(row["complaint_count_90d"]) for row in rows])
    x3 = np.array([float(row["discount_order_share"]) for row in rows])
    x5 = np.array([float(row["loyalty_score"]) for row in rows])
    x6 = np.array([float(row["cod_flag"]) for row in rows])
    x7 = np.array([float(row["plus_member_flag"]) for row in rows])
    return rows, x1, x2, x3, x4, x5, x6, x7, value


def simple_regression(x, y):
    """Closed-form simple linear regression using sample covariance/variance."""
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    x_centered = x - x_mean
    y_centered = y - y_mean

    covariance = float(np.sum(x_centered * y_centered) / (len(x) - 1))
    variance = float(np.sum(x_centered * x_centered) / (len(x) - 1))
    slope = covariance / variance
    intercept = y_mean - slope * x_mean
    fitted = intercept + slope * x

    tss = float(np.sum((y - y_mean) ** 2))
    ess = float(np.sum((fitted - y_mean) ** 2))
    residuals = y - fitted
    rss = float(np.sum(residuals ** 2))
    r2 = 1.0 - rss / tss
    rmse = math.sqrt(rss / len(y))
    mae = float(np.mean(np.abs(residuals)))

    return {
        "slope": slope,
        "intercept": intercept,
        "fitted": fitted,
        "residuals": residuals,
        "tss": tss,
        "ess": ess,
        "rss": rss,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
    }


def residual_group_sds(fitted, residuals):
    # Stable sort is required because many fitted values are tied.
    order = np.argsort(fitted, kind="stable")
    lower = residuals[order[:400]]
    upper = residuals[order[400:]]
    return float(np.std(lower, ddof=1)), float(np.std(upper, ddof=1))


def gradient_descent(x, y):
    n = len(y)
    x_mean = float(np.mean(x))
    x_sd = float(np.std(x, ddof=1))
    z = (x - x_mean) / x_sd

    m = 0.0
    c = 0.0
    learning_rate = 0.1

    def cost(current_m, current_c):
        errors = y - (current_m * z + current_c)
        return float(np.mean(errors ** 2))

    costs = {"gd_J0": cost(m, c)}
    for step in range(1, 501):
        errors = y - (m * z + c)
        dm = -(2.0 / n) * float(np.sum(z * errors))
        dc = -(2.0 / n) * float(np.sum(errors))
        m -= learning_rate * dm
        c -= learning_rate * dc

        if step in (1, 10, 100, 500):
            costs[f"gd_J{step}"] = cost(m, c)

    slope = m / x_sd
    intercept = c - m * x_mean / x_sd
    return costs, slope, intercept


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

    _, x1, x2, x3, x4, x5, x6, x7, value = read_data(DATA_PATH)
    n = len(value)

    # Task 1.1: estimate mean and 95% confidence interval.
    mean_value = float(np.mean(value))
    sd_value = float(np.std(value, ddof=1))
    se_value = sd_value / math.sqrt(n)
    ci95_low = mean_value - 1.96 * se_value
    ci95_high = mean_value + 1.96 * se_value

    # Task 1.2: two-sided z test of H0: population mean = 8.0.
    z_stat = (mean_value - 8.0) / se_value
    reject_h0 = abs(z_stat) > 1.96

    # Task 1.3: required sample size for margin of error 0.5.
    n_required = math.ceil((1.96 * sd_value / 0.5) ** 2)

    # Task 1.4: closed-form regression of value on X4.
    raw_model = simple_regression(x4, value)
    tss_equals = (
        abs(raw_model["tss"] - raw_model["ess"] - raw_model["rss"])
        < 1e-6 * raw_model["tss"]
    )

    # Task 1.5: batch gradient descent on standardised X4.
    gd_costs, gd_slope, gd_intercept = gradient_descent(x4, value)
    gd_slope_gap = abs(gd_slope - raw_model["slope"])

    # Task 1.6: compare residual spread before and after log transformation.
    raw_lower_sd, raw_upper_sd = residual_group_sds(
        raw_model["fitted"], raw_model["residuals"]
    )
    resid_sd_ratio_raw = raw_upper_sd / raw_lower_sd

    log_x4 = np.log(x4)
    log_value = np.log(value)
    log_model = simple_regression(log_x4, log_value)
    log_lower_sd, log_upper_sd = residual_group_sds(
        log_model["fitted"], log_model["residuals"]
    )
    resid_sd_ratio_log = log_upper_sd / log_lower_sd

    # Task 1.7: multiple regression using the specified normal equation.
    design = np.column_stack(
        [
            np.ones(n),
            log_x4,
            x1,
            x2,
            x3,
            x5,
            x6,
            x7,
        ]
    )
    beta = np.linalg.solve(design.T @ design, design.T @ log_value)
    mr_fitted = design @ beta
    mr_tss = float(np.sum((log_value - np.mean(log_value)) ** 2))
    mr_rss = float(np.sum((log_value - mr_fitted) ** 2))
    mr_r2 = 1.0 - mr_rss / mr_tss
    p = 7
    mr_adj_r2 = 1.0 - (1.0 - mr_r2) * (n - 1) / (n - p - 1)
    loglog_adj_r2 = 1.0 - (1.0 - log_model["r2"]) * (n - 1) / (n - 1 - 1)

    results = {
        "n": n,
        "mean_value": mean_value,
        "sd_value": sd_value,
        "se_value": se_value,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "z_stat": z_stat,
        "reject_h0": reject_h0,
        "n_required": n_required,
        "reg_slope": raw_model["slope"],
        "reg_intercept": raw_model["intercept"],
        "tss": raw_model["tss"],
        "ess": raw_model["ess"],
        "rss": raw_model["rss"],
        "tss_equals_ess_plus_rss": tss_equals,
        "r2": raw_model["r2"],
        "rmse": raw_model["rmse"],
        "mae": raw_model["mae"],
        **gd_costs,
        "gd_slope": gd_slope,
        "gd_intercept": gd_intercept,
        "gd_slope_gap": gd_slope_gap,
        "resid_sd_lower_raw": raw_lower_sd,
        "resid_sd_upper_raw": raw_upper_sd,
        "resid_sd_ratio_raw": resid_sd_ratio_raw,
        "loglog_intercept": log_model["intercept"],
        "loglog_slope": log_model["slope"],
        "loglog_r2": log_model["r2"],
        "loglog_rmse": log_model["rmse"],
        "resid_sd_lower_log": log_lower_sd,
        "resid_sd_upper_log": log_upper_sd,
        "resid_sd_ratio_log": resid_sd_ratio_log,
        **{f"mr_beta_{i}": beta[i] for i in range(8)},
        "mr_r2": mr_r2,
        "mr_adj_r2": mr_adj_r2,
        "loglog_adj_r2": loglog_adj_r2,
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        for key, result in results.items():
            handle.write(f"{key} = {fmt(result)}\n")

    print(f"Part 1 completed. Results saved to: {OUTPUT_PATH}")
    print(f"Rows analysed: {n}")
    print(f"Mean order value: {mean_value:.4f} hundreds of INR")
    print(f"Gradient-descent slope gap: {gd_slope_gap:.6f}")
    print(f"Result keys written: {len(results)}")


if __name__ == "__main__":
    main()
