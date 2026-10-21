import csv
import os

import numpy as np

from plotting.comparison import plot_loss_comparison, plot_metric_comparison, plot_test_comparison

# config columns copied from the runs into the summary, in this order
CONFIG_COLUMNS = ["dataset", "layers", "parameters", "activation", "output_activation",
                  "loss_function", "learning_rate", "epochs", "batch_size"]


def aggregate(records, metrics):
    # one row per variant and test subset, metrics as mean and std over the seeds
    groups = {}
    for r in records:
        groups.setdefault((r["variant"], r["subset"]), []).append(r)

    rows = []
    for (variant, subset), runs in groups.items():
        row = {"variant": variant, "subset": subset, "seeds": len(runs), "points": runs[0]["points"]}
        row.update({c: runs[0][c] for c in CONFIG_COLUMNS})
        for m in metrics + ["final_train_loss", "train_seconds"]:
            values = np.array([run[m] for run in runs], dtype=float)
            row[f"{m}_mean"] = float(values.mean())
            row[f"{m}_std"] = float(values.std())
        rows.append(row)
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def best_rows(rows, metric, direction):
    # the best variant of every test subset by the mean of the metric
    sign = 1 if direction == "lower" else -1
    best = {}
    for row in rows:
        current = best.get(row["subset"])
        if current is None or sign * row[f"{metric}_mean"] < sign * current[f"{metric}_mean"]:
            best[row["subset"]] = row
    return {id(r) for r in best.values()}


def format_value(mean, std, seeds):
    return f"{mean:.5f} ± {std:.5f}" if seeds > 1 else f"{mean:.5f}"


def results_table(rows, task):
    # markdown table, the best value of the main metric in every subset is bold
    metrics = list(task.metrics)
    best = best_rows(rows, task.main_metric, task.metrics[task.main_metric])
    header = ["variant", "layers", "params", "subset"] + metrics + ["final train loss"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        cells = [row["variant"], row["layers"], str(row["parameters"]), row["subset"]]
        for m in metrics:
            value = format_value(row[f"{m}_mean"], row[f"{m}_std"], row["seeds"])
            cells.append(f"**{value}**" if m == task.main_metric and id(row) in best else value)
        cells.append(format_value(row["final_train_loss_mean"], row["final_train_loss_std"], row["seeds"]))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_report(experiment, task, records, curves, test_curves, out_dir):
    metrics = list(task.metrics)
    rows = aggregate(records, metrics)
    write_csv(records, os.path.join(out_dir, "runs.csv"))
    write_csv(rows, os.path.join(out_dir, "summary.csv"))

    seeds = sorted({r["seed"] for r in records})
    table = results_table(rows, task)
    with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as f:
        f.write(f"# {experiment.name}\n\n")
        if experiment.description:
            f.write(f"{experiment.description}\n\n")
        f.write(f"base config: `{experiment.base}`\n\n")
        f.write(f"seeds: {seeds}" + (", values are mean ± std over seeds" if len(seeds) > 1 else "") + "\n\n")
        f.write(table + "\n")
    print("\n" + table.replace("±", "+-"))  # the windows console cannot always print ±

    # test metrics after every epoch, also for a single variant
    for m in metrics:
        plot_test_comparison(test_curves, m, task.metrics[m], os.path.join(out_dir, f"test_{m}_per_epoch.png"),
                             title=experiment.name)

    # comparison plots only make sense with more than one variant
    if len(curves) > 1:
        plot_loss_comparison(curves, os.path.join(out_dir, "loss_comparison.png"), title=experiment.name)
        for m in metrics:
            plot_metric_comparison(rows, m, task.metrics[m], os.path.join(out_dir, f"{m}_comparison.png"), title=experiment.name)
