import os
import re
import time
from datetime import datetime

from experiments.config import value_label
from experiments.report import save_report
from experiments.tasks import task_for
from nn.mlp import MLP
from plotting.network import plot_network_html
from plotting.results import plot_loss

RESULTS_DIR = "wyniki"


def new_run_dir(name):
    # every experiment gets its own folder wyniki/<date_time>_<name>, all of its files go there and nowhere else
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(RESULTS_DIR, f"{stamp}_{name}")
    os.makedirs(path, exist_ok=True)
    return path


def safe_name(label):
    # variant label as a folder name
    return re.sub(r"[^\w.-]+", "_", label).strip("_")


def build_model(cfg, task, data):
    layers = [data.x.shape[1], *cfg.hidden, data.y.shape[1]]
    output_activation = cfg.output_activation or task.default_output_activation
    loss = cfg.loss or task.default_loss
    return MLP(layers, cfg.activation(), output_activation(), loss(), cfg.learning_rate, seed=cfg.seed)


def count_params(layers):
    # weights plus biases of every layer
    return sum(layers[i] * layers[i + 1] + layers[i + 1] for i in range(len(layers) - 1))


def describe(cfg, task, model):
    # the config as flat columns for the summary tables
    return {
        "dataset": cfg.dataset,
        "layers": value_label(model.layers),
        "parameters": count_params(model.layers),
        "activation": value_label(cfg.activation),
        "output_activation": type(model.output_activation).__name__,
        "loss_function": type(model.loss_function).__name__,
        "learning_rate": cfg.learning_rate,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
    }


def run_experiment(experiment, seeds=None, epochs=None):
    # seeds and epochs override the experiment's own values, handy for a quick check
    seeds = seeds or experiment.seeds
    out_dir = new_run_dir(experiment.name)
    print(f"experiment {experiment.name}: {len(experiment.variants)} variants x {len(seeds)} seeds -> {out_dir}")

    data_cache = {}  # every variant on the same dataset sees exactly the same data
    records = []     # one row per variant, seed and test subset
    curves = {}      # variant label -> list of loss curves, one per seed
    task = None
    for label, changes in experiment.variants.items():
        curves[label] = []
        for i, seed in enumerate(seeds):
            cfg = experiment.base.with_(**changes, seed=seed)
            if epochs:
                cfg = cfg.with_(epochs=epochs)
            task = task_for(cfg.dataset)
            if cfg.dataset not in data_cache:
                data_cache[cfg.dataset] = task.load(cfg.dataset)
            data = data_cache[cfg.dataset]

            model = build_model(cfg, task, data)
            start = time.perf_counter()
            losses = model.train(cfg.epochs, cfg.batch_size, data.x, data.y, verbose=False)
            seconds = time.perf_counter() - start
            curves[label].append(losses)

            results = task.evaluate(model, data)
            scores = ", ".join(f"{r['subset']} {task.main_metric}={r[task.main_metric]:.5f}" for r in results)
            print(f"  [{label}] seed {seed}: final loss {losses[-1]:.5f}, {scores} ({seconds:.1f}s)")

            for r in results:
                records.append({
                    "variant": label, "seed": seed, **describe(cfg, task, model),
                    "subset": r["subset"], "points": r["points"], "final_train_loss": losses[-1],
                    **{m: r[m] for m in task.metrics}, "train_seconds": round(seconds, 2),
                })

            # detailed files only for the first seed, otherwise the folder gets huge
            if i == 0:
                run_dir = os.path.join(out_dir, "runs", safe_name(label))
                os.makedirs(run_dir, exist_ok=True)
                title = f"{experiment.name}: {label}"
                model.save(os.path.join(run_dir, "model.json"))
                plot_loss(losses, os.path.join(run_dir, "loss.png"), title=title)
                for r in results:
                    task.save_plots(r, data, run_dir, title)
                if experiment.network_html:
                    plot_network_html(model.history, losses, model.layers, os.path.join(run_dir, "network.html"), title=title)

    save_report(experiment, task, records, curves, out_dir)
    print(f"\nresults saved to {out_dir}")
    return out_dir
