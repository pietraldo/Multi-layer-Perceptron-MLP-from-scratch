import numpy as np
import matplotlib.pyplot as plt

from plotting.results import SURFACE, INK, INK_SECONDARY, MUTED, GRID, BASELINE

# categorical palette in fixed order, a variant keeps its color in every plot of one experiment
# past 8 variants the loss plot is split into several files instead of inventing new colors
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_loss_comparison(curves, out_path, title=None):
    # curves: variant label -> list of loss curves, one per seed
    # the line is the mean over seeds, the band is the min-max range when there are several seeds
    labels = list(curves)
    chunks = [labels[i:i + len(SERIES)] for i in range(0, len(labels), len(SERIES))]
    paths = []
    for n, chunk in enumerate(chunks):
        fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
        style_axes(ax)
        for label, color in zip(chunk, SERIES):
            runs = np.array(curves[label], dtype=float)
            epochs = np.arange(1, runs.shape[1] + 1)
            ax.plot(epochs, runs.mean(axis=0), color=color, linewidth=2, label=label)
            if len(runs) > 1:
                ax.fill_between(epochs, runs.min(axis=0), runs.max(axis=0), color=color, alpha=0.15, linewidth=0)
        if all(np.min(curves[label]) > 0 for label in chunk):
            ax.set_yscale("log")
        ax.set_xlabel("epoka", color=INK_SECONDARY)
        ax.set_ylabel("błąd treningowy (funkcja straty wariantu)", color=INK_SECONDARY)
        ax.set_title("Przebieg błędu treningowego", color=INK, loc="left", fontsize=12)
        ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=8)
        if title:
            fig.suptitle(title, color=INK, fontsize=11)
        fig.tight_layout()
        path = out_path if len(chunks) == 1 else out_path.replace(".png", f"_{n + 1}.png")
        fig.savefig(path, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_metric_comparison(rows, metric, direction, out_path, title=None):
    # horizontal bars, one row per variant; error bars are the std over seeds
    # every test subset gets its own panel with its own scale, their values can differ by orders of magnitude
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    subsets = list(dict.fromkeys(r["subset"] for r in rows))
    by_key = {(r["variant"], r["subset"]): r for r in rows}
    better = "mniej = lepiej" if direction == "lower" else "więcej = lepiej"

    fig, axes = plt.subplots(1, len(subsets), figsize=(5 + 4 * len(subsets), 1.4 + 0.45 * len(variants)),
                             facecolor=SURFACE, sharey=True, squeeze=False)
    y = np.arange(len(variants))
    for ax, subset in zip(axes[0], subsets):
        style_axes(ax)
        ax.grid(False, axis="y")
        means = np.array([by_key[(v, subset)][f"{metric}_mean"] if (v, subset) in by_key else np.nan for v in variants])
        stds = np.array([by_key[(v, subset)][f"{metric}_std"] if (v, subset) in by_key else 0 for v in variants])
        ax.barh(y, means, height=0.6, xerr=stds if stds.any() else None, color=SERIES[0],
                error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1, "capsize": 3})
        for p, m, s in zip(y, means, stds):
            if not np.isnan(m):
                ax.annotate(f"{m:.4g}", (m + s, p), xytext=(4, 0), textcoords="offset points",
                            va="center", fontsize=8, color=INK_SECONDARY)
        ax.set_xlim(0, np.nanmax(means + stds) * 1.2)  # room for the value labels
        ax.set_xlabel(f"{metric} ({better})", color=INK_SECONDARY)
        ax.set_title(f"{metric}, {subset}", color=INK, loc="left", fontsize=12)

    axes[0][0].set_yticks(y, variants, color=INK_SECONDARY)
    axes[0][0].invert_yaxis()  # first variant on top, same order as the table
    if title:
        fig.suptitle(title, color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_path
