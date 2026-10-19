import numpy as np
import matplotlib.pyplot as plt

# colors from the reference palette (light mode)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
TARGET = "#2a78d6"      # categorical slot 1, blue
PREDICTED = "#eb6834"   # categorical slot 2, orange
AQUA = "#1baf7a"        # categorical slot 3, aqua
# the first three palette slots pass the color checks together, past three the palette says to fold classes
CLASS_COLORS = [TARGET, PREDICTED, AQUA]


def plot_loss(losses, out_path, title=None):
    # losses is the list returned by MLP.train, one loss value per epoch
    epochs = np.arange(1, len(losses) + 1)

    fig, ax = plt.subplots(figsize=(7, 4.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    ax.plot(epochs, losses, color=TARGET, linewidth=2, label="błąd treningowy")
    ax.set_xlabel("epoka", color=INK_SECONDARY)
    ax.set_ylabel("błąd (suma po zbiorze)", color=INK_SECONDARY)
    ax.set_title(f"Przebieg błędu, końcowy = {losses[-1]:.4f}", color=INK, loc="left", fontsize=12)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY)

    if title:
        fig.suptitle(title, color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def plot_regression(x, csv_path, out_path, title=None):
    # csv_path is the file written by test(): columns predicted,target
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, ndmin=2)
    predicted = data[:, 0]
    target = data[:, 1]
    mse = np.mean((predicted - target) ** 2)

    order = np.argsort(x[:, 0])
    xs = x[order, 0]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 4.8), facecolor=SURFACE,
        gridspec_kw={"width_ratios": [1.6, 1]},
    )
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(BASELINE)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    # left: both values against x, sorted by x so the curves read as functions
    ax1.plot(xs, target[order], color=TARGET, linewidth=2, marker="o", markersize=4, label="cel (y)")
    ax1.plot(xs, predicted[order], color=PREDICTED, linewidth=2, marker="o", markersize=4, label="predykcja sieci")
    ax1.set_xlabel("x", color=INK_SECONDARY)
    ax1.set_ylabel("y (przeskalowane)", color=INK_SECONDARY)
    ax1.set_title("Wartości w funkcji x", color=INK, loc="left", fontsize=12)
    ax1.legend(frameon=False, labelcolor=INK_SECONDARY)

    # right: predicted vs target, the dashed diagonal is a perfect fit
    lo = min(target.min(), predicted.min())
    hi = max(target.max(), predicted.max())
    ax2.plot([lo, hi], [lo, hi], color=BASELINE, linewidth=1, linestyle="--", label="idealnie")
    ax2.scatter(target, predicted, s=28, color=PREDICTED, edgecolor=SURFACE, linewidth=0.5, label="próbki")
    ax2.set_xlabel("cel (y)", color=INK_SECONDARY)
    ax2.set_ylabel("predykcja", color=INK_SECONDARY)
    ax2.set_title(f"Predykcja vs cel, MSE = {mse:.4f}", color=INK, loc="left", fontsize=12)
    ax2.legend(frameon=False, labelcolor=INK_SECONDARY)

    if title:
        fig.suptitle(title, color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def plot_classification(x, csv_path, out_path, title=None):
    # csv_path is the file written by predict_to_csv: columns predicted,target (classes 1 and 2)
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, ndmin=2)
    predicted = data[:, 0]
    target = data[:, 1]
    accuracy = np.mean(predicted == target)
    wrong = predicted != target

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.8), facecolor=SURFACE, sharex=True, sharey=True)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(BASELINE)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("x1", color=INK_SECONDARY)
        ax.set_ylabel("x2", color=INK_SECONDARY)

    # one color per class, in palette order, so the same class always gets the same color
    classes = np.unique(np.concatenate([target, predicted]))
    if len(classes) > len(CLASS_COLORS):
        raise ValueError(f"plot supports up to {len(CLASS_COLORS)} classes, got {len(classes)}")

    # left: true classes, right: what the network predicted
    for cls, color in zip(classes, CLASS_COLORS):
        mask = target == cls
        ax1.scatter(x[mask, 0], x[mask, 1], s=28, color=color, edgecolor=SURFACE, linewidth=0.5, label=f"klasa {int(cls)}")
        mask = predicted == cls
        ax2.scatter(x[mask, 0], x[mask, 1], s=28, color=color, edgecolor=SURFACE, linewidth=0.5, label=f"klasa {int(cls)}")

    # errors get a ring on the predicted side, so they are easy to find
    ax2.scatter(x[wrong, 0], x[wrong, 1], s=90, facecolors="none", edgecolors=INK, linewidths=1, label="błąd")

    ax1.set_title("Prawdziwe klasy", color=INK, loc="left", fontsize=12)
    ax2.set_title(f"Predykcja, accuracy = {accuracy:.2%}", color=INK, loc="left", fontsize=12)
    ax1.legend(frameon=False, labelcolor=INK_SECONDARY)
    ax2.legend(frameon=False, labelcolor=INK_SECONDARY)

    if title:
        fig.suptitle(title, color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_path
