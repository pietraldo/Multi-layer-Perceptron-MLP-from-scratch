import argparse
import glob
import os
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from nn.mlp import MLP
from training_mnist_model import MODELS_DIR, ROOT, flatten_images

# demo model shipped with the project, used when no model has been trained yet
DEMO_MODEL = os.path.join(ROOT, "models", "demo", "model.json")

# drawing area, the drawing is kept as a grayscale image of this size and scaled down to 28x28 for the model
CANVAS_SIZE = 320
BRUSH_WIDTH = 26
PREDICTION_DELAY_MS = 60  # while drawing, predict at most this often

# MNIST digits fit in a 20x20 box placed in a 28x28 image, centered by the center of mass
MNIST_SIZE = 28
MNIST_BOX = 20

# colors from the project palette (plotting/results.py)
SURFACE = "#fcfcfb"
CARD = "#ffffff"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
ACCENT = "#2a78d6"
WARNING = "#eb6834"
FONT = "Segoe UI"


# ---------------------------------------------------------------- model

def find_latest_model(models_dir):
    # model.json from the last epoch of the newest run, folder names sort by date and epoch number
    paths = sorted(glob.glob(os.path.join(models_dir, "run_*", "epoch_*", "model.json")))
    return paths[-1] if paths else None


def display_path(path):
    # short path inside the project (models\mnist\run_...\epoch_...), full path elsewhere
    path = os.path.abspath(path)
    return os.path.relpath(path, ROOT) if path.startswith(ROOT + os.sep) else path


def load_model(path):
    model = MLP.load(path)
    if model.layers[0] != MNIST_SIZE * MNIST_SIZE:
        raise ValueError(f"model expects {model.layers[0]} inputs, a 28x28 image gives {MNIST_SIZE * MNIST_SIZE}")
    return model


def predict(model, x):
    # probabilities of the 10 digits for one flattened image
    return model.feedforward(x)[0][-1]


# ---------------------------------------------------------------- image processing

def to_mnist_image(drawing):
    # drawing: grayscale PIL image, 255 where the digit is drawn and 0 elsewhere (same as MNIST)
    # returns a 28x28 uint8 array prepared like MNIST, or None when nothing is drawn
    pixels = np.array(drawing)
    rows = np.flatnonzero(pixels.max(axis=1))
    cols = np.flatnonzero(pixels.max(axis=0))
    if len(rows) == 0:
        return None

    # crop to the digit and scale the longer side to 20 px, keeping the proportions
    digit = drawing.crop((cols[0], rows[0], cols[-1] + 1, rows[-1] + 1))
    scale = MNIST_BOX / max(digit.size)
    size = (max(1, round(digit.width * scale)), max(1, round(digit.height * scale)))
    digit = np.array(digit.resize(size, Image.LANCZOS), dtype=float)

    # place it so the center of mass lands in the middle of the 28x28 image
    ys, xs = np.indices(digit.shape)
    total = digit.sum()
    cy, cx = (ys * digit).sum() / total, (xs * digit).sum() / total
    top = int(np.clip(round(MNIST_SIZE / 2 - cy), 0, MNIST_SIZE - digit.shape[0]))
    left = int(np.clip(round(MNIST_SIZE / 2 - cx), 0, MNIST_SIZE - digit.shape[1]))

    image = np.zeros((MNIST_SIZE, MNIST_SIZE))
    image[top:top + digit.shape[0], left:left + digit.shape[1]] = digit
    return np.clip(image, 0, 255).astype(np.uint8)


def prepare_input(image):
    # the same flattening and scaling to [0, 1] as the training data
    return flatten_images([image])[0]


# ---------------------------------------------------------------- user interface

class DigitRecognizerApp:

    def __init__(self, root, model_path):
        self.root = root
        self.model_path = model_path
        self.model = None
        self.last_point = None
        self.pending_prediction = None

        self.drawing = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.pen = ImageDraw.Draw(self.drawing)

        self.build_ui()
        self.bind_events()
        self.load(model_path)
        self.show_empty_result()

    # ---------- layout

    def build_ui(self):
        self.root.title("Rozpoznawanie cyfr, MNIST")
        self.root.configure(bg=SURFACE)
        self.root.resizable(False, False)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", font=(FONT, 10), padding=(14, 7), background=CARD, foreground=INK,
                        bordercolor=GRID, lightcolor=CARD, darkcolor=CARD, focuscolor=CARD)
        style.map("TButton", background=[("active", GRID)])
        style.configure("Accent.TButton", background=ACCENT, foreground="white", bordercolor=ACCENT,
                        lightcolor=ACCENT, darkcolor=ACCENT)
        style.map("Accent.TButton", background=[("active", "#1f62b3")])

        main = tk.Frame(self.root, bg=SURFACE, padx=24, pady=20)
        main.pack()

        self.build_drawing_panel(main).pack(side="left", anchor="n")
        self.build_result_panel(main).pack(side="left", anchor="n", padx=(28, 0))

        # width=1 so a long model path is cut off instead of widening the window
        self.status = tk.Label(self.root, anchor="w", width=1, font=(FONT, 9), bg=GRID, fg=INK_SECONDARY,
                               padx=12, pady=4)
        self.status.pack(fill="x", side="bottom")

    def build_drawing_panel(self, parent):
        panel = tk.Frame(parent, bg=SURFACE)
        tk.Label(panel, text="Narysuj cyfrę", font=(FONT, 16, "bold"), bg=SURFACE, fg=INK).pack(anchor="w")
        tk.Label(panel, text="lewy przycisk myszy rysuje, prawy wyciera", font=(FONT, 9), bg=SURFACE, fg=MUTED) \
            .pack(anchor="w", pady=(0, 10))
        self.build_canvas(panel).pack()
        self.build_buttons(panel).pack(fill="x", pady=(12, 0))
        return panel

    def build_canvas(self, parent):
        border = tk.Frame(parent, bg=GRID, padx=1, pady=1)
        self.canvas = tk.Canvas(border, width=CANVAS_SIZE, height=CANVAS_SIZE, bg=CARD,
                                highlightthickness=0, cursor="pencil")
        self.canvas.pack()
        return border

    def build_buttons(self, parent):
        frame = tk.Frame(parent, bg=SURFACE)
        ttk.Button(frame, text="Wyczyść  (C)", style="Accent.TButton", command=self.clear).pack(side="left")
        ttk.Button(frame, text="Reset  (R)", command=self.reset).pack(side="left", padx=8)
        ttk.Button(frame, text="Wczytaj model…", command=self.choose_model).pack(side="right")
        return frame

    def build_result_panel(self, parent):
        panel = tk.Frame(parent, bg=SURFACE)

        tk.Label(panel, text="Rozpoznana cyfra", font=(FONT, 10), bg=SURFACE, fg=INK_SECONDARY).pack(anchor="w")
        self.digit_label = tk.Label(panel, font=(FONT, 72, "bold"), bg=SURFACE, fg=INK, width=2, anchor="w")
        self.digit_label.pack(anchor="w")
        self.confidence_label = tk.Label(panel, font=(FONT, 13), bg=SURFACE, fg=INK_SECONDARY, anchor="w")
        self.confidence_label.pack(anchor="w", pady=(0, 14))

        tk.Label(panel, text="Prawdopodobieństwo każdej cyfry", font=(FONT, 10), bg=SURFACE, fg=INK_SECONDARY) \
            .pack(anchor="w", pady=(0, 4))
        self.bars = tk.Canvas(panel, width=280, height=10 * 22, bg=SURFACE, highlightthickness=0)
        self.bars.pack(anchor="w")

        preview_row = tk.Frame(panel, bg=SURFACE)
        preview_row.pack(anchor="w", pady=(16, 0))
        self.preview_label = tk.Label(preview_row, bg=INK, bd=0)
        self.preview_label.pack(side="left")
        tk.Label(preview_row, text="Tak widzi to model:\n28 × 28 pikseli,\nwyśrodkowane jak w MNIST",
                 font=(FONT, 9), bg=SURFACE, fg=MUTED, justify="left").pack(side="left", padx=(10, 0), anchor="n")
        return panel

    def bind_events(self):
        self.canvas.bind("<ButtonPress-1>", lambda e: self.start_stroke(e, erase=False))
        self.canvas.bind("<B1-Motion>", lambda e: self.continue_stroke(e, erase=False))
        self.canvas.bind("<ButtonPress-3>", lambda e: self.start_stroke(e, erase=True))
        self.canvas.bind("<B3-Motion>", lambda e: self.continue_stroke(e, erase=True))
        self.canvas.bind("<ButtonRelease-1>", self.end_stroke)
        self.canvas.bind("<ButtonRelease-3>", self.end_stroke)
        self.root.bind("<KeyPress-c>", lambda e: self.clear())
        self.root.bind("<KeyPress-r>", lambda e: self.reset())
        self.root.bind("<Escape>", lambda e: self.root.destroy())

    # ---------- drawing

    def start_stroke(self, event, erase):
        self.last_point = (event.x, event.y)
        self.draw_segment(self.last_point, self.last_point, erase)

    def continue_stroke(self, event, erase):
        point = (event.x, event.y)
        self.draw_segment(self.last_point or point, point, erase)
        self.last_point = point
        self.schedule_prediction()

    def end_stroke(self, event):
        self.last_point = None
        self.update_prediction()

    def draw_segment(self, start, end, erase):
        # the same stroke goes to the screen and to the image the model reads
        width = BRUSH_WIDTH * 2 if erase else BRUSH_WIDTH
        color = CARD if erase else INK
        self.canvas.create_line(*start, *end, width=width, fill=color, capstyle="round", smooth=True)

        r = width / 2
        value = 0 if erase else 255
        self.pen.line([start, end], fill=value, width=width)
        for x, y in (start, end):  # round ends, so joined segments have no gaps
            self.pen.ellipse([x - r, y - r, x + r, y + r], fill=value)

    def clear(self):
        self.canvas.delete("all")
        self.pen.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=0)
        self.show_empty_result()

    def reset(self):
        # clean start: empty canvas and the model read again from disk, e.g. after more training epochs
        self.clear()
        self.load(self.model_path)

    # ---------- model and prediction

    def load(self, path):
        if not path:
            self.model = None
            self.set_status("brak modelu, wybierz plik model.json przyciskiem „Wczytaj model…”", error=True)
            return
        try:
            self.model = load_model(path)
        except (OSError, ValueError, KeyError) as error:
            self.model = None
            self.set_status(f"nie udało się wczytać {path}: {error}", error=True)
            return
        self.model_path = path
        self.set_status(f"architektura {'-'.join(map(str, self.model.layers))}   ·   model: {display_path(path)}")

    def choose_model(self):
        start = os.path.dirname(self.model_path) if self.model_path else MODELS_DIR
        path = filedialog.askopenfilename(title="Wybierz model", initialdir=start,
                                          filetypes=[("model sieci", "*.json"), ("wszystkie pliki", "*.*")])
        if path:
            self.load(path)
            self.update_prediction()

    def schedule_prediction(self):
        # predicting on every mouse move would lag, so moves within a short time share one prediction
        if self.pending_prediction is None:
            self.pending_prediction = self.root.after(PREDICTION_DELAY_MS, self.update_prediction)

    def update_prediction(self):
        if self.pending_prediction is not None:
            self.root.after_cancel(self.pending_prediction)
            self.pending_prediction = None

        image = to_mnist_image(self.drawing)
        if image is None:
            self.show_empty_result()
            return
        self.show_preview(image)
        if self.model is not None:
            self.show_result(predict(self.model, prepare_input(image)))

    # ---------- result display

    def show_result(self, probabilities):
        digit = int(np.argmax(probabilities))
        confidence = 100 * probabilities[digit]
        unsure = confidence < 60
        self.digit_label.config(text=str(digit), fg=WARNING if unsure else INK)
        self.confidence_label.config(text=f"pewność {confidence:.1f}%" + ("  ·  model nie jest pewny" if unsure else ""),
                                     fg=WARNING if unsure else INK_SECONDARY)
        self.draw_bars(probabilities)

    def show_empty_result(self):
        self.digit_label.config(text="–", fg=GRID)
        self.confidence_label.config(text="narysuj cyfrę w polu obok", fg=MUTED)
        self.draw_bars(np.zeros(10))
        self.show_preview(np.zeros((MNIST_SIZE, MNIST_SIZE), dtype=np.uint8))

    def draw_bars(self, probabilities):
        # one row per digit: label, bar on a light track, percentage; the predicted digit is highlighted
        self.bars.delete("all")
        best = int(np.argmax(probabilities)) if probabilities.any() else None
        row, label_w, value_w = 22, 18, 56
        track_w = int(self.bars["width"]) - label_w - value_w
        for digit, p in enumerate(probabilities):
            y = digit * row + row / 2
            color = ACCENT if digit == best else MUTED
            text_color = INK if digit == best else INK_SECONDARY
            weight = "bold" if digit == best else "normal"
            self.bars.create_text(0, y, text=str(digit), anchor="w", font=(FONT, 10, weight), fill=text_color)
            self.bars.create_rectangle(label_w, y - 5, label_w + track_w, y + 5, fill=GRID, width=0)
            if p * track_w >= 1:
                self.bars.create_rectangle(label_w, y - 5, label_w + p * track_w, y + 5, fill=color, width=0)
            self.bars.create_text(label_w + track_w + value_w, y, text=f"{100 * p:.1f}%", anchor="e",
                                  font=(FONT, 9, weight), fill=text_color)

    def show_preview(self, image):
        # the 28x28 input enlarged with hard pixel edges, so single pixels stay visible
        enlarged = Image.fromarray(image).resize((MNIST_SIZE * 4, MNIST_SIZE * 4), Image.NEAREST)
        self.preview_image = ImageTk.PhotoImage(enlarged)  # tkinter needs the reference kept
        self.preview_label.config(image=self.preview_image)

    def set_status(self, text, error=False):
        self.status.config(text=text, fg=WARNING if error else INK_SECONDARY)


def parse_args():
    parser = argparse.ArgumentParser(description="rysowanie cyfr myszką i rozpoznawanie ich wytrenowanym modelem")
    parser.add_argument("model", nargs="?", help="ścieżka do model.json, domyślnie ostatnia epoka najnowszego treningu "
                                                 "albo models/demo/model.json, gdy nie ma żadnego treningu")
    return parser.parse_args()


def enable_sharp_rendering():
    # on Windows with display scaling tkinter is blurry unless the process says it handles DPI itself
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except (ImportError, AttributeError, OSError):
        pass


def main():
    args = parse_args()
    enable_sharp_rendering()
    root = tk.Tk()
    DigitRecognizerApp(root, args.model or find_latest_model(MODELS_DIR) or DEMO_MODEL)
    root.mainloop()


if __name__ == "__main__":
    main()
