import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class DataVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Wizualizacja danych z dane/projekt1")
        self.root.geometry("1200x800")

        self.data_dir = Path("dane/projekt1/projekt1")

        # Panel sterowania
        control_frame = ttk.Frame(root)
        control_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=10, pady=10)

        ttk.Label(control_frame, text="Typ danych:", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(10, 5))
        self.type_var = tk.StringVar(value="classification")
        ttk.Radiobutton(control_frame, text="Klasyfikacja", variable=self.type_var,
                       value="classification", command=self.update_datasets).pack(anchor=tk.W)
        ttk.Radiobutton(control_frame, text="Regresja", variable=self.type_var,
                       value="regression", command=self.update_datasets).pack(anchor=tk.W)

        ttk.Label(control_frame, text="Dataset:", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(15, 5))
        self.dataset_var = tk.StringVar()
        self.dataset_combo = ttk.Combobox(control_frame, textvariable=self.dataset_var, state="readonly")
        self.dataset_combo.pack(anchor=tk.W, fill=tk.X, pady=(0, 5))
        self.dataset_combo.bind("<<ComboboxSelected>>", lambda e: self.update_datasets_list())

        ttk.Label(control_frame, text="Rozmiar zbioru:", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(15, 5))
        self.size_var = tk.StringVar()
        self.size_combo = ttk.Combobox(control_frame, textvariable=self.size_var, state="readonly")
        self.size_combo.pack(anchor=tk.W, fill=tk.X, pady=(0, 5))
        self.size_combo.bind("<<ComboboxSelected>>", lambda e: self.plot_data())

        ttk.Label(control_frame, text="Typ zbioru:", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(15, 5))
        self.split_var = tk.StringVar(value="train")
        ttk.Radiobutton(control_frame, text="Treningowy (train)", variable=self.split_var,
                       value="train", command=self.plot_data).pack(anchor=tk.W)
        ttk.Radiobutton(control_frame, text="Testowy (test)", variable=self.split_var,
                       value="test", command=self.plot_data).pack(anchor=tk.W)

        # Informacje o danych
        ttk.Label(control_frame, text="Statystyki:", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(15, 5))
        self.info_text = tk.Text(control_frame, height=10, width=35, font=("Courier", 9))
        self.info_text.pack(anchor=tk.W, fill=tk.BOTH, expand=True, pady=(0, 10))

        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X)
        ttk.Button(button_frame, text="Odśwież", command=self.plot_data).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Zapisz wykres", command=self.save_plot).pack(side=tk.LEFT)

        # Canvas dla wykresu
        self.canvas_frame = ttk.Frame(root)
        self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Inicjalizacja
        self.update_datasets()

    def update_datasets(self):
        data_type = self.type_var.get()
        type_dir = self.data_dir / data_type

        # Zbierz unikalne nazwy datasetów
        datasets = set()
        if type_dir.exists():
            for file in type_dir.glob("*.csv"):
                # Wyodrębnij nazwę datasetu (część przed .train lub .test)
                name = file.stem.rsplit('.train.', 1)[0].rsplit('.test.', 1)[0]
                datasets.add(name)

        datasets = sorted(list(datasets))
        self.dataset_combo['values'] = datasets
        if datasets:
            self.dataset_combo.current(0)
            self.update_datasets_list()

    def update_datasets_list(self):
        data_type = self.type_var.get()
        dataset = self.dataset_var.get()
        type_dir = self.data_dir / data_type

        # Zbierz rozmiary dla tego datasetu
        sizes = set()
        if type_dir.exists():
            for file in type_dir.glob(f"{dataset}.*.csv"):
                # Wyodrębnij rozmiar (ostatnia liczba przed .csv)
                parts = file.stem.split('.')
                if parts[-1].isdigit():
                    sizes.add(parts[-1])

        sizes = sorted(list(sizes), key=int)
        self.size_combo['values'] = sizes
        if sizes:
            self.size_combo.current(0)
            self.plot_data()

    def plot_data(self):
        data_type = self.type_var.get()
        dataset = self.dataset_var.get()
        size = self.size_var.get()
        split = self.split_var.get()

        if not dataset or not size:
            return

        # Konstruuj ścieżkę do pliku
        file_path = self.data_dir / data_type / f"{dataset}.{split}.{size}.csv"

        if not file_path.exists():
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, f"Plik nie znaleziony:\n{file_path}")
            return

        # Wczytaj dane
        try:
            df = pd.read_csv(file_path)

            # Wyczyść poprzedni canvas
            for widget in self.canvas_frame.winfo_children():
                widget.destroy()

            # Stwórz figurę
            fig, ax = plt.subplots(figsize=(8, 7), dpi=100)

            # Wykreśl dane
            if data_type == "classification":
                # Klasyfikacja - koloruj po klasach
                classes = df['cls'].unique()
                colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))

                for idx, cls in enumerate(sorted(classes)):
                    mask = df['cls'] == cls
                    ax.scatter(df[mask]['x'], df[mask]['y'],
                             c=[colors[idx]], label=f'Klasa {int(cls)}',
                             s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
                ax.set_title(f"Dane klasyfikacyjne: {dataset}\n({split}, {size} próbek)", fontsize=14, fontweight='bold')
            else:
                # Regresja - koloruj po wartościach
                scatter = ax.scatter(df['x'], df['y'], c=df['y'],
                                   cmap='viridis', s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
                plt.colorbar(scatter, ax=ax, label='Wartość wyjścia')
                ax.set_title(f"Dane regresyjne: {dataset}\n({split}, {size} próbek)", fontsize=14, fontweight='bold')

            ax.set_xlabel('x', fontsize=12)
            ax.set_ylabel('y', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')

            # Umieść canvas
            canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Statystyki
            self.info_text.delete(1.0, tk.END)
            info = f"Dataset: {dataset}\n"
            info += f"Typ: {data_type}\n"
            info += f"Rozmiar: {size} próbek\n"
            info += f"Zbiór: {split}\n\n"
            info += f"Statystyki X:\n"
            info += f"  Min: {df['x'].min():.4f}\n"
            info += f"  Max: {df['x'].max():.4f}\n"
            info += f"  Średnia: {df['x'].mean():.4f}\n"
            info += f"  Std Dev: {df['x'].std():.4f}\n\n"
            info += f"Statystyki Y:\n"
            info += f"  Min: {df['y'].min():.4f}\n"
            info += f"  Max: {df['y'].max():.4f}\n"
            info += f"  Średnia: {df['y'].mean():.4f}\n"
            info += f"  Std Dev: {df['y'].std():.4f}\n"

            if data_type == "classification":
                info += f"\nKlasy: {sorted(df['cls'].unique())}\n"
                for cls in sorted(df['cls'].unique()):
                    count = (df['cls'] == cls).sum()
                    info += f"  Klasa {int(cls)}: {count} próbek\n"

            self.info_text.insert(tk.END, info)

        except Exception as e:
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, f"Błąd przy wczytywaniu:\n{str(e)}")

    def save_plot(self):
        data_type = self.type_var.get()
        dataset = self.dataset_var.get()
        size = self.size_var.get()
        split = self.split_var.get()

        if not dataset or not size:
            return

        filename = f"{dataset}_{split}_{size}.png"
        path = filedialog.asksaveasfilename(defaultextension=".png", initialfile=filename,
                                           filetypes=[("PNG files", "*.png"), ("All files", "*.*")])

        if path:
            # Pobierz aktualną figurę i zapisz
            canvas_widget = self.canvas_frame.winfo_children()
            if canvas_widget:
                for widget in canvas_widget:
                    if hasattr(widget, 'figure'):
                        widget.figure.savefig(path, dpi=150, bbox_inches='tight')
                        print(f"Zapisano: {path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DataVisualizer(root)
    root.mainloop()
