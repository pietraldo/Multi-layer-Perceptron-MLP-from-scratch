import os
from datetime import datetime

RESULTS_DIR = "wyniki"


def new_run_dir(config):
    # every run gets its own folder wyniki/<date_time>_<config>, all of its files go there and nowhere else
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(RESULTS_DIR, f"{stamp}_{config}")
    os.makedirs(path, exist_ok=True)
    return path
