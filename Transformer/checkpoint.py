import os
import shutil
from datetime import datetime


def timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def make_ckpt_name(model, step):
    cls = model.__class__.__name__
    return f"{cls}_step_{step}_{timestamp()}.pt"


def save_checkpoint(model, runtime_dir, drive_dir, step, mode):
    ensure_dirs(drive_dir)

    fname = make_ckpt_name(model, step)

    drive_path = os.path.join(drive_dir, fname)

    print(f"[Checkpoint] step={step}")

    if mode == "colab":
        ensure_dirs(runtime_dir)
        runtime_path = os.path.join(runtime_dir, fname)
        model.save(runtime_path)
        shutil.copy(runtime_path, drive_path)
        print(f" saved → {runtime_path}")
    else:
        model.save(drive_path)

    print(f" copied → {drive_path}")


def find_latest_checkpoint(path):
    if not os.path.exists(path):
        return None

    files = [f for f in os.listdir(path) if f.endswith(".pt")]
    if not files:
        return None

    files.sort()
    return os.path.join(path, files[-1])
