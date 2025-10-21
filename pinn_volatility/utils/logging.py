
from pathlib import Path
def ensure_dir(d):
    p = Path(d)
    p.mkdir(parents=True, exist_ok=True)
    return p
def save_checkpoint(model, path: str):
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
