from pathlib import Path
import torch

def find_model_path(models_dir: Path) -> Path:
    patterns = [
        "moe_dqn_sor_SYNTHETIC.pth",
        "moe_dqn_sor_*_12m.pth",
        "moe_dqn_sor.pth",
        "*.pth",
    ]
    for pat in patterns:
        hits = sorted(models_dir.glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"Nenhum .pth em {models_dir}")

def save_model(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)