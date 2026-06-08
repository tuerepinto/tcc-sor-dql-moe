from pathlib import Path

def find_project_root(start: Path | None = None) -> Path:
    p = (start or Path.cwd()).resolve()
    while p != p.parent:
        if (p / "src").exists():
            return p
        p = p.parent
    raise FileNotFoundError("Não encontrei raiz do projeto (pasta com 'src').")

# Constantes de caminho (compatíveis com imports antigos/novos)
PROJECT_ROOT = find_project_root(Path(__file__).resolve())
DATA_DIR = PROJECT_ROOT / "data"
L2_DIR = DATA_DIR / "l2"
PARQUET_DIR = DATA_DIR / "l2_parquet"

def _has_venues(root: Path) -> bool:
    return root.exists() and any(root.glob("venue=*"))

def find_data_root(project_root: Path | None = None) -> Path:
    pr = project_root or PROJECT_ROOT
    candidates = [pr / "data" / "l2", pr / "data" / "l2_parquet"]

    populated = [c for c in candidates if _has_venues(c)]
    if populated:
        return populated[0]

    existing = [c for c in candidates if c.exists()]
    if existing:
        raise FileNotFoundError(
            f"Encontrei diretório de dados, mas sem 'venue=*': {existing}"
        )

    raise FileNotFoundError(f"Não encontrei dataset em: {candidates}")