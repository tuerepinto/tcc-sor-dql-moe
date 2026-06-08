from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

# =========================================================
# Bootstrap de path (precisa vir ANTES de importar src.*)
# =========================================================
def find_project_root(start: Path) -> Path:
    p = start.resolve()
    while p != p.parent:
        if (p / "src").exists():
            return p
        p = p.parent
    raise FileNotFoundError("Raiz do projeto não encontrada (pasta 'src').")

PROJECT_ROOT = find_project_root(Path(__file__).resolve())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# =========================================================
# Imports da nova arquitetura
# =========================================================
from src.data.paths import find_data_root
from src.data.l2_dataset import safe_dates
from src.envs.factory import make_parquet_env
from src.trainers.dqn_runner import TrainConfig, train_moe_dqn
from src.models.model_io import save_model


def save_rewards(df: pd.DataFrame, out_path_parquet: Path) -> Path:
    """
    Tenta salvar em parquet; se faltar engine (pyarrow/fastparquet), salva CSV.
    Retorna o path final salvo.
    """
    try:
        df.to_parquet(out_path_parquet, index=False)
        return out_path_parquet
    except ImportError:
        out_csv = out_path_parquet.with_suffix(".csv")
        df.to_csv(out_csv, index=False)
        print(f"[WARN] sem engine parquet; salvando CSV em {out_csv}")
        return out_csv


def attach_common_dates(env, common_dates: list[str]) -> None:
    """
    Compatibilidade com implementações diferentes de env.
    """
    if hasattr(env, "set_dates") and callable(getattr(env, "set_dates")):
        env.set_dates(common_dates)
    elif hasattr(env, "_dates"):
        env._dates = common_dates  # fallback temporário
    else:
        print("[WARN] env não expõe set_dates/_dates; seguindo sem forçar datas comuns.")


def main() -> None:
    root = PROJECT_ROOT
    data_root = find_data_root(root)

    symbols = ["PETR4", "VALE3", "ITUB4"]
    venue_b3, venue_base = "B3", "BASE"

    out_models = root / "models"
    out_logs = root / "logs"
    out_models.mkdir(parents=True, exist_ok=True)
    out_logs.mkdir(parents=True, exist_ok=True)

    print("PROJECT_ROOT =", root)
    print("DATA_ROOT    =", data_root)

    for sym in symbols:
        d_b3 = safe_dates(data_root, venue_b3, sym)
        d_base = safe_dates(data_root, venue_base, sym)
        common = sorted(set(d_b3) & set(d_base))

        if not d_b3:
            print(f"[SKIP] {sym}: sem dias em venue={venue_b3}")
            continue
        if not d_base:
            print(f"[SKIP] {sym}: sem dias em venue={venue_base}")
            continue
        if not common:
            print(f"[SKIP] {sym}: sem datas em comum")
            continue

        print(f"[OK] {sym}: B3={len(d_b3)} BASE={len(d_base)} comum={len(common)}")

        env = make_parquet_env(
            root=data_root,
            symbol=sym,
            venue_b3=venue_b3,
            venue_base=venue_base,
            episode_len=500,
            total_inventory=10_000,
            max_slippage_pct=0.001,
            seed=42,
        )
        attach_common_dates(env, common)

        model, rewards = train_moe_dqn(env, TrainConfig(seed=42))

        model_path = out_models / f"moe_dqn_sor_{sym}_12m.pth"
        save_model(model, model_path)

        rewards_df = pd.DataFrame({
            "episode": np.arange(len(rewards), dtype=int),
            "reward": rewards
        })
        rewards_path = save_rewards(rewards_df, out_logs / f"rewards_{sym}_12m.parquet")

        print(f"[SAVE] model   -> {model_path}")
        print(f"[SAVE] rewards -> {rewards_path}")


if __name__ == "__main__":
    main()