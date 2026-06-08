from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

# =========================================================
# Bootstrap de path (garante import de src.* ao rodar via scripts/)
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# =========================================================
# Imports da arquitetura nova
# =========================================================
from src.data.paths import find_project_root
from src.data.offline_dataset import df_to_l2_numpy
from src.envs.factory import make_numpy_env
from src.trainers.dqn_runner import TrainConfig, train_moe_dqn
from src.models.model_io import save_model


def make_synth(steps: int = 100, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    df_b3 = pd.DataFrame({
        "ask_1": rng.uniform(35.00, 35.10, steps), "vol_ask_1": rng.integers(100, 1000, steps),
        "ask_2": rng.uniform(35.11, 35.20, steps), "vol_ask_2": rng.integers(100, 1000, steps),
        "ask_3": rng.uniform(35.21, 35.30, steps), "vol_ask_3": rng.integers(100, 1000, steps),
        "ask_4": rng.uniform(35.31, 35.40, steps), "vol_ask_4": rng.integers(100, 1000, steps),
        "ask_5": rng.uniform(35.41, 35.50, steps), "vol_ask_5": rng.integers(100, 1000, steps),
    })

    df_base = pd.DataFrame({
        "ask_1": rng.uniform(34.98, 35.08, steps), "vol_ask_1": rng.integers(50, 800, steps),
        "ask_2": rng.uniform(35.09, 35.18, steps), "vol_ask_2": rng.integers(50, 800, steps),
        "ask_3": rng.uniform(35.19, 35.28, steps), "vol_ask_3": rng.integers(50, 800, steps),
        "ask_4": rng.uniform(35.29, 35.38, steps), "vol_ask_4": rng.integers(50, 800, steps),
        "ask_5": rng.uniform(35.39, 35.48, steps), "vol_ask_5": rng.integers(50, 800, steps),
    })

    return df_b3, df_base


def save_rewards_with_fallback(rewards: list[float], out_logs: Path) -> Path:
    df = pd.DataFrame({
        "episode": np.arange(len(rewards), dtype=int),
        "reward": rewards
    })

    parquet_path = out_logs / "rewards_synthetic.parquet"
    try:
        df.to_parquet(parquet_path, index=False)
        print(f"[OK] rewards salvos em {parquet_path}")
        return parquet_path
    except ImportError:
        csv_path = out_logs / "rewards_synthetic.csv"
        df.to_csv(csv_path, index=False)
        print(f"[WARN] sem engine parquet; salvando CSV em {csv_path}")
        return csv_path


def main() -> None:
    root = find_project_root(Path.cwd())

    out_models = root / "models"
    out_logs = root / "logs"
    out_models.mkdir(parents=True, exist_ok=True)
    out_logs.mkdir(parents=True, exist_ok=True)

    # 1) gera dados sintéticos
    df_b3, df_base = make_synth(steps=100, seed=42)

    # 2) ambiente numpy
    env = make_numpy_env(
        df_to_l2_numpy(df_b3),
        df_to_l2_numpy(df_base),
        total_inventory=5_000,
        max_slippage_pct=0.001,
    )

    # 3) treino
    model, rewards = train_moe_dqn(env, TrainConfig(seed=42))

    # 4) salva modelo
    model_path = out_models / "moe_dqn_sor_SYNTHETIC.pth"
    save_model(model, model_path)
    print(f"[OK] modelo salvo em {model_path}")

    # 5) salva rewards (parquet com fallback csv)
    save_rewards_with_fallback(rewards, out_logs)

    print("[OK] treinamento sintético finalizado")


if __name__ == "__main__":
    main()