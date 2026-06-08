from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
import pandas as pd
import torch

from src.models.model_io import find_model_path
from src.models.moe_network import MoENetwork
from src.envs.factory import make_numpy_env
from src.data.offline_dataset import df_to_l2_numpy
from src.eval.replay import run_and_log_twap, run_and_log_agent, summarize_log

def main():
    root = Path.cwd()
    models_dir = root / "models"
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # exemplo: replay sintético
    # (trocar por dataset real/snapshot conforme sua estratégia)
    import numpy as np
    steps = 200
    np.random.seed(99)
    df_b3 = pd.DataFrame({"ask_1": np.random.uniform(47.05, 47.15, steps), "vol_ask_1": np.random.randint(100, 500, steps),
                          "ask_2": np.random.uniform(47.16, 47.25, steps), "vol_ask_2": np.random.randint(100, 500, steps),
                          "ask_3": np.random.uniform(47.26, 47.35, steps), "vol_ask_3": np.random.randint(100, 500, steps),
                          "ask_4": np.random.uniform(47.36, 47.45, steps), "vol_ask_4": np.random.randint(100, 500, steps),
                          "ask_5": np.random.uniform(47.46, 47.55, steps), "vol_ask_5": np.random.randint(100, 500, steps)})
    df_base = pd.DataFrame({"ask_1": np.random.uniform(47.00, 47.10, steps), "vol_ask_1": np.random.randint(50, 300, steps),
                            "ask_2": np.random.uniform(47.11, 47.20, steps), "vol_ask_2": np.random.randint(50, 300, steps),
                            "ask_3": np.random.uniform(47.21, 47.30, steps), "vol_ask_3": np.random.randint(50, 300, steps),
                            "ask_4": np.random.uniform(47.31, 47.40, steps), "vol_ask_4": np.random.randint(50, 300, steps),
                            "ask_5": np.random.uniform(47.41, 47.50, steps), "vol_ask_5": np.random.randint(50, 300, steps)})

    env_t = make_numpy_env(df_to_l2_numpy(df_b3), df_to_l2_numpy(df_base), total_inventory=10_000)
    env_i = make_numpy_env(df_to_l2_numpy(df_b3), df_to_l2_numpy(df_base), total_inventory=10_000)

    model_path = find_model_path(models_dir)
    model = MoENetwork(input_dim=5, output_dim=4, num_experts=3)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    log_t = run_and_log_twap(env_t)
    log_i = run_and_log_agent(env_i, model)

    s_t = summarize_log(log_t, total_inventory=10_000)
    s_i = summarize_log(log_i, total_inventory=10_000)

    log_t.to_parquet(logs_dir / "eval_twap.parquet", index=False)
    log_i.to_parquet(logs_dir / "eval_ia.parquet", index=False)

    print("TWAP:", s_t)
    print("IA  :", s_i)
    print("Economia (R$):", s_t["total_cost"] - s_i["total_cost"])

if __name__ == "__main__":
    main()