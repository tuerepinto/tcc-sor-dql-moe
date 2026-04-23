import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

from src.sor_env import MultiVenueSOREnv
from src.moe_dqn import MoENetwork


def simulate_twap(env: MultiVenueSOREnv, total_inventory: float) -> Tuple[float, float, Dict]:
    """
    TWAP discreto e JUSTO:
    - usa env.step() (mesmas regras e validações do ambiente)
    - action=1 (B3) repetido é o TWAP "cego" no seu action space atual
    """
    _state, _ = env.reset()
    done = False

    total_cost = 0.0
    total_vol = 0.0
    rejects = 0

    while not done:
        _state, _reward, terminated, truncated, info = env.step(1)
        done = bool(terminated or truncated)

        total_cost += float(info["executed_cost"])
        total_vol += float(info["executed_volume"])
        if not bool(info["is_valid"]):
            rejects += 1

    avg_price = (total_cost / total_vol) if total_vol > 0 else 0.0
    return float(avg_price), float(total_vol), {"rejects": int(rejects)}


def evaluate_agent(model, env: MultiVenueSOREnv) -> Tuple[float, float, Dict]:
    state, _ = env.reset()
    done = False

    total_cost = 0.0
    total_vol = 0.0
    rejects = 0

    while not done:
        st = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q = model(st)
            if q.dim() == 3:
                q = q.mean(dim=-1)
        action = int(torch.argmax(q, dim=1).item())

        state, _reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        total_cost += float(info["executed_cost"])
        total_vol += float(info["executed_volume"])
        if not bool(info["is_valid"]):
            rejects += 1

    avg_price = (total_cost / total_vol) if total_vol > 0 else 0.0
    return float(avg_price), float(total_vol), {"rejects": int(rejects)}


if __name__ == "__main__":
    steps = 500
    df_b3 = pd.DataFrame({
        "ask_1": np.random.uniform(47.05, 47.15, steps), "vol_ask_1": np.random.randint(100, 500, steps),
        "ask_2": np.random.uniform(47.16, 47.25, steps), "vol_ask_2": np.random.randint(100, 500, steps),
        "ask_3": np.random.uniform(47.26, 47.35, steps), "vol_ask_3": np.random.randint(100, 500, steps),
        "ask_4": np.random.uniform(47.36, 47.45, steps), "vol_ask_4": np.random.randint(100, 500, steps),
        "ask_5": np.random.uniform(47.46, 47.55, steps), "vol_ask_5": np.random.randint(100, 500, steps),
    })

    df_base = pd.DataFrame({
        "ask_1": np.random.uniform(47.00, 47.10, steps), "vol_ask_1": np.random.randint(50, 300, steps),
        "ask_2": np.random.uniform(47.11, 47.20, steps), "vol_ask_2": np.random.randint(50, 300, steps),
        "ask_3": np.random.uniform(47.21, 47.30, steps), "vol_ask_3": np.random.randint(50, 300, steps),
        "ask_4": np.random.uniform(47.31, 47.40, steps), "vol_ask_4": np.random.randint(50, 300, steps),
        "ask_5": np.random.uniform(47.41, 47.50, steps), "vol_ask_5": np.random.randint(50, 300, steps),
    })

    env = MultiVenueSOREnv(df_b3, df_base, total_inventory=10000)

    twap_price, twap_vol, twap_info = simulate_twap(env, env.total_inventory)
    print(f"TWAP Volume: {twap_vol:.2f}")
    print(f"TWAP Average Price: {twap_price:.4f}")
    print(f"TWAP Rejects: {twap_info.get('rejects', 0)}")

    model_path = Path("models/moe_dqn_sor.pth")
    if model_path.exists():
        model = MoENetwork(input_dim=5, output_dim=4, num_experts=3)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

        ia_price, ia_vol, ia_info = evaluate_agent(model, env)
        print(f"IA Volume: {ia_vol:.2f}")
        print(f"IA Average Price: {ia_price:.4f}")
        print(f"IA Rejects: {ia_info.get('rejects', 0)}")

        diff = twap_price - ia_price
        if ia_price < twap_price:
            print(f"Conclusion: IA performed better (lower average price by {diff:.4f}).")
        elif ia_price > twap_price:
            print(f"Conclusion: TWAP performed better (lower average price by {-diff:.4f}).")
        else:
            print("Conclusion: Both performed equally.")
    else:
        print(f"IA model not found at {model_path}. IA evaluation skipped.")