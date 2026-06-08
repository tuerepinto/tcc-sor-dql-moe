import numpy as np
import pandas as pd
import torch

def summarize_log(df: pd.DataFrame, total_inventory: float):
    total_cost = float(df["executed_cost"].fillna(0).sum())
    total_vol = float(df["executed_volume"].fillna(0).sum())
    avg_price = (total_cost / total_vol) if total_vol > 0 else 0.0
    arrival = float(df["arrival_price"].dropna().iloc[0]) if df["arrival_price"].notna().any() else np.nan
    rejects = int((~df["is_valid"].astype(bool)).sum()) if "is_valid" in df else 0
    return {
        "arrival_price": arrival,
        "avg_price": avg_price,
        "total_cost": total_cost,
        "total_vol": total_vol,
        "fill_rate": total_vol / float(total_inventory) if total_inventory > 0 else np.nan,
        "rejects": rejects,
        "steps": int(df.shape[0]),
    }

def run_and_log_twap(env):
    rows = []
    state, _ = env.reset()
    done = False
    step = 0
    while not done:
        action = 1
        state, reward, term, trunc, info = env.step(action)
        done = bool(term or trunc)
        rows.append({
            "step": step, "action": action, "reward": float(reward),
            **{k: info.get(k, None) for k in [
                "inventory_left","arrival_price","executed_volume","executed_cost",
                "avg_price","slippage","is_valid","rejection_reason","t","T"
            ]}
        })
        step += 1
    return pd.DataFrame(rows)

def run_and_log_agent(env, model):
    rows = []
    state, _ = env.reset()
    done = False
    step = 0
    while not done:
        st = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q = model(st)
            action = int(torch.argmax(q, dim=1).item())
            gate = torch.softmax(model.gating_network(st), dim=-1).squeeze(0).cpu().numpy()

        state, reward, term, trunc, info = env.step(action)
        done = bool(term or trunc)
        rows.append({
            "step": step, "action": action, "reward": float(reward),
            "gate_0": float(gate[0]), "gate_1": float(gate[1]), "gate_2": float(gate[2]),
            **{k: info.get(k, None) for k in [
                "inventory_left","arrival_price","executed_volume","executed_cost",
                "avg_price","slippage","is_valid","rejection_reason","t","T"
            ]}
        })
        step += 1
    return pd.DataFrame(rows)