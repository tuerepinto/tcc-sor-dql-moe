from __future__ import annotations

from typing import Any, Dict, Tuple
import torch

def simulate_twap(env: Any, total_inventory: float) -> Tuple[float, float, Dict]:
    state, _ = env.reset()
    done = False

    total_cost = 0.0
    total_vol = 0.0
    rejects = 0
    steps = 0
    last_info = None

    while not done:
        state, reward, terminated, truncated, info = env.step(1)
        done = bool(terminated or truncated)

        steps += 1
        last_info = info
        total_cost += float(info["executed_cost"])
        total_vol  += float(info["executed_volume"])
        rejects += int(not bool(info["is_valid"]))

    avg_price = (total_cost / total_vol) if total_vol > 0 else 0.0
    arrival = float(last_info["arrival_price"]) if last_info else float(getattr(env, "arrival_price", 0.0))

    return float(avg_price), float(total_vol), {
        "rejects": int(rejects),
        "steps": int(steps),
        "arrival_price": float(arrival),
        "avg_price": float(avg_price),
    }

def evaluate_agent(model: torch.nn.Module, env: Any) -> Tuple[float, float, Dict]:
    state, _ = env.reset()
    done = False

    total_cost = 0.0
    total_vol = 0.0
    rejects = 0
    steps = 0
    last_info = None

    device = next(model.parameters()).device
    model.eval()

    while not done:
        st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q = model(st)
        action = int(torch.argmax(q, dim=1).item())

        state, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        steps += 1
        last_info = info
        total_cost += float(info["executed_cost"])
        total_vol  += float(info["executed_volume"])
        rejects += int(not bool(info["is_valid"]))

    avg_price = (total_cost / total_vol) if total_vol > 0 else 0.0
    arrival = float(last_info["arrival_price"]) if last_info else float(getattr(env, "arrival_price", 0.0))

    return float(avg_price), float(total_vol), {
        "rejects": int(rejects),
        "steps": int(steps),
        "arrival_price": float(arrival),
        "avg_price": float(avg_price),
    }