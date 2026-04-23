import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
from src.sor_env import MultiVenueSOREnv
from src.evaluate_baselines import simulate_twap, evaluate_agent
from src.moe_dqn import MoENetwork

# Setup test data
np.random.seed(42)
steps = 100
df_b3 = pd.DataFrame({
    'ask_1': np.random.uniform(47.05, 47.15, steps), 'vol_ask_1': np.random.randint(100, 500, steps),
    'ask_2': np.random.uniform(47.16, 47.25, steps), 'vol_ask_2': np.random.randint(100, 500, steps),
    'ask_3': np.random.uniform(47.26, 47.35, steps), 'vol_ask_3': np.random.randint(100, 500, steps),
    'ask_4': np.random.uniform(47.36, 47.45, steps), 'vol_ask_4': np.random.randint(100, 500, steps),
    'ask_5': np.random.uniform(47.46, 47.55, steps), 'vol_ask_5': np.random.randint(100, 500, steps),
})

df_base = pd.DataFrame({
    'ask_1': np.random.uniform(47.00, 47.10, steps), 'vol_ask_1': np.random.randint(50, 300, steps),
    'ask_2': np.random.uniform(47.11, 47.20, steps), 'vol_ask_2': np.random.randint(50, 300, steps),
    'ask_3': np.random.uniform(47.21, 47.30, steps), 'vol_ask_3': np.random.randint(50, 300, steps),
    'ask_4': np.random.uniform(47.31, 47.40, steps), 'vol_ask_4': np.random.randint(50, 300, steps),
    'ask_5': np.random.uniform(47.41, 47.50, steps), 'vol_ask_5': np.random.randint(50, 300, steps),
})

env = MultiVenueSOREnv(lob_b3=df_b3, lob_base=df_base, total_inventory=1000)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

ia_evaluated = False
if os.path.exists("moe_dqn_model.pth"):
    try:
        model = MoENetwork(input_dim=state_dim, output_dim=action_dim, num_experts=3)
        model.load_state_dict(torch.load("moe_dqn_model.pth", weights_only=True))
        model.eval()
        ia_price, ia_vol, ia_info = evaluate_agent(model, env)
        print(f"IA Volume: {ia_vol:.2f}")
        print(f"IA Average Price: {ia_price:.4f}")
        print(f"IA Rejects: {ia_info.get('rejects', 0)}")
        ia_evaluated = True
    except Exception as e:
        print(f"Could not load IA model: {e}")
else:
    print("IA Model (moe_dqn_model.pth) not found.")

twap_price, twap_vol, twap_info = simulate_twap(env, 1000)
print(f"TWAP Volume: {twap_vol:.2f}")
print(f"TWAP Average Price: {twap_price:.4f}")
print(f"TWAP Rejects: {twap_info.get('rejects', 0)}")

if ia_evaluated:
    diff = twap_price - ia_price
    if ia_price < twap_price:
        print(f"Conclusion: IA performed better (lower average price by {diff:.4f}).")
    elif ia_price > twap_price:
        print(f"Conclusion: TWAP performed better (lower average price by {-diff:.4f}).")
    else:
        print("Conclusion: Both performed equally.")
else:
    print("Conclusion: IA evaluation skipped.")
