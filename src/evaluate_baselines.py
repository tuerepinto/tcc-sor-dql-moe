import torch
import numpy as np
import pandas as pd
from src.sor_env import MultiVenueSOREnv
from src.moe_dqn import MoENetwork

def simulate_twap(env, total_inventory):
    """
    Simula o algoritmo TWAP (Time-Weighted Average Price).
    Divide a ordem em fatias iguais ao longo do tempo, roteando apenas para a B3.
    """
    env.reset()
    total_steps = len(env.lob_b3) - 1
    volume_per_step = total_inventory // total_steps

    inventory = total_inventory
    total_cost = 0.0
    vol_executado_total = 0

    for step in range(total_steps):
        if inventory <= 0:
            break

        # TWAP tradicional é cego para fragmentação, roteia para a bolsa principal (B3)
        row_b3 = env.lob_b3.iloc[step]

        vol_to_execute = min(volume_per_step, inventory)
        custo_step, vol_exec = env._execute_order(vol_to_execute, row_b3)

        total_cost += custo_step
        vol_executado_total += vol_exec
        inventory -= vol_exec

    preco_medio_twap = total_cost / vol_executado_total if vol_executado_total > 0 else 0
    return preco_medio_twap, vol_executado_total

def evaluate_agent(model, env):
    """
    Simula a execução usando a IA treinada (MoE-DQN).
    """
    state, _ = env.reset()
    done = False

    inventory_inicial = env.total_inventory
    arrival_price = env.arrival_price

    # Vamos rastrear o custo total deduzindo da recompensa (Implementation Shortfall)
    # Recompensa = -(Preço_Medio - Arrival_Price) * Volume
    total_shortfall_cost = 0.0 

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = model(state_tensor)

        # Escolhe a melhor ação (Explotação pura, sem Epsilon)
        action = torch.argmax(q_values).item()

        next_state, reward, done, _, _ = env.step(action)

        # Acumula apenas as recompensas de execução (ignorando penalidades de tempo esgotado para o cálculo do preço)
        if reward < 0 and env.inventory_remaining > 0: 
            total_shortfall_cost += abs(reward)

        state = next_state

    vol_executado = inventory_inicial - env.inventory_remaining

    # Calcula o preço médio real que a IA conseguiu
    if vol_executado > 0:
        # Shortfall Total = (Preço Medio IA - Arrival Price) * Vol
        # Preço Medio IA = (Shortfall Total / Vol) + Arrival Price
        preco_medio_ia = (total_shortfall_cost / vol_executado) + arrival_price
    else:
        preco_medio_ia = 0.0

    return preco_medio_ia, vol_executado

if __name__ == "__main__":
    print("--- Iniciando Avaliação de Baselines (Capítulo 8) ---")

    # 1. Gerando Dados Mockados (Substitua pelos seus DataFrames reais de PETR4)
    steps = 500
    df_b3 = pd.DataFrame({
        'ask_1': np.random.uniform(47.00, 47.10, steps),
        'vol_ask_1': np.random.randint(100, 500, steps),
        'ask_2': np.random.uniform(47.11, 47.20, steps), 'vol_ask_2': np.random.randint(100, 500, steps),
        'ask_3': np.random.uniform(47.21, 47.30, steps), 'vol_ask_3': np.random.randint(100, 500, steps),
        'ask_4': np.random.uniform(47.31, 47.40, steps), 'vol_ask_4': np.random.randint(100, 500, steps),
        'ask_5': np.random.uniform(47.41, 47.50, steps), 'vol_ask_5': np.random.randint(100, 500, steps),
    })

    df_base = pd.DataFrame({
        'ask_1': np.random.uniform(46.98, 47.12, steps), # Base Exchange levemente diferente
        'vol_ask_1': np.random.randint(50, 300, steps),
        'ask_2': np.random.uniform(47.13, 47.22, steps), 'vol_ask_2': np.random.randint(50, 300, steps),
        'ask_3': np.random.uniform(47.23, 47.32, steps), 'vol_ask_3': np.random.randint(50, 300, steps),
        'ask_4': np.random.uniform(47.33, 47.42, steps), 'vol_ask_4': np.random.randint(50, 300, steps),
        'ask_5': np.random.uniform(47.43, 47.52, steps), 'vol_ask_5': np.random.randint(50, 300, steps),
    })

    TOTAL_ORDER = 10000
    env = MultiVenueSOREnv(df_b3, df_base, total_inventory=TOTAL_ORDER)

    # 2. Rodando o TWAP
    print("\nExecutando Baseline: TWAP...")
    preco_twap, vol_twap = simulate_twap(env, TOTAL_ORDER)
    print(f"TWAP finalizado. Volume: {vol_twap} | Preço Médio: R$ {preco_twap:.4f}")

    # 3. Rodando a IA (Carregando o modelo treinado)
    print("\nExecutando Inteligência Artificial (MoE-DQN)...")
    model_ia = MoEDQN(state_dim=5, action_dim=4, num_experts=3)
    # NOTA: Aqui você usaria model_ia.load_state_dict(torch.load('modelo_treinado.pth'))

    preco_ia, vol_ia = evaluate_agent(model_ia, env)
    print(f"IA finalizada. Volume: {vol_ia} | Preço Médio: R$ {preco_ia:.4f}")

    # 4. Resultados para o TCC
    print("\n" + "="*40)
    print("RESULTADOS DA COMPARAÇÃO (Slippage)")
    print("="*40)
    economia_por_acao = preco_twap - preco_ia
    economia_total = economia_por_acao * TOTAL_ORDER

    if economia_total > 0:
        print(f"A IA superou o TWAP!")
        print(f"Economia de R$ {economia_por_acao:.4f} por ação.")
        print(f"Economia Total na Ordem Institucional: R$ {economia_total:.2f}")
    else:
        print("O TWAP foi melhor ou igual (O modelo IA precisa de mais épocas de treinamento).")