import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
from collections import deque
import pandas as pd

# Importando as classes do seu projeto
from src.sor_env import MultiVenueSOREnv
from src.moe_dqn import MoENetwork

# --- HIPERPARÂMETROS DO TREINAMENTO ---
EPISODES = 500          # Quantas vezes o agente vai operar a ordem completa
BATCH_SIZE = 64         # Tamanho do lote de memórias para aprender por vez
GAMMA = 0.99            # Fator de desconto (foco no longo prazo)
EPSILON_START = 1.0     # Começa 100% aleatório (Exploração)
EPSILON_END = 0.01      # Termina com 1% de aleatoriedade
EPSILON_DECAY = 0.995   # Taxa de decaimento da exploração
LEARNING_RATE = 1e-3    # Taxa de aprendizado da rede neural
MEMORY_SIZE = 10000     # Capacidade máxima da memória de replay
TARGET_UPDATE_EVERY = 500  # steps (não episódios)
GRAD_CLIP_NORM = 10.0

def train_dqn(env, model):
    """
    Loop principal de treinamento por Aprendizado por Reforço.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    target_model = copy.deepcopy(model).to(device)
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.SmoothL1Loss()  # Huber
    memory = deque(maxlen=MEMORY_SIZE)

    epsilon = EPSILON_START
    historico_recompensas = []
    global_step = 0

    print("Iniciando o treinamento do Smart Order Router (MoE-DQN)...")

    for episode in range(EPISODES):
        estado, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            global_step += 1

            # 1. POLÍTICA EPSILON-GREEDY (Exploração vs Explotação)
            if random.random() < epsilon:
                acao = env.action_space.sample() # Tenta algo novo (aleatório)
            else:
                estado_tensor = torch.tensor(estado, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    q_values = model(estado_tensor)
                acao = int(torch.argmax(q_values, dim=1).item()) # Usa a inteligência da rede

            # 2. INTERAÇÃO COM O AMBIENTE
            proximo_estado, recompensa, terminado, truncado, _ = env.step(acao)
            done = bool(terminado or truncado)

            # 3. ARMAZENAMENTO NA MEMÓRIA (Experience Replay)
            memory.append((estado, acao, recompensa, proximo_estado, done))
            estado = proximo_estado
            total_reward += float(recompensa)

            # 4. APRENDIZADO (Atualização dos Pesos da Rede)
            if len(memory) >= BATCH_SIZE:
                # Pega uma amostra aleatória do passado
                batch = random.sample(memory, BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)

                # Converte para tensores do PyTorch
                states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
                actions = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
                next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
                dones_t = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

                # Calcula os Q-Values atuais e perda auxiliar de balanceamento do gate
                q_values, aux_loss, aux_info = model(states, return_aux=True)

                # Calcula o Q-Value alvo (Double DQN)
                with torch.no_grad():
                    next_actions = torch.argmax(model(next_states), dim=1, keepdim=True)
                    next_q = target_model(next_states).gather(1, next_actions)
                    target_q = rewards + (GAMMA * next_q * (1.0 - dones_t))

                # Calcula o erro principal + regularização auxiliar do MoE
                q_loss = loss_fn(q_values.gather(1, actions), target_q)
                loss = q_loss + 0.01 * aux_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()

                # Atualização periódica da rede alvo
                if global_step % TARGET_UPDATE_EVERY == 0:
                    target_model.load_state_dict(model.state_dict())
                    target_model.eval()

        # 5. DECAIMENTO DA EXPLORAÇÃO
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        historico_recompensas.append(total_reward)

        # Log de progresso a cada 50 episódios
        if (episode + 1) % 50 == 0:
            print(f"Episódio {episode+1}/{EPISODES} | Recompensa Total (Custo): {total_reward:.2f} | Epsilon: {epsilon:.2f}")

    print("Treinamento concluído!")
    return model, historico_recompensas

# --- BLOCO DE EXECUÇÃO ---
if __name__ == "__main__":
    # Assumindo que df_b3 e df_base já foram carregados (como no seu notebook)
    # env = MultiVenueSOREnv(lob_b3=df_b3, lob_base=df_base, total_inventory=5000)
    # agente_moe = MoENetwork(input_dim=5, output_dim=4, num_experts=3)

    # modelo_treinado, recompensas = train_dqn(env, agente_moe)

    # Salva os pesos da rede treinada para usar na avaliação depois
    # torch.save(modelo_treinado.state_dict(), 'modelo_sor_treinado.pth')
    pass