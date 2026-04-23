from __future__ import annotations

import copy
import os
import random
from collections import deque
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from src.qr_loss import quantile_huber_loss
except ModuleNotFoundError:
    # Suporta execucao direta: `python src/train_agent.py`.
    from qr_loss import quantile_huber_loss


# --- HIPERPARÂMETROS DO TREINAMENTO ---
EPISODES = 500
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 1e-3
MEMORY_SIZE = 10_000
TARGET_UPDATE_EVERY = 500  # steps
GRAD_CLIP_NORM = 10.0

AUX_WEIGHT = 0.01  # peso do aux_loss do MoE


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _model_forward(
    model: nn.Module, x: torch.Tensor, return_aux: bool
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    out = model(x, return_aux=return_aux)

    if return_aux:
        if not (isinstance(out, tuple) and len(out) == 3):
            raise ValueError("Modelo com return_aux=True deve retornar (q_values, aux_loss, aux_info).")
        q_values, aux_loss, aux_info = out
        return q_values, aux_loss, aux_info

    if isinstance(out, tuple):
        return out[0], torch.zeros((), device=x.device), {}
    return out, torch.zeros((), device=x.device), {}


def _action_values(q: torch.Tensor) -> torch.Tensor:
    """(B,A) -> (B,A); (B,A,NQ) -> mean nos quantis -> (B,A)"""
    return q.mean(dim=-1) if q.dim() == 3 else q


def train_dqn(env, model, seed: int | None = None):
    if seed is not None:
        seed_everything(seed)
        env.reset(seed=seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).train()

    target_model = copy.deepcopy(model).to(device).eval()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.SmoothL1Loss()
    memory = deque(maxlen=MEMORY_SIZE)

    epsilon = EPSILON_START
    historico_recompensas = []
    global_step = 0

    for episode in range(EPISODES):
        estado, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            global_step += 1

            # ação
            if random.random() < epsilon:
                acao = env.action_space.sample()
            else:
                estado_tensor = torch.tensor(estado, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    q_values, _, _ = _model_forward(model, estado_tensor, return_aux=False)
                    q_values = _action_values(q_values)
                acao = int(torch.argmax(q_values, dim=1).item())

            proximo_estado, recompensa, terminado, truncado, _info = env.step(acao)
            done = bool(terminado or truncado)

            memory.append((estado, acao, float(recompensa), proximo_estado, done))
            estado = proximo_estado
            total_reward += float(recompensa)

            if len(memory) >= BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)

                states_t = torch.tensor(np.array(states), dtype=torch.float32, device=device)
                actions_t = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
                rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
                next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
                dones_t = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

                # Q(s,a) + aux MoE
                q_all, aux_loss, _aux_info = _model_forward(model, states_t, return_aux=True)
                q_sa = q_all.gather(1, actions_t)

                # Double DQN target
                with torch.no_grad():
                    online_next, _, _ = _model_forward(model, next_states_t, return_aux=False)
                    target_next, _, _ = _model_forward(target_model, next_states_t, return_aux=False)

                    next_actions = torch.argmax(online_next, dim=1, keepdim=True)
                    next_q = target_next.gather(1, next_actions)

                    target_q = rewards_t + (GAMMA * next_q * (1.0 - dones_t))

                q_loss = loss_fn(q_sa, target_q)
                loss = q_loss + AUX_WEIGHT * aux_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()

                if global_step % TARGET_UPDATE_EVERY == 0:
                    target_model.load_state_dict(model.state_dict())
                    target_model.eval()

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        historico_recompensas.append(total_reward)

    return model, historico_recompensas


def train_qr_dqn(env, model, n_quantiles: int = 51, kappa: float = 1.0, seed: int | None = None):
    if seed is not None:
        seed_everything(seed)
        env.reset(seed=seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).train()
    target_model = copy.deepcopy(model).to(device).eval()

    # sanity check: QR precisa (B,A,NQ)
    s0, _ = env.reset()
    x0 = torch.tensor(s0, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q0, _, _ = _model_forward(model, x0, return_aux=False)
    if q0.ndim != 3 or q0.size(-1) != n_quantiles:
        raise ValueError(f"QR-DQN requer saída (B,A,NQ={n_quantiles}). Recebido: {tuple(q0.shape)}")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    memory = deque(maxlen=MEMORY_SIZE)

    taus = (torch.arange(n_quantiles, device=device, dtype=torch.float32) + 0.5) / n_quantiles

    epsilon = EPSILON_START
    historico_recompensas = []
    global_step = 0

    for episode in range(EPISODES):
        estado, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            global_step += 1

            if random.random() < epsilon:
                acao = env.action_space.sample()
            else:
                st = torch.tensor(estado, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    q_dist, _, _ = _model_forward(model, st, return_aux=False)  # (1,A,NQ)
                    q_mean = q_dist.mean(dim=-1)  # (1,A)
                acao = int(torch.argmax(q_mean, dim=1).item())

            proximo_estado, recompensa, terminado, truncado, _info = env.step(acao)
            done = bool(terminado or truncado)

            memory.append((estado, acao, float(recompensa), proximo_estado, done))
            estado = proximo_estado
            total_reward += float(recompensa)

            if len(memory) >= BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)

                states_t = torch.tensor(np.array(states), dtype=torch.float32, device=device)
                next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
                actions_t = torch.tensor(actions, dtype=torch.int64, device=device)  # (B,)
                rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)  # (B,1)
                dones_t = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)      # (B,1)

                q_all, aux_loss, _aux_info = _model_forward(model, states_t, return_aux=True)  # (B,A,NQ)
                q_pred = q_all[torch.arange(BATCH_SIZE, device=device), actions_t, :]          # (B,NQ)

                with torch.no_grad():
                    online_next, _, _ = _model_forward(model, next_states_t, return_aux=False)  # (B,A,NQ)
                    next_actions = torch.argmax(online_next.mean(dim=-1), dim=1)               # (B,)

                    target_next, _, _ = _model_forward(target_model, next_states_t, return_aux=False)
                    next_q = target_next[torch.arange(BATCH_SIZE, device=device), next_actions, :]  # (B,NQ)

                    target_q = rewards_t + (GAMMA * next_q * (1.0 - dones_t))  # (B,NQ) broadcast

                loss_qr = quantile_huber_loss(q_pred, target_q, taus, kappa=kappa)
                loss = loss_qr + AUX_WEIGHT * aux_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()

                if global_step % TARGET_UPDATE_EVERY == 0:
                    target_model.load_state_dict(model.state_dict())
                    target_model.eval()

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        historico_recompensas.append(total_reward)

    return model, historico_recompensas


def train_agent(env, model, method: str = "dqn", **kwargs):
    method = method.lower().strip()
    if method == "dqn":
        return train_dqn(env, model, **kwargs)
    if method in {"qr", "qrdqn", "qr_dqn"}:
        return train_qr_dqn(env, model, **kwargs)
    raise ValueError(f"method inválido: {method}")