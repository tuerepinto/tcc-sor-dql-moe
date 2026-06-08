from dataclasses import dataclass
from src.models.moe_network import MoENetwork
from src.train_agent import train_dqn

@dataclass
class TrainConfig:
    seed: int = 42
    input_dim: int = 5
    output_dim: int = 4
    num_experts: int = 3

def train_moe_dqn(env, cfg: TrainConfig):
    model = MoENetwork(
        input_dim=cfg.input_dim,
        output_dim=cfg.output_dim,
        num_experts=cfg.num_experts,
    )
    model_trained, rewards = train_dqn(env, model, seed=cfg.seed)
    return model_trained, rewards