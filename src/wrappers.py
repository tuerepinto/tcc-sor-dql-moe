import gymnasium as gym
import numpy as np


class SORFeatureWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, vol_scale: float = 1000.0):
        super().__init__(env)
        self.vol_scale = float(vol_scale)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        ask_b3, vol_b3, ask_base, vol_base, inv = obs.astype(np.float32)

        arrival = float(self.env.arrival_price)
        t = float(self.env.current_step)
        T = float(max(1, len(self.env.lob_b3) - 1))

        rel_b3 = (ask_b3 / arrival) - 1.0
        rel_base = (ask_base / arrival) - 1.0
        rel_edge = rel_base - rel_b3
        time_frac = t / T
        inv_frac = inv / float(self.env.total_inventory)

        vol_b3_n = vol_b3 / self.vol_scale
        vol_base_n = vol_base / self.vol_scale

        return np.array([rel_b3, vol_b3_n, rel_base, vol_base_n, inv_frac, rel_edge, time_frac, 1.0], dtype=np.float32)