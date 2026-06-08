from src.envs.sor_env_numpy import MultiVenueSOREnvNumpy
from src.envs.sor_env_parquet import MultiVenueSOREnvParquet

def make_numpy_env(lob_b3, lob_base, total_inventory=10_000, max_slippage_pct=0.001):
    return MultiVenueSOREnvNumpy(
        lob_b3=lob_b3,
        lob_base=lob_base,
        total_inventory=total_inventory,
        max_slippage_pct=max_slippage_pct,
    )

def make_parquet_env(root, symbol, venue_b3="B3", venue_base="BASE",
                     episode_len=500, total_inventory=10_000,
                     max_slippage_pct=0.001, seed=42):
    return MultiVenueSOREnvParquet(
        root=str(root),
        symbol=symbol,
        venue_b3=venue_b3,
        venue_base=venue_base,
        episode_len=episode_len,
        total_inventory=total_inventory,
        max_slippage_pct=max_slippage_pct,
        seed=seed,
    )