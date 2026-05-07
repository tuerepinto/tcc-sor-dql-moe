from pathlib import Path
import torch

from src.sor_env_parquet import MultiVenueSOREnvParquet
from src.moe_dqn import MoENetwork
from src.train_agent import train_dqn
from src.offline_dataset import list_available_dates


def find_data_root(project_root: Path) -> Path:
    candidates = [
        project_root / "data" / "l2",
        project_root / "data" / "l2_parquet",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Não encontrei dataset em: {candidates}")


def list_venues_symbols(root: Path, max_items: int = 20):
    venues = sorted([p.name.split("venue=")[-1] for p in root.glob("venue=*") if p.is_dir()])
    pairs = []
    for v in root.glob("venue=*"):
        if not v.is_dir():
            continue
        for s in v.glob("symbol=*"):
            if s.is_dir():
                pairs.append((v.name, s.name))
    pairs = sorted(pairs)[:max_items]
    return venues, pairs


def safe_dates(root: Path, venue: str, symbol: str):
    try:
        return list_available_dates(root, venue, symbol)
    except FileNotFoundError:
        return []


def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent  # ajuste se seu layout for diferente
    ROOT = find_data_root(PROJECT_ROOT)

    VENUE_B3 = "B3"
    VENUE_BASE = "BASE"

    SYMBOLS = ["PETR4", "VALE3", "ITUB4"]

    EP_LEN = 500
    TOTAL_ORDER = 10_000

    print("DATA_ROOT =", ROOT)

    venues, sample_pairs = list_venues_symbols(ROOT)
    print("Venues encontrados:", venues)
    print("Amostra (venue, symbol):", sample_pairs)

    for symbol in SYMBOLS:
        d_b3 = safe_dates(ROOT, VENUE_B3, symbol)
        d_base = safe_dates(ROOT, VENUE_BASE, symbol)

        common = sorted(set(d_b3) & set(d_base))

        if not d_b3:
            print(f"[SKIP] {symbol}: não existe em venue={VENUE_B3} dentro de {ROOT}")
            continue
        if not d_base:
            print(f"[SKIP] {symbol}: não existe em venue={VENUE_BASE} dentro de {ROOT}")
            continue
        if not common:
            print(f"[SKIP] {symbol}: sem datas em comum entre {VENUE_B3} e {VENUE_BASE}")
            continue

        print(f"\n[OK] {symbol}: dias B3={len(d_b3)} BASE={len(d_base)} comum={len(common)}")

        env = MultiVenueSOREnvParquet(
            root=str(ROOT),
            symbol=symbol,
            venue_b3=VENUE_B3,
            venue_base=VENUE_BASE,
            episode_len=EP_LEN,
            total_inventory=TOTAL_ORDER,
            max_slippage_pct=0.001,
            seed=42,
        )

        # GARANTIA: só usar dias comuns (senão pode resetar num dia que não existe no BASE)
        env._dates = common  # workaround prático

        model = MoENetwork(input_dim=5, output_dim=4, num_experts=3)

        model_trained, rewards = train_dqn(env, model, seed=42)

        out = PROJECT_ROOT / "models"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"moe_dqn_sor_{symbol}_12m.pth"
        torch.save(model_trained.state_dict(), path)
        print(f"[SAVE] {symbol}: {path}")


if __name__ == "__main__":
    main()