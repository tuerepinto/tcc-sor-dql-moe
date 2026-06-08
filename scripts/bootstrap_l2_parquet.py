from pathlib import Path
import numpy as np
import pandas as pd


def make_df(n=500, seed=42, base=35.0):
    rng = np.random.default_rng(seed)
    px = base + rng.normal(0, 0.02, n)
    return pd.DataFrame({
        "ask_1": px + 0.01, "vol_ask_1": rng.integers(100, 900, n),
        "ask_2": px + 0.02, "vol_ask_2": rng.integers(100, 900, n),
        "ask_3": px + 0.03, "vol_ask_3": rng.integers(100, 900, n),
        "ask_4": px + 0.04, "vol_ask_4": rng.integers(100, 900, n),
        "ask_5": px + 0.05, "vol_ask_5": rng.integers(100, 900, n),
    })


def save_part(df: pd.DataFrame, root: Path, venue: str, symbol: str, date: str):
    out = root / f"venue={venue}" / f"symbol={symbol}" / f"date={date}"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "part-000.parquet", index=False)


def generate_dates(start_date: str = "2026-01-02", n_days: int = 30, business_days: bool = True):
    """
    Retorna lista de datas no formato YYYY-MM-DD.
    business_days=True -> usa dias úteis (B)
    business_days=False -> usa dias corridos (D)
    """
    freq = "B" if business_days else "D"
    return pd.date_range(start=start_date, periods=n_days, freq=freq).strftime("%Y-%m-%d").tolist()


def main():
    root = Path("data/l2_parquet")
    symbols = ["PETR4", "VALE3", "ITUB4"]
    dates = generate_dates(start_date="2026-01-02", n_days=30, business_days=True)  # 30 dias úteis

    for s_i, sym in enumerate(symbols):
        for d_i, d in enumerate(dates):
            df_b3 = make_df(n=500, seed=100 + s_i * 1000 + d_i, base=35.00 + s_i)
            df_base = make_df(n=500, seed=200 + s_i * 1000 + d_i, base=34.98 + s_i)

            save_part(df_b3, root, "B3", sym, d)
            save_part(df_base, root, "BASE", sym, d)

    print(f"[OK] Dataset sintético salvo em {root}")
    print(f"[OK] Datas geradas: {len(dates)} (de {dates[0]} até {dates[-1]})")


if __name__ == "__main__":
    main()