from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import yfinance as yf

from src.models.moe_network import MoENetwork
from src.models.model_io import find_model_path


def yf_symbol(sym: str) -> str:
    return sym if sym.endswith(".SA") else f"{sym}.SA"


def _extract_series(df: pd.DataFrame, field: str) -> pd.Series:
    """
    Extrai coluna como Series 1D mesmo quando yfinance retorna MultiIndex.
    field típico: 'Close' ou 'Volume'
    """
    if df.empty:
        return pd.Series(dtype=float)

    # Caso simples (colunas normais)
    if not isinstance(df.columns, pd.MultiIndex):
        if field in df.columns:
            return pd.to_numeric(df[field], errors="coerce")
        if field == "Close" and "Adj Close" in df.columns:
            return pd.to_numeric(df["Adj Close"], errors="coerce")
        return pd.Series(np.nan, index=df.index, dtype=float)

    # Caso MultiIndex
    lv0 = df.columns.get_level_values(0)
    lv1 = df.columns.get_level_values(1)

    s = None
    if field in lv0:
        tmp = df.xs(field, axis=1, level=0, drop_level=False)
        s = tmp.iloc[:, 0] if isinstance(tmp, pd.DataFrame) else tmp
    elif field in lv1:
        tmp = df.xs(field, axis=1, level=1, drop_level=False)
        s = tmp.iloc[:, 0] if isinstance(tmp, pd.DataFrame) else tmp
    elif field == "Close" and "Adj Close" in lv0:
        tmp = df.xs("Adj Close", axis=1, level=0, drop_level=False)
        s = tmp.iloc[:, 0] if isinstance(tmp, pd.DataFrame) else tmp

    if s is None:
        return pd.Series(np.nan, index=df.index, dtype=float)

    return pd.to_numeric(s, errors="coerce")


def to_proxy(df: pd.DataFrame, spread_bps: float = 2.0) -> pd.DataFrame:
    df = df.reset_index().copy()

    ts_col = "Datetime" if "Datetime" in df.columns else ("Date" if "Date" in df.columns else df.columns[0])
    df = df.rename(columns={ts_col: "ts"})

    close = _extract_series(df, "Close")
    vol = _extract_series(df, "Volume").fillna(0.0)

    half = (spread_bps / 10000.0) / 2.0
    ask = close * (1.0 + half)
    ask_size = np.maximum(1.0, vol / 1000.0)

    out = pd.DataFrame({
        "ts": pd.to_datetime(df["ts"], errors="coerce", utc=True),
        "ask": ask.astype(float),
        "ask_size": ask_size.astype(float),
    }).dropna().reset_index(drop=True)

    if out.empty:
        raise RuntimeError("Sem dados válidos após transformação to_proxy().")

    return out


def run_symbol(symbol: str, model: torch.nn.Module, period: str = "1d", interval: str = "1m") -> pd.DataFrame:
    raw = yf.download(
        yf_symbol(symbol),
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False,
        group_by="column",
        threads=False,
    )

    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance retornou vazio para {symbol} ({period}, {interval}).")

    q = to_proxy(raw)

    rows = []
    inv = 10_000.0
    for _, r in q.iterrows():
        ask_b3 = float(r["ask"])
        vol_b3 = float(r["ask_size"])

        # proxy BASE
        ask_base = ask_b3 * 1.0001
        vol_base = vol_b3 * 0.8

        obs = np.array([ask_b3, vol_b3, ask_base, vol_base, inv], dtype=np.float32)
        st = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            qv = model(st)
            gate = torch.softmax(model.gating_network(st), dim=-1)
            action = int(torch.argmax(qv, dim=1).item())

        rows.append({
            "symbol": symbol,
            "ts": r["ts"],
            "ask_b3": ask_b3,
            "vol_b3": vol_b3,
            "ask_base": ask_base,
            "vol_base": vol_base,
            "inv": inv,
            "action": action,
            "q0": float(qv[0, 0]),
            "q1": float(qv[0, 1]),
            "q2": float(qv[0, 2]),
            "q3": float(qv[0, 3]),
            "gate_0": float(gate[0, 0]),
            "gate_1": float(gate[0, 1]),
            "gate_2": float(gate[0, 2]),
        })

    return pd.DataFrame(rows)


def main():
    symbols = ["PETR4", "VALE3", "ITUB4"]

    model_path = find_model_path(Path("models"))
    model = MoENetwork(input_dim=5, output_dim=4, num_experts=3)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    out = Path("logs")
    out.mkdir(parents=True, exist_ok=True)

    all_logs = []
    for sym in symbols:
        try:
            df_sym = run_symbol(sym, model, period="1d", interval="1m")
            all_logs.append(df_sym)
            df_sym.to_parquet(out / f"online_agent_behavior_yfinance_{sym}.parquet", index=False)
            print(f"[OK] {sym}: {len(df_sym)} linhas")
        except Exception as e:
            print(f"[WARN] {sym}: {e}")

    if all_logs:
        df_all = pd.concat(all_logs, ignore_index=True)
        df_all.to_parquet(out / "online_agent_behavior_yfinance_all.parquet", index=False)
        print("[OK] logs/online_agent_behavior_yfinance_all.parquet")
    else:
        raise RuntimeError("Nenhum ativo gerou dados válidos.")


if __name__ == "__main__":
    main()