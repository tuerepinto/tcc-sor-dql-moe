from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch

from src.ib_realtime_topbook import IBRealTimeLOB, criar_contrato_acao_br  # sua classe já existe


def build_obs_from_topbook(
    b3_quote: Dict[str, Any],
    base_quote: Dict[str, Any],
    inventory_remaining: float,
) -> np.ndarray:
    """
    Proxy L1 -> estado 5D esperado pelo modelo:
      [ask_b3, vol_ask_b3, ask_base, vol_ask_base, inv]
    Aqui usamos ask_size como vol_ask_1.
    """
    ask_b3 = float(b3_quote.get("ask") or 0.0)
    vol_b3 = float(b3_quote.get("ask_size") or 0.0)
    ask_base = float(base_quote.get("ask") or 0.0)
    vol_base = float(base_quote.get("ask_size") or 0.0)

    return np.array([ask_b3, vol_b3, ask_base, vol_base, float(inventory_remaining)], dtype=np.float32)


def main():
    # --- configuração ---
    symbols = ["PETR4", "VALE3", "ITUB4"]
    tickers_map = {i + 1: sym for i, sym in enumerate(symbols)}

    # carrega modelo (inferência)
    model_path = Path("models/moe_dqn_sor_PETR4_12m.pth")  # ajuste
    model = torch.nn.Module()  # placeholder só para tipo

    # Você vai instanciar o mesmo tipo usado no treino
    from src.moe_dqn import MoENetwork
    model = MoENetwork(input_dim=5, output_dim=4, num_experts=3)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    device = torch.device("cpu")
    model.to(device)

    # --- conecta IB ---
    app = IBRealTimeLOB(tickers_map=tickers_map)
    app.connect("127.0.0.1", 7497, clientId=7)  # ajuste host/port/clientId
    t = threading.Thread(target=app.run_loop, daemon=True)
    t.start()

    if not app.ready.wait(timeout=10):
        raise RuntimeError("IBKR não respondeu nextValidId a tempo.")

    # reqMktData para cada símbolo (L1)
    for reqId, sym in tickers_map.items():
        c = criar_contrato_acao_br(sym)
        app.reqMktData(reqId, c, "", False, False, [])

    # --- loop de inferência + logging ---
    rows: List[Dict[str, Any]] = []
    inventory_remaining = 10_000.0  # “paper inventory” só para estado

    start = time.time()
    try:
        while True:
            time.sleep(1.0)  # frequência de decisão/log

            # snapshot interno
            with app._lock:
                snap = {sym: dict(app.lob_data[sym]) for sym in symbols}

            # Exemplo: usar PETR4 como "B3" e VALE3 como "Base" é só demo.
            # Para multi-venue real, você precisa de depth/quotes por exchange.
            b3_quote = snap["PETR4"]
            base_quote = snap["VALE3"]

            obs = build_obs_from_topbook(b3_quote, base_quote, inventory_remaining)
            st = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                q = model(st)  # (1,4)
                gate = torch.softmax(model.gating_network(st), dim=-1)  # (1,E)

            action = int(torch.argmax(q, dim=1).item())

            rows.append({
                "ts": datetime.now(timezone.utc),
                "ask_b3": float(obs[0]),
                "vol_b3": float(obs[1]),
                "ask_base": float(obs[2]),
                "vol_base": float(obs[3]),
                "inv": float(obs[4]),
                "action": action,
                "q0": float(q[0,0].item()),
                "q1": float(q[0,1].item()),
                "q2": float(q[0,2].item()),
                "q3": float(q[0,3].item()),
                "gate_e1": float(gate[0,0].item()),
                "gate_e2": float(gate[0,1].item()),
                "gate_e3": float(gate[0,2].item()),
            })

            # flush simples a cada 60 linhas
            if len(rows) % 60 == 0:
                df = pd.DataFrame(rows)
                out = Path("logs")
                out.mkdir(parents=True, exist_ok=True)
                df.to_parquet(out / "online_agent_behavior.parquet", index=False)

    finally:
        app.stop()
        df = pd.DataFrame(rows)
        out = Path("logs")
        out.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out / "online_agent_behavior.parquet", index=False)

if __name__ == "__main__":
    main()