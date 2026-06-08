import threading
import time
from datetime import datetime, timezone

import pandas as pd

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.ticktype import TickTypeEnum


def criar_contrato_acao_br(symbol: str) -> Contract:
    """
    Contrato genérico para ação.
    Observação: exchange/primaryExchange podem precisar ajuste.
    Comece com SMART e valide com ContractDetails se necessário.
    """
    c = Contract()
    c.secType = "STK"
    c.symbol = symbol
    c.currency = "BRL"
    c.exchange = "SMART"
    return c


class IBRealTimeLOB(EWrapper, EClient):
    """
    Coleta L1 (bid/ask e sizes) via reqMktData.
    Mantém estado por ticker e também gera um log tabular.
    """

    def __init__(self, tickers_map: dict[int, str]):
        EClient.__init__(self, self)

        self.tickers_map = dict(tickers_map)  # {reqId: "PETR4", ...}

        self.ready = threading.Event()
        self._lock = threading.Lock()

        # estado atual
        self.lob_data = {
            sym: {"bid": None, "ask": None, "bid_size": None, "ask_size": None, "ts": None}
            for sym in self.tickers_map.values()
        }

        # log
        self.rows = []
        self._stop = threading.Event()

    # -------- lifecycle / conexão --------
    def nextValidId(self, orderId: int):
        self.ready.set()

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        # Você pode filtrar aqui códigos "informativos" se quiser,
        # mas no começo é melhor logar tudo para depurar contrato/subscription.
        print(f"[IB][ERROR] reqId={reqId} code={errorCode} msg={errorString}")

    # -------- callbacks de market data (L1) --------
    def tickPrice(self, reqId, tickType, price, attrib):
        sym = self.tickers_map.get(reqId)
        if sym is None:
            return

        with self._lock:
            if tickType == TickTypeEnum.BID:
                self.lob_data[sym]["bid"] = float(price)
            elif tickType == TickTypeEnum.ASK:
                self.lob_data[sym]["ask"] = float(price)

            self.lob_data[sym]["ts"] = datetime.now(timezone.utc)

    def tickSize(self, reqId, tickType, size):
        sym = self.tickers_map.get(reqId)
        if sym is None:
            return

        with self._lock:
            if tickType == TickTypeEnum.BID_SIZE:
                self.lob_data[sym]["bid_size"] = int(size)
            elif tickType == TickTypeEnum.ASK_SIZE:
                self.lob_data[sym]["ask_size"] = int(size)

            self.lob_data[sym]["ts"] = datetime.now(timezone.utc)

    # -------- util: snapshot para log --------
    def snapshot_to_rows(self):
        now = datetime.now(timezone.utc)
        with self._lock:
            for sym, d in self.lob_data.items():
                self.rows.append({
                    "ts": now,
                    "symbol": sym,
                    "bid": d["bid"],
                    "ask": d["ask"],
                    "bid_size": d["bid_size"],
                    "ask_size": d["ask_size"],
                })

    def run_loop(self):
        # Loop da IB API (bloqueia). Por isso roda em thread.
        self.run()

    def stop(self):
        self._stop.set()
        self.disconnect()