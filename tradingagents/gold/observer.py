"""CLI entry point for live observation (no execution): ``python -m tradingagents.gold.observer``.

Deterministic-only by default; pass --with-analysis to run the full
multi-agent graph per cycle (REQUIRES LLM API keys in the environment).
Data comes from the Yahoo GC=F proxy (labelled); a spot/MT5 provider can be
substituted programmatically via LiveObserver.
"""

from __future__ import annotations

import argparse
import logging

from tradingagents.gold.config import CostMode, default_gold_config
from tradingagents.gold.data.yahoo_gold import YahooGoldProxyProvider
from tradingagents.gold.graph import GoldTradingAgentsGraph
from tradingagents.gold.observation import LiveObserver

logger = logging.getLogger("gold.observer")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tradingagents.gold.observer",
        description="Observe live XAUUSD, log auditable hypothetical decisions. NO execution.",
    )
    parser.add_argument("--cycles", type=int, default=1, help="number of observation cycles")
    parser.add_argument("--interval", type=float, default=900.0,
                        help="seconds between cycles (default 15 min = one M15 bar)")
    parser.add_argument("--with-analysis", action="store_true",
                        help="run the full multi-agent graph each cycle (REQUIRES LLM API keys)")
    parser.add_argument("--cost-mode", choices=[m.value for m in CostMode], default=CostMode.FAST.value)
    parser.add_argument("--equity", type=float, default=10_000.0, help="paper account equity")
    parser.add_argument("--log-path", default=None, help="JSONL observation log path")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    config = default_gold_config(cost_mode=CostMode(args.cost_mode))
    provider = YahooGoldProxyProvider(config.data)

    graph_factory = None
    if args.with_analysis:
        def graph_factory():
            return GoldTradingAgentsGraph(gold_config=config)

    observer = LiveObserver(
        provider,
        gold_config=config,
        graph_factory=graph_factory,
        account_equity=args.equity,
        log_path=__import__("pathlib").Path(args.log_path) if args.log_path else None,
    )
    records = observer.run(args.cycles, args.interval)
    for record in records:
        status = "OK" if record.snapshot_ok else "DATA ERROR"
        logger.info("cycle %d: %s, decision=%s, gate=%s",
                    record.cycle, status,
                    (record.decision or {}).get("action", "n/a"),
                    "approved" if record.gate_approved else "no-trade")
    logger.info("observation log: %s", observer.log.path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
