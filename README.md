
# Prediction Market Trading System

A **sanitized portfolio edition** of an event-driven prediction-market trading and execution system built in Python.

The repository focuses on engineering architecture rather than publishing a production trading strategy. It demonstrates asynchronous market-data ingestion, exact order-lifecycle ownership, reconciliation, fill accounting, portfolio risk aggregation, quantitative state modeling, one-shot admission authority, and a thin orchestration engine.

> **Safety:** the standalone `src/main.py` entrypoint is intentionally **observe-only**. Production strategy rules, tuned thresholds, sizing logic, historical trade corpora, and turnkey live-execution wiring are not included.

## Highlights

- Resilient WebSocket market-data client with reconnect, heartbeat, silence detection, duplicate-frame filtering, bounded handler delivery, and source-aware order-book state.
- Serialized CLOB transport boundary with write priority, lifecycle-read fairness, retry/backoff, generation-aware status caching, and no automatic retry after ambiguous raw order submission.
- Explicit order lifecycle state machine for submission ambiguity, exact OID ownership, cancellation uncertainty, monotonic fill evidence, generation handoff, and terminal-zero release.
- Exact-order reconciliation across status, trade, and wallet evidence where **UNKNOWN is never treated as ZERO**.
- Idempotent fill accounting with cumulative/incremental evidence, realized-price quality, overfill preservation, and alias-aware fill deduplication.
- Position accounting that separates confirmed quantity from price certainty and supports late exact-price hydration.
- Generic portfolio risk manager consuming confirmed inventory plus unresolved execution ownership.
- Immutable candidate, measurement-only quality, explicit `ALLOW / DENY / DEFER` admission, and one-shot permits bound through the final pre-network boundary.
- Generic discretized Markov transition model with decay, reproducible first-passage simulation, and local/persistent state separation.
- Thin event-driven engine that coordinates services without recreating a monolithic mutable strategy state.

## Architecture

```text
Market discovery
      │
      ▼
WebSocket ───────► OrderBookStore
                      │
                      ▼
                TradingEngine
                  │   │   │
        candidate │   │   └──► RiskManager
                  │   └──────► Quality / Admission
                  ▼
           one-shot permit
                  │
                  ▼
             PRE_NETWORK
                  │
                  ▼
             ClobTransport
                  │
          ┌───────┴────────┐
          ▼                ▼
   OrderLifecycle     FillAccounting
          │                │
          └──────► Reconciliation
                           │
                           ▼
                      PositionBook
```

A more detailed design walkthrough is in [`docs/architecture.md`](docs/architecture.md).

## Repository layout

```text
src/
├── main.py
├── engine.py
├── market/
│   ├── types.py
│   ├── discovery.py
│   ├── orderbook.py
│   └── websocket.py
├── execution/
│   ├── types.py
│   ├── clob_transport.py
│   ├── order_manager.py
│   ├── order_lifecycle.py
│   ├── reconciliation.py
│   └── fill_accounting.py
├── models/
│   └── markov.py
├── strategy/
│   ├── candidate.py
│   ├── admission.py
│   ├── quality.py
│   └── public_policy.py
├── risk/
│   ├── risk_manager.py
│   └── position_state.py
└── runtime/
    ├── logging.py
    └── config.py

tests/
├── test_order_lifecycle.py
├── test_reconciliation.py
├── test_admission.py
├── test_markov.py
└── test_infrastructure.py
```

## Run locally

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
pytest -q
```

The regression suite currently contains **100 tests**.

To start the sanitized observe-only runtime:

```bash
python -m src.main
```

The process may connect to public market-data endpoints and rotate active market subscriptions, but the bundled candidate producer is intentionally empty and the execution client is fail-closed.

## Safety and sanitization boundary

This repository intentionally does **not** contain:

- production/private strategy implementation;
- tuned asset-specific entry or completion thresholds;
- historical winning/losing setup corpora;
- production bankroll or sizing policy;
- private keys, API credentials, or production logs;
- claims of guaranteed or historical profitability.

The public code exposes generic contracts and engineering patterns while keeping proprietary decision logic outside the repository.

## Testing philosophy

The tests emphasize invariants rather than happy-path mocks. Examples include:

- a raw-POST ambiguity remains owned until reconciled;
- stale status cannot regress stronger cumulative fill evidence;
- a successful cancel request is not automatically zero-fill proof;
- foreign OID evidence cannot be relabeled to another lifecycle;
- positive exact evidence overrides provisional zero interpretations;
- an admission permit cannot be replayed, resurrected, or rebound to another lifecycle;
- Markov state is contract-local where required and seeded simulations are reproducible;
- top-only WebSocket events cannot silently refresh stale full-depth liquidity.

## Status

This is a **portfolio snapshot**, not a packaged trading product or financial-performance claim. The standalone composition root is designed for architecture review and observe-only demonstration.

