# Architecture

## Purpose

This repository is a sanitized portfolio snapshot of a larger event-driven prediction-market system. The public architecture preserves the difficult software-engineering boundaries—concurrency, ownership, reconciliation, evidence integrity, and service composition—while intentionally omitting proprietary strategy rules and production parameters.

The central design principle is **single ownership of facts**. Market data, execution lifecycle, fill evidence, economic inventory, risk, admission authority, and orchestration each have a dedicated owner instead of being repeatedly reconstructed inside one large strategy class.

## Runtime data flow

```text
Gamma / market discovery
        │
        ▼
MarketDefinition
        │
        ▼
MarketWebSocketClient
        │
        ├── full depth ───────────────┐
        ├── deltas                    │
        └── top-only observations     │
                                      ▼
                               OrderBookStore
                                      │
                              MarketBooks snapshot
                                      │
                                      ▼
                                TradingEngine
                         ┌────────────┼────────────┐
                         ▼            ▼            ▼
                CandidateProducer  Quality     RiskManager
                         │            │            │
                         └────────────┼────────────┘
                                      ▼
                                AdmissionPolicy
                                      │
                                ALLOW permit
                                      │
                                      ▼
                          final PRE_NETWORK check
                                      │
                                      ▼
                                ClobTransport
                                      │
                         exact raw-POST handoff
                                      │
                ┌─────────────────────┴─────────────────────┐
                ▼                                           ▼
      OrderLifecycleService                           FillAccounting
                │                                           │
                └──────────────────┬────────────────────────┘
                                   ▼
                         ReconciliationService
                                   │
                                   ▼
                              PositionBook
```

## Market-data layer

`market/discovery.py` resolves active binary markets and separates discovery/cache concerns from strategy selection.

`market/orderbook.py` owns normalized order-book state. WebSocket depth and REST recovery remain source-separated. A top-only observation is stored as synthetic one-level state and cannot refresh or inherit stale full-depth liquidity.

`market/websocket.py` owns connection generations, subscriptions, heartbeat, silence detection, duplicate-frame suppression, coverage telemetry, and bounded event delivery. Normalized book state is committed before downstream notification, so a slow async consumer cannot block socket receive indefinitely.

## Execution transport

`execution/clob_transport.py` wraps one synchronous venue SDK behind a thread-safe coordinator. It provides:

- serialized SDK access;
- write priority with bounded lifecycle-read starvation escape;
- retry/backoff for suitable calls;
- generation-aware exact-order status caching;
- market-metadata prewarming;
- an exact synchronous final validation point immediately before raw order submission.

A raw order POST is never automatically retried after the irreversible raw-entry boundary. Ambiguity is transferred to reconciliation instead.

## Lifecycle ownership

`execution/order_lifecycle.py` is the canonical owner of local execution state. Its important invariants are:

- one lifecycle generation receives at most one submission handoff;
- exact OID affinity cannot change;
- cumulative fill evidence is monotonic;
- stale `LIVE` observations cannot reopen a filled order;
- cancellation uncertainty is represented explicitly;
- a transport-level cancel success is not automatically zero-fill proof;
- a raw-POST terminal lifecycle cannot be released or superseded without explicit terminal-zero reconciliation evidence;
- stale generations cannot overwrite newer generations.

## Fill accounting and reconciliation

`execution/fill_accounting.py` owns exact-order execution evidence. It separates cumulative status evidence, incremental fills/trades, realized-price quality, and wallet-baseline observations. Native execution IDs deduplicate rows even when the venue exposes the same execution under multiple response aliases.

`execution/reconciliation.py` merges exact-order evidence conservatively. The key rule is:

> **UNKNOWN ≠ ZERO**

An unreadable status, a missing OID, or a zero wallet delta without the required ownership context cannot silently become terminal-zero proof. Positive exact evidence always overrides provisional zero interpretations.

## Economic inventory

`risk/position_state.py` owns confirmed economic quantity independently from order status. Quantity can be known while cost basis remains unknown; later stronger exact price evidence may hydrate previously unpriced confirmed BUY inventory without inventing another fill.

This separation prevents a terminal order from being confused with a flat economic position.

## Portfolio risk

`risk/risk_manager.py` consumes two independent truth domains:

1. confirmed economic inventory from `PositionBook`;
2. unresolved/live execution ownership from `OrderLifecycleService`.

Risk decisions are immutable and bound to the exact `ProposedExposure` they evaluated. At the final raw-POST revalidation boundary, only the currently validated lifecycle may be excluded from duplicate-owner counting; all other owners and portfolio exposure remain authoritative.

## Candidate and admission contracts

`strategy/candidate.py` defines immutable candidate identity and provenance.

`strategy/quality.py` is measurement-only. It describes current evidence and does not rank candidates or issue trading decisions.

`strategy/admission.py` consumes candidate, quality, exact risk proposal, and exact risk decision into explicit `ALLOW`, `DENY`, or `DEFER` semantics.

`strategy/public_policy.py` demonstrates a neutral public policy and one-shot permit bridge. An ALLOW permit is bound to one candidate, policy generation, validity window, and lifecycle preparation. It is consumed synchronously at raw-POST entry and cannot be replayed or resurrected.

## Quantitative model

`models/markov.py` is a generic discretized Markov transition model. It is deliberately separated from admission policy. It provides:

- transition counting with prior mass;
- optional time decay;
- normalized probability matrices;
- configurable probability refresh cadence;
- seeded reproducible first-passage simulation;
- bounded simulation caching;
- local/persistent view separation to avoid accidental cross-contract transitions.

The model emits measurements, not public trading rules.

## Engine

`engine.py` is intentionally thin compared with the private monolithic system it replaces. It coordinates dedicated services rather than owning duplicate mutable truth.

Candidate generation and enrichment happen outside the tiny portfolio-commit lock. The final serialized section performs current quality measurement, current risk assessment, admission, permit preparation, and submission handoff. Immediately before raw POST, the transport validator re-reads the current market generation and re-evaluates the exact candidate/risk/admission binding without network I/O.

## Standalone application

`main.py` is a composition root, not a strategy module. The public runtime:

- constructs the service graph;
- uses an empty default candidate producer;
- runs `TradingEngine` in `OBSERVE_ONLY` mode;
- uses a fail-closed disabled execution client;
- performs asynchronous market discovery and WebSocket rotation;
- owns shutdown of socket, engine, execution manager, order-book store, and logging.

Environment flags alone cannot turn the sanitized portfolio entrypoint into a live trader.

## Regression suite

The repository currently contains 100 tests covering lifecycle ownership, reconciliation evidence, admission authority, Markov numerical invariants, and infrastructure hardening.

The tests are intended to document architectural invariants as executable specifications rather than demonstrate financial performance.
