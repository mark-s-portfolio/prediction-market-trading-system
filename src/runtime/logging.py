"""Runtime logging utilities for the public portfolio edition.

The trading engine is event-driven, so terminal I/O must never become a
latency bottleneck for market-data callbacks or order-lifecycle tasks.

This module provides:
- broken-pipe-safe stdout/stderr wrappers;
- a bounded non-blocking log queue;
- a dedicated writer thread;
- lightweight queue/drop telemetry;
- bounded shutdown draining.

No strategy parameters or market-specific trading logic live here.
"""

from __future__ import annotations

import atexit
import builtins
import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path
from typing import TextIO


DEFAULT_FALLBACK_LOG_PATH = os.getenv(
    "TRADING_FALLBACK_LOG_PATH",
    "runtime-fallback.log",
)
DEFAULT_QUEUE_MAX = max(
    1_000,
    int(os.getenv("TRADING_LOG_QUEUE_MAX", "50000")),
)


class BrokenPipeSafeStream:
    """Wrap a text stream and fall back to a file after terminal disconnect."""

    def __init__(self, wrapped: TextIO, fallback_path: str | os.PathLike[str]):
        self._wrapped = wrapped
        self._fallback_path = Path(fallback_path)
        self._terminal_ok = True
        self.encoding = getattr(wrapped, "encoding", "utf-8")

    def write(self, data: str) -> int:
        if not data:
            return 0

        if self._terminal_ok and self._wrapped is not None:
            try:
                return self._wrapped.write(data)
            except (BrokenPipeError, OSError):
                self._terminal_ok = False

        try:
            self._fallback_path.parent.mkdir(parents=True, exist_ok=True)
            with self._fallback_path.open("a", encoding="utf-8", errors="replace") as handle:
                return handle.write(data)
        except Exception:
            return 0

    def flush(self) -> None:
        if self._terminal_ok and self._wrapped is not None:
            try:
                self._wrapped.flush()
            except (BrokenPipeError, OSError):
                self._terminal_ok = False

    def isatty(self) -> bool:
        try:
            return bool(
                self._terminal_ok
                and self._wrapped is not None
                and self._wrapped.isatty()
            )
        except Exception:
            return False

    def fileno(self) -> int:
        return self._wrapped.fileno()


class NonBlockingRuntimeLogger:
    """Bounded asynchronous writer for latency-sensitive runtime output."""

    _SENTINEL = object()

    def __init__(
        self,
        *,
        queue_max: int = DEFAULT_QUEUE_MAX,
        fallback_path: str | os.PathLike[str] = DEFAULT_FALLBACK_LOG_PATH,
        flush_interval_sec: float = 0.05,
        shutdown_drain_limit: int = 10_000,
    ) -> None:
        self._fallback_path = Path(fallback_path)
        self._flush_interval_sec = max(0.0, float(flush_interval_sec))
        self._shutdown_drain_limit = max(0, int(shutdown_drain_limit))
        self._queue: queue.Queue[object] = queue.Queue(maxsize=max(1_000, int(queue_max)))
        self._thread: threading.Thread | None = None
        self._active = False
        self._stats_lock = threading.Lock()
        self._stats = {
            "queued": 0,
            "written": 0,
            "dropped": 0,
            "max_depth": 0,
        }

    @property
    def active(self) -> bool:
        return self._active

    def stats(self, *, reset: bool = False) -> dict[str, int]:
        with self._stats_lock:
            result = dict(self._stats)
            result["depth"] = self._queue.qsize()
            if reset:
                self._stats.update(queued=0, written=0, dropped=0)
                self._stats["max_depth"] = result["depth"]
            return result

    def _increment(self, key: str, amount: int = 1) -> None:
        with self._stats_lock:
            self._stats[key] = int(self._stats.get(key, 0)) + int(amount)

    def start(self) -> None:
        if self._active:
            return
        self._active = True
        self._thread = threading.Thread(
            target=self._worker,
            name="runtime-log-writer",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if not self._active:
            return

        self._active = False
        try:
            self._queue.put_nowait(self._SENTINEL)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(self._SENTINEL)
            except Exception:
                pass

        thread = self._thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)

    def write(
        self,
        *args: object,
        sep: str = " ",
        end: str = "\n",
        file: TextIO | None = None,
        flush: bool = False,
    ) -> None:
        """Queue a log line without blocking the caller on terminal I/O."""

        stream = file if file is not None else sys.stdout

        if not self._active or stream not in (sys.stdout, sys.stderr):
            builtins.print(*args, sep=sep, end=end, file=file, flush=flush)
            return

        try:
            text = sep.join(str(value) for value in args) + end
            self._queue.put_nowait((text, stream, bool(flush)))
            depth = self._queue.qsize()
            with self._stats_lock:
                self._stats["queued"] += 1
                self._stats["max_depth"] = max(self._stats["max_depth"], depth)
        except queue.Full:
            self._increment("dropped")
        except Exception:
            builtins.print(*args, sep=sep, end=end, file=file, flush=flush)

    def _fallback_write(self, text: str) -> None:
        try:
            self._fallback_path.parent.mkdir(parents=True, exist_ok=True)
            with self._fallback_path.open("a", encoding="utf-8", errors="replace") as handle:
                handle.write(text)
        except Exception:
            pass

    def _worker(self) -> None:
        last_flush = time.monotonic()
        reported_drops = 0

        while True:
            try:
                item = self._queue.get(timeout=0.10)
            except queue.Empty:
                item = None

            if item is self._SENTINEL:
                break

            if item is not None:
                text, stream, force_flush = item
                try:
                    stream.write(text)
                    self._increment("written")
                except Exception:
                    self._fallback_write(text)

                now = time.monotonic()
                if force_flush or now - last_flush >= self._flush_interval_sec:
                    try:
                        stream.flush()
                    except Exception:
                        pass
                    last_flush = now

            current_drops = self.stats()["dropped"]
            if (
                current_drops > reported_drops
                and self._queue.qsize() < max(1, self._queue.maxsize // 2)
            ):
                delta = current_drops - reported_drops
                reported_drops = current_drops
                try:
                    sys.stdout.write(
                        f"[runtime-log] queue overflow: dropped={delta} "
                        f"total={current_drops}\n"
                    )
                    sys.stdout.flush()
                except Exception:
                    pass

        for _ in range(self._shutdown_drain_limit):
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break

            if item is self._SENTINEL:
                continue

            try:
                text, stream, _force_flush = item
                stream.write(text)
                self._increment("written")
            except Exception:
                pass

        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass


def install_disconnect_resilience(
    *,
    fallback_path: str | os.PathLike[str] = DEFAULT_FALLBACK_LOG_PATH,
    enabled: bool = True,
) -> None:
    """Protect stdout/stderr from SSH/PTY disconnects where supported."""

    if not enabled:
        return

    for signal_name in ("SIGHUP", "SIGPIPE"):
        try:
            if hasattr(signal, signal_name):
                signal.signal(getattr(signal, signal_name), signal.SIG_IGN)
        except Exception:
            pass

    try:
        if not isinstance(sys.stdout, BrokenPipeSafeStream):
            sys.stdout = BrokenPipeSafeStream(sys.stdout, fallback_path)
        if not isinstance(sys.stderr, BrokenPipeSafeStream):
            sys.stderr = BrokenPipeSafeStream(sys.stderr, fallback_path)
    except Exception:
        pass


runtime_logger = NonBlockingRuntimeLogger()
runtime_print = runtime_logger.write

atexit.register(runtime_logger.stop)
