"""Visible progress and structured incidents without configuring global logging."""

from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


class RunReporter:
    """Report each independent unit; optional JSONL is flushed after every event."""

    def __init__(self, *, verbose: bool = True, log_path: str | Path | None = None):
        self.verbose = verbose
        self.log_path = Path(log_path) if log_path else None
        self.events: list[dict] = []
        self.started = time.monotonic()
        self.logger = logging.getLogger("weight_analysis.experiments")
        self._log_error_reported = False

    def emit(self, status: str, unit_id: str, message: str, *, exception=None, **context):
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "unit_id": unit_id,
            "message": message,
            "elapsed_seconds": time.monotonic() - self.started,
            **context,
        }
        if exception is not None:
            event.update(
                error_type=type(exception).__name__,
                traceback="".join(traceback.format_exception(exception)),
            )
        self.events.append(event)
        level = logging.ERROR if exception is not None else logging.INFO
        self.logger.log(level, "%s %s %s", status, unit_id, message)
        if self.verbose:
            detail = " " + json.dumps(context, default=str, ensure_ascii=False) if context else ""
            print(f"[{status.upper()}] {unit_id}: {message}{detail}", flush=True)
            if exception is not None:
                print(event["traceback"], file=sys.stderr, flush=True)
        if self.log_path:
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a", encoding="utf8") as stream:
                    stream.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
                    stream.flush()
            except OSError as exc:
                if not self._log_error_reported:
                    self.events.append(
                        {
                            "timestamp": event["timestamp"],
                            "status": "warning",
                            "unit_id": "log",
                            "message": str(exc),
                            "error_type": type(exc).__name__,
                            "stage": "logging",
                        }
                    )
                    self._log_error_reported = True
                self.logger.error("Cannot write experiment log: %s", exc)
                if self.verbose:
                    print(f"[LOG ERROR] {exc}", file=sys.stderr, flush=True)
        return event

    def attempt(self, unit_id, operation, *, on_error="continue", **context):
        if on_error not in ("continue", "raise"):
            raise ValueError("on_error must be continue or raise")
        self.emit("started", unit_id, "Starting computation", **context)
        try:
            value = operation()
        except Exception as exc:
            self.emit("failed", unit_id, str(exc), exception=exc, **context)
            if on_error == "raise":
                raise
            return None
        self.emit("completed", unit_id, "Computation complete", **context)
        return value

    def summary(self):
        counts = {
            status: sum(e["status"] == status for e in self.events)
            for status in ("completed", "failed", "skipped", "cached")
        }
        self.emit("summary", "run", str(counts), **counts)
        return counts
