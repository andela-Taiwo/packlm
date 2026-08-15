"""Structured logging to stderr.

stdout belongs to the documented output (step lines and ``wrote <out>``), so
every log record goes to stderr. Configure with environment variables:

- ``LM_LOG_LEVEL``: DEBUG, INFO, WARNING (default) or ERROR
- ``LM_LOG_FORMAT``: ``json`` (default, one object per line) or ``text``
"""

import json
import logging
import os
import sys
from datetime import UTC, datetime

_configured = False


class JsonFormatter(logging.Formatter):
    def format(self, record):
        entry = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": record.getMessage(),
            **getattr(record, "fields", {}),
        }
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


class TextFormatter(logging.Formatter):
    def format(self, record):
        fields = " ".join(f"{k}={v}" for k, v in getattr(record, "fields", {}).items())
        return f"{record.levelname.lower():<7} {record.name} {record.getMessage()} {fields}".rstrip()


def _configure():
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stderr)
    fmt = os.environ.get("LM_LOG_FORMAT", "json").lower()
    handler.setFormatter(TextFormatter() if fmt == "text" else JsonFormatter())
    root = logging.getLogger("lm")
    root.addHandler(handler)
    root.setLevel(os.environ.get("LM_LOG_LEVEL", "WARNING").upper())
    root.propagate = False
    _configured = True


def get_logger(name):
    _configure()
    return logging.getLogger(name)


def event(logger, level, name, **fields):
    """Log ``name`` with ``fields`` attached as structured data."""
    logger.log(level, name, extra={"fields": fields})
