"""JSON logging with request-scoped correlation identifiers."""

import contextvars
import json
import logging
import sys
import uuid

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class JSONFormatter(logging.Formatter):
    """把每一条日志，格式化成一行 JSON 字符串。"""

    _BUILTIN_ATTRS = frozenset(
        {
            "args",
            "asctime",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "message",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
            "taskName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
        }

        for key, value in record.__dict__.items():
            if key in self._BUILTIN_ATTRS or key.startswith("_"):
                continue
            if key in payload:
                key = f"extra_{key}"
            payload[key] = value

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(level: str = "INFO") -> None:
    """在 main.py 启动时调用一次，把 root logger 换成结构化输出。"""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


def new_request_id() -> str:
    """生成一个短一点的 ID，够用来在日志里区分请求就行，不需要完整 UUID 那么长。"""
    return uuid.uuid4().hex[:12]
