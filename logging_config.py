"""
================================================================================
logging_config.py —— 结构化日志
================================================================================
之前的问题：全项目只有零散的 print(...)——没法按请求追踪、没法按级别过滤、
更没法接入真正的日志系统（比如 ELK、Datadog 这类，它们都指望日志是
一行一个 JSON，而不是随手写的中文提示字符串）。

这里做两件事：
    1. 把日志输出成一行一个 JSON（生产环境的日志系统基本都靠这个格式解析）
    2. 用 contextvars 存一个 request_id，让"同一次请求"触发的所有日志行，
       都能通过这个 ID 串起来——问题排查时，你能一眼看出
       "这几行日志是不是同一个用户这次请求打出来的"，而不是在一堆
       并发请求的日志里根本分不清谁是谁。
================================================================================
"""

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
