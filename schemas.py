import re

from pydantic import BaseModel, Field, field_validator

_THREAD_ID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="用户输入的问题，1-4000 字符",
    )
    thread_id: str | None = Field(default=None)

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, v: str) -> str:

        if not v.strip():
            raise ValueError("message 不能是空白内容")
        return v

    @field_validator("thread_id")
    @classmethod
    def thread_id_format(cls, v: str | None) -> str | None:
        if v is not None and not _THREAD_ID_PATTERN.match(v):
            raise ValueError("thread_id 格式不合法，应为标准 UUID 格式")
        return v
