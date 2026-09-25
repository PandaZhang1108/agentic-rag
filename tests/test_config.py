"""配置分工测试：建索引不拿密钥，聊天服务缺密钥必须立刻失败。"""

import pytest

from config import Settings


def test_index_settings_do_not_require_agent_secrets():
    """只使用索引参数时，Settings 应该能够创建成功。"""
    settings = Settings(
        _env_file=None,
        postgres_url="",
        deepseek_api_key="",
        tavily_api_key="",
        api_key="",
    )

    assert settings.chunk_size == 500


def test_agent_runtime_rejects_missing_secrets():
    """真正启动聊天服务时，四张“门票”缺一张都不允许通过。"""
    settings = Settings(
        _env_file=None,
        postgres_url="",
        deepseek_api_key="",
        tavily_api_key="",
        api_key="",
    )

    with pytest.raises(ValueError, match="POSTGRES_URL.*DEEPSEEK_API_KEY.*TAVILY_API_KEY.*API_KEY"):
        settings.require_agent_runtime()


def test_agent_runtime_accepts_complete_settings():
    """四项都存在时，验票函数正常结束，不应该抛异常。"""
    settings = Settings(
        _env_file=None,
        postgres_url="postgresql://example",
        deepseek_api_key="deepseek-test",
        tavily_api_key="tavily-test",
        api_key="api-test",
    )

    settings.require_agent_runtime()
