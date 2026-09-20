import pytest

from config import Settings


def test_index_settings_do_not_require_agent_secrets():

    settings = Settings(
        _env_file=None,
        postgres_url="",
        deepseek_api_key="",
        tavily_api_key="",
        api_key="",
    )

    assert settings.chunk_size == 500


def test_agent_runtime_rejects_missing_secrets():

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

    settings = Settings(
        _env_file=None,
        postgres_url="postgresql://example",
        deepseek_api_key="deepseek-test",
        tavily_api_key="tavily-test",
        api_key="api-test",
    )

    settings.require_agent_runtime()
