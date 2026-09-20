FASTAPI_SOURCES = {
    "first_steps": "https://fastapi.tiangolo.com/zh/tutorial/first-steps/",
    "path_params": "https://fastapi.tiangolo.com/zh/tutorial/path-params/",
    "query_params": "https://fastapi.tiangolo.com/zh/tutorial/query-params/",
    "request_body": "https://fastapi.tiangolo.com/zh/tutorial/body/",
    "response_model": "https://fastapi.tiangolo.com/zh/tutorial/response-model/",
    "handling_errors": "https://fastapi.tiangolo.com/zh/tutorial/handling-errors/",
    "dependencies": "https://fastapi.tiangolo.com/zh/tutorial/dependencies/",
    "global_dependencies": "https://fastapi.tiangolo.com/zh/tutorial/dependencies/global-dependencies/",
    "middleware": "https://fastapi.tiangolo.com/zh/tutorial/middleware/",
    "testing": "https://fastapi.tiangolo.com/zh/tutorial/testing/",
    "lifespan": "https://fastapi.tiangolo.com/zh/advanced/events/",
    "async_tests": "https://fastapi.tiangolo.com/zh/advanced/async-tests/",
    "sse": "https://fastapi.tiangolo.com/zh/tutorial/server-sent-events/",
    "security_first_steps": "https://fastapi.tiangolo.com/zh/tutorial/security/first-steps/",
}

SOURCE_URLS = list(FASTAPI_SOURCES.values())
SOURCE_ID_BY_URL = {url: source_id for source_id, url in FASTAPI_SOURCES.items()}
