from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app import create_app
from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.entity import PredictionConfig

MAX_CHARS = 50


class FakePipeline:
    """Stands in for PredictionPipeline: no model is loaded, and calls are recorded."""

    def __init__(self, config: PredictionConfig) -> None:
        self.calls: list[str] = []

    def summarize(self, dialogue: str) -> str:
        self.calls.append(dialogue)
        return "Amanda baked cookies for Jerry."


@pytest.fixture
def client():
    config = replace(ConfigurationManager(smoke=False).get_prediction_config(), max_input_chars=MAX_CHARS)
    with TestClient(create_app(config, FakePipeline)) as client:  # `with` runs the startup (lifespan) code
        yield client


def test_health(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["max_input_chars"] == MAX_CHARS


def test_summarize_returns_the_pipeline_summary(client: TestClient) -> None:
    response = client.post("/summarize", json={"dialogue": "  Amanda: Cookies?\nJerry: Sure!  "})

    assert response.status_code == 200
    assert response.json() == {"summary": "Amanda baked cookies for Jerry."}
    # Surrounding whitespace is stripped; line breaks inside the chat are kept.
    assert client.app.state.pipeline.calls == ["Amanda: Cookies?\nJerry: Sure!"]


@pytest.mark.parametrize(
    "body",
    [
        {"dialogue": ""},
        {"dialogue": "   \n  "},
        {"dialogue": "x" * (MAX_CHARS + 1)},
        {},
        {"dialogue": 123},
    ],
    ids=["empty", "whitespace", "too-long", "missing-field", "not-a-string"],
)
def test_invalid_input_is_rejected_with_422(client: TestClient, body: dict) -> None:
    response = client.post("/summarize", json=body)

    assert response.status_code == 422
    assert client.app.state.pipeline.calls == []  # the model is never called for bad input


def test_input_at_the_limit_is_accepted(client: TestClient) -> None:
    assert client.post("/summarize", json={"dialogue": "x" * MAX_CHARS}).status_code == 200


def test_index_serves_the_web_page(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<textarea" in response.text
