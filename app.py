"""FastAPI service for the dialogue summarizer.

Run from the project root:  uvicorn app:app --port 8080
Then open http://localhost:8080 (web page) or http://localhost:8080/docs (API docs).
"""
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.entity import PredictionConfig
from textSummarizer.logging import logger
from textSummarizer.pipeline.prediction import PredictionPipeline

INDEX_HTML = Path(__file__).parent / "static" / "index.html"


class Summarizer(Protocol):
    def summarize(self, dialogue: str) -> str: ...


class SummaryResponse(BaseModel):
    summary: str


def create_app(config: PredictionConfig, load_pipeline: Callable[[PredictionConfig], Summarizer]) -> FastAPI:
    class SummarizeRequest(BaseModel):
        dialogue: str = Field(max_length=config.max_input_chars, examples=["Amanda: I baked cookies. Do you want some?\nJerry: Sure!"])

        @field_validator("dialogue")
        @classmethod
        def not_blank(cls, dialogue: str) -> str:
            if not dialogue.strip():
                raise ValueError("dialogue must not be empty")
            return dialogue.strip()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load the model once at startup, not per request: loading takes seconds, a summary ~1-2 s.
        app.state.pipeline = load_pipeline(config)
        logger.info("Model loaded, ready to summarize")
        yield

    app = FastAPI(title="Dialogue Summarizer", description="Pegasus fine-tuned on SAMSum.", lifespan=lifespan)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(INDEX_HTML)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": config.model_path, "max_input_chars": config.max_input_chars}

    @app.post("/summarize", response_model=SummaryResponse)
    def summarize(body: SummarizeRequest, request: Request) -> SummaryResponse:
        # A plain `def` runs in FastAPI's thread pool, so slow generation doesn't block other requests.
        return SummaryResponse(summary=request.app.state.pipeline.summarize(body.dialogue))

    return app


app = create_app(ConfigurationManager(smoke=False).get_prediction_config(), PredictionPipeline)
